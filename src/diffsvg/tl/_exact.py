"""Exact posterior of the Gaussian GAM, for all genes at once.

The default model is linear-Gaussian: ``y ~ N(X theta, sigma^2 / w)`` with independent normal priors on
every coefficient in ``theta``. Given ``sigma`` the posterior of ``theta`` is Gaussian (conjugacy), and
``sigma`` is a single number, so it is integrated out numerically on a per-gene grid of ``log sigma``.
The result is the same posterior NUTS samples from, without Monte Carlo error.

All genes share the design ``X`` (and the row weights), so ``X'WX`` and its eigendecomposition are
computed once; per gene only ``X'Wy`` and ``y'Wy`` (and, for weighted contiguous bins, the within-bin sum
of squares) are needed.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import logsumexp

#: ``log p(sigma)`` for the noise priors `_PRIOR_DISTRIBUTIONS` offers, keyed by distribution name.
_SIGMA_LOGPDF = {
    "halfnormal": lambda s, scale: stats.halfnorm.logpdf(s, scale=scale),
    "halfcauchy": lambda s, beta: stats.halfcauchy.logpdf(s, scale=beta),
    "exponential": lambda s, lam: stats.expon.logpdf(s, scale=1.0 / lam),
}


def supported(priors, likelihood, has_samples):
    """Whether the exact posterior applies.

    It needs a Gaussian likelihood and normal coefficient priors (with slides, any one-dimensional positive
    prior on the two slide scales).
    """
    coefficient_priors = ("intercept", "beta", "beta_1", "categorical_beta")
    scales = ("sigma", "sample_level_sd", "sample_curve_sd") if has_samples else ("sigma",)
    return (
        likelihood == "normal"
        and all(priors[name][0] == "normal" for name in coefficient_priors)
        and all(priors[name][0] in _SIGMA_LOGPDF for name in scales)
    )


@dataclass
class ExactPosterior:
    """Posterior of ``theta`` for every gene, ``sigma`` integrated out.

    ``mean`` (G, p) and ``cov`` (G, p, p) are the exact posterior moments; ``sigma2_mean`` is E[sigma^2];
    ``edge_mass`` is the largest quadrature weight at either end of the gene's ``log sigma`` grid (near 0
    means the grid covers the posterior). The remaining fields let `draws` sample the exact posterior.
    """

    mean: np.ndarray
    cov: np.ndarray
    sigma2_mean: np.ndarray
    edge_mass: np.ndarray
    log_sigma_grid: np.ndarray
    grid_weights: np.ndarray
    eigvals: np.ndarray
    transform: np.ndarray  # theta = prior_mean + transform @ u, u in the whitened eigenbasis
    projected: np.ndarray  # (p, G): the whitened, rotated X'W(y - X prior_mean)
    prior_mean: np.ndarray


def sufficient_statistics(X, Y, weights=None):
    """``X'WX`` (p, p), ``X'WY`` (p, G) and ``diag(Y'WY)`` (G,). `Y` is (n, G), dense or scipy.sparse."""
    Xw = X if weights is None else X * weights[:, None]
    XtWX = X.T @ Xw
    XtWY = np.asarray((Y.T @ Xw).T)
    if hasattr(Y, "multiply"):
        squared = Y.multiply(Y)
        yWy = np.asarray(squared.sum(axis=0) if weights is None else squared.T @ weights).ravel()
    else:
        yWy = (Y**2).sum(axis=0) if weights is None else weights @ (Y**2)
    return XtWX, XtWY, np.asarray(yWy, dtype=float)


def posterior(
    XtWX,
    XtWY,
    yWy,
    n_rows,
    prior_mean,
    prior_sd,
    sigma_prior,
    extra_df=None,
    extra_ss=None,
    n_grid=81,
    width=10.0,
    chunk=1024,
):
    """Exact posterior moments for every gene (columns of `XtWY`).

    `sigma_prior` is a ``(distribution, *params)`` tuple. `extra_df`/`extra_ss` add the within-bin term of
    weighted contiguous bins, ``-extra_df * log(sigma) - extra_ss / (2 sigma^2)``, to the likelihood.
    """
    m0 = np.asarray(prior_mean, dtype=float)
    s0 = np.asarray(prior_sd, dtype=float)
    G = XtWY.shape[1]
    logpdf = _SIGMA_LOGPDF[sigma_prior[0]]
    sigma_params = sigma_prior[1:]

    # Shift to a zero prior mean and whiten the prior: theta = m0 + S0 u, u ~ N(0, I).
    r = XtWY - (XtWX @ m0)[:, None]
    yy = yWy - 2.0 * (m0 @ XtWY) + m0 @ XtWX @ m0
    d, U = np.linalg.eigh(s0[:, None] * XtWX * s0[None, :])
    d = np.clip(d, 0.0, None)
    c = U.T @ (s0[:, None] * r)  # (p, G)
    df = np.zeros(G) if extra_df is None else np.broadcast_to(np.asarray(extra_df, dtype=float), (G,))
    ss = np.zeros(G) if extra_ss is None else np.asarray(extra_ss, dtype=float)
    n_tot = n_rows + df

    def log_post(log_s, idx):
        """Unnormalised log p(log sigma | y) on a (g, S) grid for the genes `idx`."""
        s2 = np.exp(2.0 * log_s)
        cg2 = (c[:, idx] ** 2)[:, :, None]
        quad = (cg2 / (s2[None] * (d[:, None, None] + s2[None]))).sum(axis=0)
        logdet = np.log1p(d[:, None, None] / s2[None]).sum(axis=0)
        loglik = -n_tot[idx, None] * log_s - 0.5 * logdet - 0.5 * (yy[idx, None] + ss[idx, None]) / s2 + 0.5 * quad
        return loglik + logpdf(np.exp(log_s), *sigma_params) + log_s  # + log-Jacobian of sigma -> log sigma

    grid = np.empty((G, n_grid))
    weights = np.empty((G, n_grid))
    t = np.linspace(-1.0, 1.0, n_grid)
    coarse = np.linspace(np.log(1e-6), np.log(1e3), 800)
    for start in range(0, G, chunk):
        idx = np.arange(start, min(start + chunk, G))
        lp = log_post(np.tile(coarse, (len(idx), 1)), idx)
        centre = coarse[lp.argmax(axis=1)]
        half = width / np.sqrt(2.0 * np.maximum(n_tot[idx], 2.0))  # sd(log sigma) ~ 1 / sqrt(2 n)
        for _ in range(4):  # recentre on the posterior mean; the coarse step can exceed its sd
            g = centre[:, None] + half[:, None] * t[None]
            lp = log_post(g, idx)
            w = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))  # uniform grid: normalised quadrature weights
            centre = (w * g).sum(axis=1)
        grid[idx], weights[idx] = g, w

    s2 = np.exp(2.0 * grid)  # (G, S)
    shrink = 1.0 / (d[None, :, None] + s2[:, None, :])  # (G, p, S)
    u_mean_s = c.T[:, :, None] * shrink  # E[u | sigma] in the eigenbasis
    u_mean = (u_mean_s * weights[:, None]).sum(axis=-1)
    var_diag = (s2[:, None, :] * shrink * weights[:, None]).sum(axis=-1)  # E[Var(u | sigma)], diagonal
    dev = u_mean_s - u_mean[:, :, None]
    cov_u = np.einsum("gis,gjs,gs->gij", dev, dev, weights)  # Var(E[u | sigma])
    cov_u[:, np.arange(len(d)), np.arange(len(d))] += var_diag
    T = s0[:, None] * U
    return ExactPosterior(
        mean=m0 + u_mean @ T.T,
        cov=T @ cov_u @ T.T,
        sigma2_mean=(s2 * weights).sum(axis=1),
        edge_mass=np.maximum(weights[:, 0], weights[:, -1]),
        log_sigma_grid=grid,
        grid_weights=weights,
        eigvals=d,
        transform=T,
        projected=c,
        prior_mean=m0,
    )


def draws(post, gene, n_draws, rng):
    """``n_draws`` samples of ``theta`` from gene index `gene`'s exact posterior, (n_draws, p)."""
    s_idx = rng.choice(post.grid_weights.shape[1], size=n_draws, p=post.grid_weights[gene])
    s2 = np.exp(2.0 * post.log_sigma_grid[gene, s_idx])[:, None]  # (n, 1)
    d = post.eigvals[None, :]
    u = post.projected[:, gene][None, :] / (d + s2) + np.sqrt(s2 / (d + s2)) * rng.standard_normal(
        (n_draws, d.shape[1])
    )
    return post.prior_mean[None, :] + u @ post.transform.T


# ----------------------------------------------------------------------------------------------------------
# Slide effects: random level and random curve per slide, scales `level_sd`, `curve_sd` with their own priors.
# Given (sigma, level_sd, curve_sd) the model is still linear-Gaussian in (fixed coefficients, slide effects).
# The two scales are integrated on a shared log grid (one eigendecomposition per grid point, shared by all
# genes); sigma per gene and grid point by Gauss-Hermite quadrature around its conditional mode.
# ----------------------------------------------------------------------------------------------------------


def slide_design(X_fixed, B, sample_index, n_samples):
    """Columns for the slide effects: one-hot slide levels, then per slide its copy of the spline basis."""
    n, K = B.shape
    onehot = np.zeros((n, n_samples))
    onehot[np.arange(n), sample_index] = 1.0
    curves = (onehot[:, :, None] * B[:, None, :]).reshape(n, n_samples * K)
    return np.hstack([X_fixed, onehot, curves])


def _log_sigma_mode(c, d, yy, n_tot, logpdf, params, start, steps=30):
    """Mode and curvature of log p(log sigma | y, scales) for every gene (columns of `c`), by damped Newton."""

    def f(t):
        s2 = np.exp(2.0 * t)[None, :]
        val = (
            -n_tot * t
            - 0.5 * np.log1p(d[:, None] / s2).sum(axis=0)
            - 0.5 * yy / s2[0]
            + 0.5 * (c**2 / (s2 * (d[:, None] + s2))).sum(axis=0)
        )
        return val + logpdf(np.exp(t), *params) + t

    t = start.copy()
    h = 1e-4
    for _ in range(steps):
        f0, fp, fm = f(t), f(t + h), f(t - h)
        g1 = (fp - fm) / (2 * h)
        g2 = np.minimum((fp - 2 * f0 + fm) / h**2, -1e-8)
        step = np.clip(-g1 / g2, -0.5, 0.5)
        t = t + step
        if np.abs(step).max() < 1e-7:
            break
    f0, fp, fm = f(t), f(t + h), f(t - h)
    curvature = np.minimum((fp - 2 * f0 + fm) / h**2, -1e-8)
    return t, curvature, f


def slide_posterior(
    X,
    Y,
    weights,
    n_fixed,
    n_samples,
    n_bsplines,
    prior_mean,
    prior_sd,
    sigma_prior,
    level_prior,
    curve_prior,
    extra_df=None,
    extra_ss=None,
    n_draws=1000,
    seeds=None,
    n_hyper=72,
    hyper_range=(1e-5, 5.0),
    n_nodes=9,
):
    """Exact posterior of the fixed coefficients with slide effects integrated out, for every gene.

    `X` is `slide_design`'s output, `prior_mean`/`prior_sd` cover its first `n_fixed` columns. Returns a dict
    with ``mean`` (G, n_fixed), ``cov`` (G, n_fixed, n_fixed), ``draws`` (G, n_draws, n_fixed),
    ``sigma2_mean``, ``level_sd_mean``, ``curve_sd_mean`` and ``edge_mass`` (largest posterior mass on the
    boundary of the scale grid).
    """
    XtWX, XtWY, yWy = sufficient_statistics(X, Y, weights)
    p = X.shape[1]
    G = XtWY.shape[1]
    m = np.zeros(p)
    m[:n_fixed] = prior_mean
    r = XtWY - (XtWX @ m)[:, None]
    yy0 = yWy - 2.0 * (m @ XtWY) + m @ XtWX @ m
    df = np.zeros(G) if extra_df is None else np.broadcast_to(np.asarray(extra_df, dtype=float), (G,))
    ss = np.zeros(G) if extra_ss is None else np.asarray(extra_ss, dtype=float)
    yy = yy0 + ss
    n_tot = X.shape[0] + df
    sig_logpdf, sig_params = _SIGMA_LOGPDF[sigma_prior[0]], sigma_prior[1:]
    lev_logpdf, lev_params = _SIGMA_LOGPDF[level_prior[0]], level_prior[1:]
    cur_logpdf, cur_params = _SIGMA_LOGPDF[curve_prior[0]], curve_prior[1:]

    log_grid = np.linspace(np.log(hyper_range[0]), np.log(hyper_range[1]), n_hyper)
    hyper = [(a, b) for a in log_grid for b in log_grid]
    nodes, node_w = np.polynomial.hermite_e.hermegauss(n_nodes)  # for \int e^{-x^2/2} g(x) dx

    # Pass 1: per scale pair, eigendecomposition and per gene the log-weights of the sigma nodes.
    eig = []
    log_w = np.empty((G, len(hyper), n_nodes))
    t_nodes = np.empty((G, len(hyper), n_nodes))
    start = 0.5 * np.log(np.maximum(yy / np.maximum(n_tot, 1.0), 1e-12))
    for h, (la, lb) in enumerate(hyper):
        s_full = np.concatenate([prior_sd, np.full(n_samples, np.exp(la)), np.full(n_samples * n_bsplines, np.exp(lb))])
        d, U = np.linalg.eigh(s_full[:, None] * XtWX * s_full[None, :])
        d = np.clip(d, 0.0, None)
        c = U.T @ (s_full[:, None] * r)
        eig.append((d, U, s_full))
        mode, curv, f = _log_sigma_mode(c, d, yy, n_tot, sig_logpdf, sig_params, start)
        sd = 1.0 / np.sqrt(-curv)
        t = mode[:, None] + sd[:, None] * nodes[None, :]
        vals = np.stack([f(t[:, k]) for k in range(n_nodes)], axis=1)
        log_w[:, h] = (
            vals + np.log(node_w)[None, :] + 0.5 * nodes[None, :] ** 2 + np.log(sd)[:, None]
            + lev_logpdf(np.exp(la), *lev_params) + la + cur_logpdf(np.exp(lb), *cur_params) + lb
        )  # fmt: skip
        t_nodes[:, h] = t
    log_w -= logsumexp(log_w.reshape(G, -1), axis=1)[:, None, None]
    w = np.exp(log_w)

    # Pass 2: mixture moments of the fixed block.
    M1 = np.zeros((G, n_fixed))
    M2 = np.zeros((G, n_fixed, n_fixed))
    s2_mean = np.zeros(G)
    for h, (d, U, s_full) in enumerate(eig):
        Tf = (s_full[:, None] * U)[:n_fixed]  # (n_fixed, p)
        c = U.T @ (s_full[:, None] * r)  # (p, G)
        s2 = np.exp(2.0 * t_nodes[:, h])  # (G, n_nodes)
        wk = w[:, h]  # (G, n_nodes)
        shrink = 1.0 / (d[None, None, :] + s2[:, :, None])  # (G, n_nodes, p)
        u_mean = c.T[:, None, :] * shrink
        mean = m[:n_fixed] + u_mean @ Tf.T  # (G, n_nodes, n_fixed)
        M1 += (wk[:, :, None] * mean).sum(axis=1)
        M2 += np.einsum("gk,gki,gkj->gij", wk, mean, mean)
        var_w = (wk[:, :, None] * s2[:, :, None] * shrink).sum(axis=1)  # (G, p): sum_k w Var(u | .)
        M2 += (Tf[None] * var_w[:, None, :]) @ Tf.T
        s2_mean += (wk * s2).sum(axis=1)
    cov = M2 - M1[:, :, None] * M1[:, None, :]

    # Draws of the fixed block from the mixture.
    flat = w.reshape(G, -1)
    out_draws = np.empty((G, n_draws, n_fixed))
    seeds = seeds if seeds is not None else np.random.SeedSequence(0).spawn(G)
    for g in range(G):
        rng = np.random.default_rng(seeds[g])
        idx = rng.choice(flat.shape[1], size=n_draws, p=flat[g] / flat[g].sum())
        hs, ks = np.divmod(idx, n_nodes)
        for h in np.unique(hs):
            sel = np.flatnonzero(hs == h)
            d, U, s_full = eig[h]
            Tf = (s_full[:, None] * U)[:n_fixed]
            cg = U.T @ (s_full * r[:, g])
            s2 = np.exp(2.0 * t_nodes[g, h, ks[sel]])[:, None]
            u = cg[None, :] / (d[None, :] + s2) + np.sqrt(s2 / (d[None, :] + s2)) * rng.standard_normal((len(sel), p))
            out_draws[g, sel] = m[:n_fixed] + u @ Tf.T

    a_idx = np.array([np.exp(a) for a, _ in hyper])
    b_idx = np.array([np.exp(b) for _, b in hyper])
    wh = w.sum(axis=2)  # (G, H)
    on_edge = np.array([(a in (log_grid[0], log_grid[-1])) or (b in (log_grid[0], log_grid[-1])) for a, b in hyper])
    return {
        "mean": M1,
        "cov": cov,
        "draws": out_draws,
        "sigma2_mean": s2_mean,
        "level_sd_mean": wh @ a_idx,
        "curve_sd_mean": wh @ b_idx,
        "edge_mass": wh[:, on_edge].sum(axis=1),
    }
