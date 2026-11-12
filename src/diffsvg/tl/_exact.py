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
    """Whether the exact posterior applies: Gaussian likelihood, normal coefficient priors, no slide effects."""
    coefficient_priors = ("intercept", "beta", "beta_1", "categorical_beta")
    return (
        likelihood == "normal"
        and not has_samples
        and all(priors[name][0] == "normal" for name in coefficient_priors)
        and priors["sigma"][0] in _SIGMA_LOGPDF
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
    u = post.projected[:, gene][None, :] / (d + s2) + np.sqrt(s2 / (d + s2)) * rng.standard_normal((n_draws, d.shape[1]))
    return post.prior_mean[None, :] + u @ post.transform.T
