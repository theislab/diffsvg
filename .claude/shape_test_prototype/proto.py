"""Prototype of a slope-difference (non-parallel curves) test with an interval null.

Closed-form stand-in for diffsvg's per-gene model: same design
(intercept + B beta + c * (kappa + B beta_1)), same N(0, 1) priors, sigma
plugged in. The Gaussian posterior is what NUTS samples for this model, so
the test properties carry over to the PyMC pipeline.
"""

import numpy as np
import patsy
from scipy import integrate, optimize, stats

from diffsvg.pp.utils import build_spline_basis, compute_interior_knots


# ---------------------------------------------------------------- basis
def make_basis(x_fit, n_bsplines=10, degree=3):
    knots = compute_interior_knots(n_bsplines, degree=degree)[1:-1]
    B, info = build_spline_basis(x_fit, knots, degree=degree)
    return B, info


def eval_basis(info, x):
    return np.asarray(patsy.build_design_matrices([info], {"x": np.asarray(x, dtype=float)})[0])


def slope_metric(info, n_grid=2001, h=1e-5):
    """Q with  beta' Q beta = int_0^1 (d/dx B(x) beta)^2 dx  (rank K - 1: constants vanish)."""
    x = np.linspace(h, 1 - h, n_grid)
    dB = (eval_basis(info, x + h) - eval_basis(info, x - h)) / (2 * h)
    return dB.T @ dB / n_grid


def shape_metric(info, n_grid=2001):
    """R with  beta' R beta = int_0^1 (B(x) beta - mean)^2 dx  (centred L2 amplitude, rank K - 1)."""
    x = np.linspace(0, 1, n_grid)
    Bx = eval_basis(info, x)
    Bx = Bx - Bx.mean(0)
    return Bx.T @ Bx / n_grid


# ---------------------------------------------------------------- fit
def fit_closed_form(Y, B, cond, weights=None, n_iter=3):
    """Posterior mean/cov of [intercept, beta, kappa, beta_1] for every column of Y (bins x genes).

    `weights` are relative precisions per bin (1 = homoscedastic, n_obs = sigma^2 / n).
    Returns means (G x P), covs (G x P x P), sigmas (G,).
    """
    n, K = B.shape
    X = np.column_stack([np.ones(n), B, cond, B * cond[:, None]])
    P = X.shape[1]
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    XtWX = X.T @ (X * w[:, None])
    XtWY = X.T @ (Y * w[:, None])
    G = Y.shape[1]
    sig2 = np.full(G, max(np.var(Y), 1e-12))
    for _ in range(n_iter):
        A = XtWX[None] / sig2[:, None, None] + np.eye(P)[None]
        cov = np.linalg.inv(A)
        mean = np.einsum("gij,jg->gi", cov, XtWY / sig2[None, :])
        resid = Y - X @ mean.T
        # residual variance at unit weight, df-corrected by the effective number of parameters
        edf = np.einsum("gij,ji->g", cov, XtWX) / sig2
        sig2 = np.maximum((w[:, None] * resid**2).sum(0) / np.maximum(n - edf, 1), 1e-12)
    A = XtWX[None] / sig2[:, None, None] + np.eye(P)[None]
    cov = np.linalg.inv(A)
    mean = np.einsum("gij,jg->gi", cov, XtWY / sig2[None, :])
    return mean, cov, np.sqrt(sig2), K


def beta1_block(mean, cov, K):
    s = slice(K + 2, 2 * K + 2)
    return mean[:, s], cov[:, s, s]


# ---------------------------------------------------------------- current test (as in fit_model)
def current_wald(mean, cov, sigma, K, B_pred):
    """pattern_wald_test on posterior-predictive draws: Cov(delta) = curve cov + 2 sigma^2 I, df = n_grid."""
    M = B_pred.shape[0]
    Pd = np.column_stack([np.ones(M), B_pred])  # rows: kappa + B beta_1 at the grid
    s = slice(K + 1, 2 * K + 2)
    mu = mean[:, s] @ Pd.T
    out_w, out_p = [], []
    for g in range(mean.shape[0]):
        Sig = Pd @ cov[g, s, s] @ Pd.T + 2 * sigma[g] ** 2 * np.eye(M)
        W = float(mu[g] @ np.linalg.solve(Sig, mu[g]))
        out_w.append(W)
        out_p.append(stats.chi2.sf(W, M))
    return np.array(out_w), np.array(out_p)


# ---------------------------------------------------------------- generalized chi-square tail
def gchisq_sf_imhof(x, lam, nc):
    """P(sum_i lam_i * chi2_1(nc_i) > x) by Imhof's (1961) inversion."""
    lam, nc = np.asarray(lam, float), np.asarray(nc, float)

    def f(u):
        lu = lam * u
        theta = 0.5 * np.sum(np.arctan(lu) + nc * lu / (1 + lu**2)) - 0.5 * x * u
        logrho = np.sum(0.25 * np.log1p(lu**2) + 0.5 * nc * lu**2 / (1 + lu**2))
        return np.sin(theta) / (u * np.exp(logrho))

    val, _ = integrate.quad(f, 0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-10)
    return 0.5 + val / np.pi


def gchisq_sf_spa(x, lam, nc):
    """Saddlepoint (Lugannani-Rice) tail of sum_i lam_i * chi2_1(nc_i), accurate far into the tail (Kuonen 1999)."""
    lam, nc = np.asarray(lam, float), np.asarray(nc, float)
    mean = np.sum(lam * (1 + nc))
    if x <= mean:  # SPA is for the upper tail; the bulk is irrelevant for small p and Imhof is exact there
        return gchisq_sf_imhof(x, lam, nc)

    def K(s):
        return np.sum(-0.5 * np.log1p(-2 * s * lam) + nc * lam * s / (1 - 2 * s * lam))

    def K1(s):
        d = 1 - 2 * s * lam
        return np.sum(lam / d + nc * lam / d**2)

    def K2(s):
        d = 1 - 2 * s * lam
        return np.sum(2 * lam**2 / d**2 + 4 * nc * lam**2 / d**3)

    hi = 1 / (2 * lam.max())
    s = optimize.brentq(lambda s: K1(s) - x, 0, hi * (1 - 1e-12))
    w = np.sign(s) * np.sqrt(2 * (s * x - K(s)))
    v = s * np.sqrt(K2(s))
    return float(stats.norm.sf(w) + stats.norm.pdf(w) * (1 / v - 1 / w))


# ---------------------------------------------------------------- the new test
def slope_test(b1_mean, b1_cov, Q, tau=0.0):
    """Wald test of parallel curves (tau = 0) or of RMS slope difference <= tau (interval null).

    Returns E_hat (RMS slope difference estimate), W (Wald, df = K - 1), p_point, p_interval.
    """
    evals, evecs = np.linalg.eigh(Q)
    keep = evals > evals.max() * 1e-10
    L = evecs[:, keep] * np.sqrt(evals[keep])  # Q = L L'
    out = []
    for m, V in zip(b1_mean, b1_cov, strict=True):
        z = L.T @ m
        Om = L.T @ V @ L
        E2 = float(z @ z)
        W = float(z @ np.linalg.solve(Om, z))
        p_point = stats.chi2.sf(W, keep.sum())
        if tau > 0:
            om = np.linalg.eigvalsh(Om)[::-1]  # descending: least favourable = mean on the top eigenvector
            nc = np.zeros_like(om)
            nc[0] = tau**2 / om[0]
            p_int = min(1.0, max(gchisq_sf_spa(E2, om, nc), 0.0))
        else:
            p_int = p_point
        out.append((np.sqrt(E2), W, p_point, p_int))
    return np.array(out).T


# ---------------------------------------------------------------- binning (as diffsvg.pp.bin_counts)
def bin_means(x, c, Y, n_bins=50, window_radius=0.02):
    """Windowed means for cells (x, c) and expression Y (cells x genes). Returns centers, cond, n_obs, Yb."""
    centers = np.linspace(0, 1, n_bins + 1)
    rows_x, rows_c, rows_n, rows_y = [], [], [], []
    for cc in (0.0, 1.0):
        m_c = c == cc
        for ctr in centers:
            m = m_c & (x >= max(0, ctr - window_radius)) & (x <= min(1, ctr + window_radius))
            n = int(m.sum())
            if n == 0:
                continue
            rows_x.append(ctr)
            rows_c.append(cc)
            rows_n.append(n)
            rows_y.append(Y[m].mean(0))
    return np.array(rows_x), np.array(rows_c), np.array(rows_n), np.vstack(rows_y)


def gchisq_sf_scaled(x, lam, tau2, nu):
    """Interval-null tail when the covariance scale is estimated with `nu` df (fiducial average over the scale).

    T = sum_i lam_i chi2_1(nc_i) with the true covariance lam / u, u ~ chi2_nu / nu, and the boundary mean
    (norm tau) on the top eigenvector. At tau = 0 with equal lam this is exactly the F test.
    """
    lam = np.asarray(lam, float)
    lam1 = lam.max()

    def tail_given_u(u):
        nc = np.zeros_like(lam)
        nc[np.argmax(lam)] = u * tau2 / lam1
        return gchisq_sf_spa(x * u, lam, nc) if x * u > 0 else 1.0

    dens = lambda u: stats.chi2.pdf(u * nu, nu) * nu
    lo, hi = stats.chi2.ppf([1e-10, 1 - 1e-10], nu) / nu
    val, _ = integrate.quad(lambda u: tail_given_u(u) * dens(u), lo, hi, limit=200, epsabs=1e-14)
    return min(max(val, 0.0), 1.0)


def slope_test_small_sample(b1_mean, b1_cov, Q, tau, nu):
    """As `slope_test`, but references that account for a covariance estimated from `nu` df (F at tau = 0)."""
    evals, evecs = np.linalg.eigh(Q)
    keep = evals > evals.max() * 1e-10
    L = evecs[:, keep] * np.sqrt(evals[keep])
    k = keep.sum()
    out = []
    for m, V in zip(b1_mean, b1_cov, strict=True):
        z = L.T @ m
        Om = L.T @ V @ L
        E2 = float(z @ z)
        W = float(z @ np.linalg.solve(Om, z))
        p_F = stats.f.sf(W / k, k, nu)
        p_int = gchisq_sf_scaled_fast(E2, np.linalg.eigvalsh(Om), tau**2, nu) if tau > 0 else p_F
        out.append((np.sqrt(E2), W, p_F, p_int))
    return np.array(out).T


def gchisq_sf_lr(x, lam, nc):
    """Lugannani-Rice saddlepoint tail of sum_i lam_i chi2_1(nc_i), both tails (no numerical integration)."""
    lam, nc = np.asarray(lam, float), np.asarray(nc, float)
    if x <= 0:
        return 1.0
    mean = np.sum(lam * (1 + nc))
    sd = np.sqrt(np.sum(2 * lam**2 * (1 + 2 * nc)))

    def K(s):
        return np.sum(-0.5 * np.log1p(-2 * s * lam) + nc * lam * s / (1 - 2 * s * lam))

    def K1(s):
        d = 1 - 2 * s * lam
        return np.sum(lam / d + nc * lam / d**2)

    def K2(s):
        d = 1 - 2 * s * lam
        return np.sum(2 * lam**2 / d**2 + 4 * nc * lam**2 / d**3)

    if abs(x - mean) < 1e-3 * sd:
        return float(stats.norm.sf((x - mean) / sd))
    if x > mean:
        lo, hi = 0.0, (1 - 1e-12) / (2 * lam.max())
    else:
        lo, hi = -1.0 / lam.max(), 0.0
        while K1(lo) > x:
            lo *= 2
    s = optimize.brentq(lambda s: K1(s) - x, lo, hi, xtol=1e-14)
    w = np.sign(s) * np.sqrt(max(2 * (s * x - K(s)), 0.0))
    v = s * np.sqrt(K2(s))
    if abs(w) < 1e-6 or abs(v) < 1e-12:
        return float(stats.norm.sf((x - mean) / sd))
    return float(min(max(stats.norm.sf(w) + stats.norm.pdf(w) * (1 / v - 1 / w), 0.0), 1.0))


_GL_T, _GL_W = np.polynomial.legendre.leggauss(96)


def gchisq_sf_scaled_fast(x, lam, tau2, nu):
    """`gchisq_sf_scaled` with fixed Gauss-Legendre nodes on log u (96 saddlepoint evaluations)."""
    lam = np.asarray(lam, float)
    top = np.argmax(lam)
    a, b = np.log(stats.chi2.ppf([1e-13, 1 - 1e-13], nu) / nu)
    t = 0.5 * (b - a) * _GL_T + 0.5 * (a + b)
    u = np.exp(t)
    dens = stats.chi2.pdf(u * nu, nu) * nu * u  # includes the du = u dt Jacobian
    tails = np.empty_like(u)
    for i, ui in enumerate(u):
        nc = np.zeros_like(lam)
        nc[top] = ui * tau2 / lam[top]
        tails[i] = gchisq_sf_lr(x * ui, lam, nc)
    return float(min(max(0.5 * (b - a) * np.sum(_GL_W * dens * tails), 0.0), 1.0))
