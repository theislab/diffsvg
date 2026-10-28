"""Prototype: empirical-Bayes and data-driven slide scales for ADVI, on the simulated slides of tests/test_samples.py."""

import sys
import time
import warnings

import numpy as np
import pymc as pm
from scipy import optimize
from scipy.special import logsumexp  # noqa: F401

sys.path.insert(0, "/home/lehnerl/Arbeit/github/HiWi/diffsvg/tests")
warnings.filterwarnings("ignore")
import diffsvg as dvg  # noqa: E402
from diffsvg.tl import model as M  # noqa: E402
from test_samples import _slides  # noqa: E402

K = 10
SCALES = {"sample_level_sd_log__": "sample_level_z", "sample_curve_sd_log__": "sample_curve_z"}


def layout(density):
    """Index arrays of every named block in the flat vector."""
    out, off = {}, 0
    for name, _, size, _ in density.start.point_map_info:
        out[name] = np.arange(off, off + size)
        off += size
    return out


def precision(density, x, free, step=1e-6):
    P = np.empty((len(free), len(free)))
    for a, j in enumerate(free):
        h = np.zeros(len(x))
        h[j] = step * max(1.0, abs(x[j]))
        P[:, a] = (density.dlogp(x - h)[free] - density.dlogp(x + h)[free]) / (2 * h[j])
    return (P + P.T) / 2


def mode(density, x0, free, newton_steps=3):
    """Maximise over the `free` coordinates, the rest held at x0. Returns (x, precision over free)."""
    x = x0.copy()

    def neg(v):
        x[free] = v
        return -density.logp(x), -density.dlogp(x)[free]

    res = optimize.minimize(neg, x0[free], jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8})
    x[free] = res.x
    for _ in range(newton_steps):
        P = precision(density, x, free)
        prop = x.copy()
        prop[free] = x[free] + np.linalg.solve(P, density.dlogp(x)[free])
        if not density.logp(prop) >= density.logp(x):
            break
        x = prop
    return x, precision(density, x, free)


def eb_scales(density, h0, rounds=5, tol=0.01, bounds=(np.log(1e-3), np.log(10.0))):
    """Log-scales maximising the Laplace-approximated marginal posterior (prior and Jacobian of log-scale included)."""
    lay = layout(density)
    hyper = np.concatenate([lay[n] for n in SCALES])
    free = np.setdiff1d(np.arange(len(density.start.data)), hyper)
    pos = {j: a for a, j in enumerate(free)}
    zidx = {n: np.array([pos[j] for j in lay[z]]) for n, z in SCALES.items()}
    x = density.start.data.copy()
    x[hyper] = h0
    history = []
    for _ in range(rounds):
        x, H = mode(density, x, free)
        h = x[hyper].copy()
        s = {n: np.exp(x[lay[n]][0]) for n in SCALES}
        t = np.ones(len(free))  # z = u / s on the z coordinates
        for n in SCALES:
            t[zidx[n]] = 1.0 / s[n]
        z_all = np.zeros(len(free), bool)
        for n in SCALES:
            z_all[zidx[n]] = True
        w = x[free]
        grad_F = density.dlogp(x)[free] + np.where(z_all, w, 0.0)  # drop the N(0, 1) prior of z
        A = (t[:, None] * (H - np.diag(z_all.astype(float)))) * t[None, :]
        b = t * grad_F
        v_hat = w / t
        # hyperprior (+ Jacobian) differences: with z = 0 the likelihood does not see the scales
        x_z0 = x.copy()
        x_z0[free[z_all]] = 0.0

        def objective(hh):
            xx = x_z0.copy()
            xx[hyper] = hh
            prior = density.logp(xx)
            d = np.zeros(len(free))
            for n, hn in zip(SCALES, hh):
                d[zidx[n]] = np.exp(-2 * hn)
            Mm = A + np.diag(d)
            try:
                C = np.linalg.cholesky(Mm)
            except np.linalg.LinAlgError:
                return 1e100
            delta = np.linalg.solve(Mm, b - d * v_hat)
            v = v_hat + delta
            G = b @ delta - 0.5 * delta @ A @ delta - 0.5 * (d * v * v).sum() - sum(len(zidx[n]) * hn for n, hn in zip(SCALES, hh))
            return -(G - np.log(np.diag(C)).sum() + prior)

        res = optimize.minimize(objective, h, method="L-BFGS-B", bounds=[bounds] * len(hyper))
        history.append(np.exp(res.x))
        # re-expand at the new scales: take the mode of the quadratic model there, z = u / s'
        d_new = np.zeros(len(free))
        for n, hn in zip(SCALES, res.x):
            d_new[zidx[n]] = np.exp(-2 * hn)
        v_star = v_hat + np.linalg.solve(A + np.diag(d_new), b - d_new * v_hat)
        new_t = np.ones(len(free))
        for n, hn in zip(SCALES, res.x):
            new_t[zidx[n]] = np.exp(-hn)
        x[free] = v_star * new_t
        x[hyper] = res.x
        if np.max(np.abs(res.x - h)) < tol:
            break
    return np.exp(x[hyper]), history


def guess_scales(y, B, condition, slide, n_slides, weights=None):
    """Rough slide scales from pooled least-squares residuals: RMS of per-slide level and curve coefficients."""
    w = np.ones(len(y)) if weights is None else weights
    X = np.column_stack([np.ones(len(y)), B, condition, B * condition[:, None]])
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    r = y - X @ coef
    levels, curves = [], []
    for j in range(n_slides):
        m = slide == j
        Z = np.column_stack([np.ones(m.sum()), B[m]])
        c = np.linalg.solve((Z * w[m, None]).T @ Z + 1e-3 * np.eye(K + 1), (Z * w[m, None]).T @ r[m])
        levels.append(c[0])
        curves.append(c[1:])
    return max(np.sqrt(np.mean(np.square(levels))), 1e-2), max(np.sqrt(np.mean(np.square(curves))), 1e-2)


if __name__ == "__main__":
    adata = _slides()
    b = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20, window_radius=None, sample="slide", inplace=False)
    weights, ss, _ = M._weighting(b, True)
    y = b.Y["g"]
    build = lambda scales=None: M._build_gene_model(  # noqa: E731
        y, b.B, b.B_masked, b.condition_fit, K, b.intercept_initvals["g"], weights=weights, ss_within=ss["g"],
        sample_index=b.sample_index, n_samples=b.n_samples, record_mu=False,
    )
    Q = dvg.tl.shape_metric(K, 3)
    test = lambda d: dvg.tl.parallel_curves_test(d, Q, tau=0.0, df_denominator=8)  # noqa: E731

    guess = guess_scales(y, b.B, b.condition_fit, b.sample_index, b.n_samples, weights)
    print(f"data-driven guess: level {guess[0]:.3f} curve {guess[1]:.3f}")

    model = build()
    density = M._FlatDensity(model)
    t = time.time()
    s_eb, hist = eb_scales(density, np.log(guess))
    print(f"EB: level {s_eb[0]:.3f} curve {s_eb[1]:.3f} in {time.time() - t:.1f}s, rounds {[np.round(h, 3).tolist() for h in hist]}")
    print("NUTS reference (earlier run): level 0.401 [0.254, 0.632], curve 0.546 [0.476, 0.630]; W 4.60, p 0.83")
