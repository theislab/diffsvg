"""Slide-level variability (6 vs 11 slides, as in legnini): pooled bins vs a slide-aware model."""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from proto import beta1_block, eval_basis, fit_closed_form, make_basis, shape_metric, slope_test

warnings.filterwarnings("ignore")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
G = 150
ALPHA = 0.05
TAU = 0.1  # slope-equivalent units: a linear change of the between-condition gap of 0.1 over the anchor range
N_BINS = 50
SD_LEVEL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3  # slide random intercept
SD_SLOPE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # slide random linear slope

SCENARIOS = {
    "identical": lambda x: 0 * x,
    "level 0.3": lambda x: 0.3 + 0 * x,
    "tiny slope 0.03": lambda x: 0.03 * (x - 0.5),
    "crossing slope 0.3": lambda x: 0.3 * (x - 0.5),
}


def contiguous_bins(x, groups, Y):
    """Non-overlapping windows; one row per (group, bin) with its cell count."""
    idx = np.clip(np.rint(x * N_BINS).astype(int), 0, N_BINS)
    key = groups * (N_BINS + 1) + idx
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), Y.shape[1]))
    np.add.at(sums, inv, Y)
    return uniq // (N_BINS + 1), (uniq % (N_BINS + 1)) / N_BINS, counts, sums / counts[:, None]


def fit_slide_aware(Yb, Xf, Z_int, Z_slope, w, var_comp=None):
    """Posterior of the fixed effects with slide random intercepts/slopes integrated out.

    var_comp=None estimates (sd_level, sd_slope, sigma) per gene by maximising the marginal likelihood.
    Returns beta_1 means/covs (needs K) via the full posterior.
    """
    X = np.column_stack([Xf, Z_int, Z_slope])
    pf, s = Xf.shape[1], Z_int.shape[1]
    XtWX = X.T @ (X * w[:, None])
    n = len(w)
    means, covs = [], []
    for g in range(Yb.shape[1]):
        y = Yb[:, g]
        XtWy = X.T @ (w * y)
        yWy = float(y @ (w * y))

        def post(lsa, lsu, lsig):
            prec = np.r_[np.ones(pf), np.full(s, np.exp(-2 * lsa)), np.full(s, np.exp(-2 * lsu))]
            sig2 = np.exp(2 * lsig)
            Lam = XtWX / sig2 + np.diag(prec)
            C = np.linalg.cholesky(Lam)
            b = XtWy / sig2
            m = np.linalg.solve(Lam, b)
            # log marginal likelihood (up to constants independent of the variance components)
            logdet = 2 * np.log(np.diag(C)).sum() - np.log(prec).sum() + n * np.log(sig2) - np.log(w).sum()
            quad = yWy / sig2 - b @ m
            return -0.5 * (logdet + quad), m, Lam

        if var_comp is None:
            res = optimize.minimize(lambda t: -post(*t)[0], x0=np.log([0.2, 0.05, 1.0]), method="Nelder-Mead",
                                    options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
            lsa, lsu, lsig = res.x
        else:
            lsa, lsu, lsig = np.log(var_comp)
        _, m, Lam = post(lsa, lsu, lsig)
        means.append(m)
        covs.append(np.linalg.inv(Lam))
    return np.array(means), np.array(covs)


def run():
    n_slides = (6, 11)
    sizes = rng.integers(1600, 4300, sum(n_slides))
    cond_of_slide = np.r_[np.zeros(n_slides[0]), np.ones(n_slides[1])]
    xs, cs, ss = [], [], []
    for s_, (n, cc) in enumerate(zip(sizes, cond_of_slide, strict=True)):
        a, b = rng.uniform(1.2, 2.5), rng.uniform(1.5, 4.0)  # slides differ in where their cells sit
        xs.append(np.where(rng.random(n) < 0.85, rng.beta(a, b, n), rng.random(n)))
        cs.append(np.full(n, cc))
        ss.append(np.full(n, s_))
    x, c, sl = np.concatenate(xs), np.concatenate(cs), np.concatenate(ss)
    S = sum(n_slides)

    rows = []
    for name, d in SCENARIOS.items():
        A = rng.uniform(0.2, 1.0, G)
        Bc = rng.uniform(1, 5, G)
        lev = rng.normal(0, SD_LEVEL, (S, G))
        slo = rng.normal(0, SD_SLOPE, (S, G))
        f = A[None] * np.exp(-Bc[None] * x[:, None]) + c[:, None] * d(x)[:, None]
        Y = f + lev[sl] + slo[sl] * (x[:, None] - 0.5) + rng.normal(0, 1, (len(x), G))

        # pooled: contiguous bins per condition, n-weighted (what Stage 2 would give)
        cb, xb, nb, Yb = contiguous_bins(x, c.astype(int), Y)
        cb = cb.astype(float)
        Bm, info = make_basis(xb)
        Qa = shape_metric(info) * 12  # slope-equivalent units
        mean, cov, sig, K = fit_closed_form(Yb, Bm, cb, weights=nb)
        m1, V1 = beta1_block(mean, cov, K)
        _, _, pp_pool, pi_pool = slope_test(m1, V1, Qa, TAU)

        # slide-aware: bins per slide, random intercept + random linear slope per slide
        sb, xb2, nb2, Yb2 = contiguous_bins(x, sl, Y)
        cb2 = cond_of_slide[sb]
        B2 = eval_basis(info, xb2)
        Xf = np.column_stack([np.ones(len(xb2)), B2, cb2, B2 * cb2[:, None]])
        Zi = (sb[:, None] == np.arange(S)[None]).astype(float)
        Zs = Zi * (xb2 - 0.5)[:, None]
        res = {}
        for label, vc in (("oracle", (SD_LEVEL, SD_SLOPE, 1.0)), ("estimated", None)):
            mm, cc_ = fit_slide_aware(Yb2, Xf, Zi, Zs, nb2.astype(float), vc)
            m1s, V1s = mm[:, K + 2: 2 * K + 2], cc_[:, K + 2: 2 * K + 2, K + 2: 2 * K + 2]
            _, _, pp, pi = slope_test(m1s, V1s, Qa, TAU)
            res[label] = (pp, pi)
        rows.append({
            "scenario": name,
            "pooled  tau=0": np.mean(pp_pool < ALPHA), "pooled  shape<=tau": np.mean(pi_pool < ALPHA),
            "slide oracle tau=0": np.mean(res["oracle"][0] < ALPHA), "slide oracle shape<=tau": np.mean(res["oracle"][1] < ALPHA),
            "slide est. tau=0": np.mean(res["estimated"][0] < ALPHA), "slide est. shape<=tau": np.mean(res["estimated"][1] < ALPHA),
        })
    return pd.DataFrame(rows).set_index("scenario")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(f"slide sd: level {SD_LEVEL}, slope {SD_SLOPE}; tau (slope-equivalent) {TAU}")
    print(run().round(3).to_string())
