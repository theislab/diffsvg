"""Same test on (a) a per-cell fit, (b) contiguous cell-count-weighted bins, (c) today's overlapping bins."""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from proto import bin_means, eval_basis, fit_closed_form, make_basis, shape_metric, slope_test, slope_test_small_sample
from sim_slides import contiguous_bins
from slide_curves import fit_slide_curves
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
rng = np.random.default_rng(11)
TAU = 0.1
effects = {"identical": lambda x: 0 * x, "level 0.3": lambda x: 0.3 + 0 * x,
           "crossing 0.3": lambda x: 0.3 * (x - 0.5), "peak shift": lambda x: 0.5 * np.exp(-3 * np.abs(x - 0.15)) - 0.5 * np.exp(-3 * x)}

# ---- (1) one slide per group: per-cell vs bins
N = 10000
pos = lambda n: np.where(rng.random(n) < 0.8, rng.beta(1.5, 2.5, n), rng.random(n))
x = np.r_[pos(N), pos(N)]; c = np.r_[np.zeros(N), np.ones(N)]
Y = np.column_stack([0.6 * np.exp(-3 * x) + c * d(x) for d in effects.values()]) + rng.normal(0, 1, (2 * N, len(effects)))
Bfull, info = make_basis(np.r_[0.0, x, 1.0])  # pin the boundary knots at 0 and 1
Bc = Bfull[1:-1]
Q = shape_metric(info) * 12
rows = {}
fits = {
    "per cell": (Bc, c, Y, None),
    "contiguous bins, n-weighted": None,
    "overlapping bins (today)": None,
}
cb, xb, nb, Yb = contiguous_bins(x, c.astype(int), Y)
fits["contiguous bins, n-weighted"] = (eval_basis(info, xb), cb.astype(float), Yb, nb)
xo, co, no, Yo = bin_means(x, c, Y, window_radius=0.02)
fits["overlapping bins (today)"] = (eval_basis(info, xo), co, Yo, None)
out = []
for label, (B, cc, YY, w) in fits.items():
    mean, cov, sig, K = fit_closed_form(YY, B, cc, weights=w)
    s_ = slice(K + 2, 2 * K + 2)
    E, W, p0, pt = slope_test(mean[:, s_], cov[:, s_, s_], Q, TAU)
    for g, name in enumerate(effects):
        out.append({"fit": label, "gene": name, "rows": len(YY), "sigma": sig[g], "E": E[g], "W": W[g], "p tau=0": p0[g], "p tau=0.1": pt[g]})
print("=== one slide per group, 10k cells each ===")
print(pd.DataFrame(out).set_index(["gene", "fit"]).sort_index(level=0, sort_remaining=False).round(4).to_string())

# ---- (2) replicates: per-cell vs per-slide bins, random intercept + random curve per slide
S0, S1 = 6, 11
cond_of_slide = np.r_[np.zeros(S0), np.ones(S1)]
xs, ss = [], []
for s_i in range(S0 + S1):
    n = rng.integers(1600, 4300)
    a, b = rng.uniform(1.2, 2.5), rng.uniform(1.5, 4.0)
    xs.append(np.where(rng.random(n) < 0.85, rng.beta(a, b, n), rng.random(n))); ss.append(np.full(n, s_i))
x, sl = np.concatenate(xs), np.concatenate(ss)
c = cond_of_slide[sl]
lev = rng.normal(0, 0.3, (S0 + S1, len(effects))); slo = rng.normal(0, 0.1, (S0 + S1, len(effects)))
Y = np.column_stack([0.6 * np.exp(-3 * x) + c * d(x) for d in effects.values()]) + lev[sl] + slo[sl] * (x[:, None] - 0.5) + rng.normal(0, 1, (len(x), len(effects)))
_, info = make_basis(np.r_[0.0, x, 1.0])
Q = shape_metric(info) * 12
nu = S0 + S1 - 2
out = []
sb, xb2, nb2, Yb2 = contiguous_bins(x, sl, Y)
for label, (xx, sidx, YY, w) in {"per cell": (x, sl, Y, np.ones(len(x))), "per-slide bins, n-weighted": (xb2, sb, Yb2, nb2.astype(float))}.items():
    B = eval_basis(info, xx)
    cc = cond_of_slide[sidx]
    Xf = np.column_stack([np.ones(len(xx)), B, cc, B * cc[:, None]])
    Zi = (sidx[:, None] == np.arange(S0 + S1)[None]).astype(float)
    mm, cv, par = fit_slide_curves(YY, Xf, Zi, B, sidx, w)
    K = B.shape[1]; s_ = slice(K + 2, 2 * K + 2)
    E, W, pF, pt = slope_test_small_sample(mm[:, s_], cv[:, s_, s_], Q, TAU, nu)
    for g, name in enumerate(effects):
        out.append({"fit": label, "gene": name, "rows": len(YY), "sd_slide_level": par[g, 0], "sd_slide_curve": par[g, 1], "sigma": par[g, 2],
                    "E": E[g], "W": W[g], "p tau=0 (F)": pF[g], "p tau=0.1": pt[g]})
print(f"\n=== {S0} vs {S1} slides ({len(x)} cells), random intercept + random curve per slide, nu = {nu} ===")
print(pd.DataFrame(out).set_index(["gene", "fit"]).sort_index(level=0, sort_remaining=False).round(4).to_string())
