"""Which statistic for the interval null? Power at matched calibration, contiguous n-weighted bins."""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from proto import beta1_block, fit_closed_form, make_basis, shape_metric, gchisq_sf_lr
from sim_slides import contiguous_bins
warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)
G, TAU = 300, 0.1
SC = {"identical": lambda x: 0 * x, "tiny slope 0.03": lambda x: 0.03 * (x - .5),
      "crossing 0.15": lambda x: 0.15 * (x - .5), "crossing 0.3": lambda x: 0.3 * (x - .5),
      "peak shift": lambda x: 0.5 * np.exp(-3 * np.abs(x - .15)) - 0.5 * np.exp(-3 * x)}
pos = lambda n: np.where(rng.random(n) < 0.8, rng.beta(1.5, 2.5, n), rng.random(n))
for n_cells in (5000, 20000):
    x = np.r_[pos(n_cells), pos(n_cells)]; c = np.r_[np.zeros(n_cells), np.ones(n_cells)]
    rows = []
    for name, d in SC.items():
        A, Bc = rng.uniform(.2, 1, G), rng.uniform(1, 5, G)
        Y = A[None] * np.exp(-Bc[None] * x[:, None]) + c[:, None] * d(x)[:, None] + rng.normal(0, 1, (2 * n_cells, G))
        cb, xb, nb, Yb = contiguous_bins(x, c.astype(int), Y)
        B, info = make_basis(xb)
        Q = shape_metric(info) * 12
        mean, cov, sig, K = fit_closed_form(Yb, B, cb.astype(float), weights=nb)
        m1, V1 = beta1_block(mean, cov, K)
        ev, U = np.linalg.eigh(Q); keep = ev > ev.max() * 1e-10; L = U[:, keep] * np.sqrt(ev[keep]); k = keep.sum()
        res = {k_: [] for k_ in ("W, tau=0", "E, tau=0", "E, tau=0.1", "W, tau=0.1 (lf = most precise dir)")}
        for m, V in zip(m1, V1):
            z = L.T @ m; Om = L.T @ V @ L; om = np.linalg.eigvalsh(Om)[::-1]
            W = z @ np.linalg.solve(Om, z); E2 = z @ z
            res["W, tau=0"].append(stats.chi2.sf(W, k))
            res["E, tau=0"].append(gchisq_sf_lr(E2, om, np.zeros(k)))
            res["E, tau=0.1"].append(gchisq_sf_lr(E2, om, np.r_[TAU**2 / om[0], np.zeros(k - 1)]))
            res["W, tau=0.1 (lf = most precise dir)"].append(stats.ncx2.sf(W, k, TAU**2 / om[-1]))
        rows.append({"scenario": name} | {k_: np.mean(np.array(v) < 0.05) for k_, v in res.items()})
    print(f"\n=== {n_cells} cells/condition, contiguous n-weighted bins, rejection at 0.05 ===")
    print(pd.DataFrame(rows).set_index("scenario").round(3).to_string())
