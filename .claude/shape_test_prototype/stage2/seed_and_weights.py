import time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import diffsvg as dvg
from make_data import make_adata
pd.set_option("display.width", 200)

SK = {"draws": 1000, "tune": 500, "chains": 2}
adata = make_adata(n_per_cond=3000)
# --- seeding
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
r1 = dvg.tl.fit_model(adata, setup, inplace=False, progressbar=False, sample_kwargs=SK)
r2 = dvg.tl.fit_model(adata, setup, inplace=False, progressbar=False, sample_kwargs=SK, n_workers=2)
r3 = dvg.tl.fit_model(adata, setup, inplace=False, progressbar=False, sample_kwargs=SK, predictive_seed=123)
print("default seed, rerun with 2 workers: stats identical", r1[0].equals(r2[0]), "| curves identical", all(r1[2][k].equals(r2[2][k]) for k in r1[2]))
diff = [c for c in r1[0].columns if not np.array_equal(r1[0][c], r3[0][c])]
print("predictive_seed=123 changes only:", diff)

# --- weighted contiguous bins vs per cell
x = adata.obsm["design_matrix"]
b = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", window_radius=None, inplace=False)
print("\ncontiguous bins:", len(b.n_obs), "rows,", int(b.n_obs.sum()), "cells binned of", len(x), "| n_obs_grid sums", b.n_obs_grid.sum(axis=1))
setup_w = dvg.tl.DiffSVGModelSetup(binned=b, likelihood="normal")
cols = ["shape_diff", "slope_diff", "parallel_stat", "parallel_pval", "integral", "categorical_beta_mean"]
res = {}
for label, ub in (("weighted bins", True), ("per cell", False)):
    t = time.time()
    s, preds, curves = dvg.tl.fit_model(adata, setup_w, inplace=False, use_binned=ub, progressbar=False, sample_kwargs=SK, test="parallel", n_workers=4)
    res[label] = (s, curves)
    print(f"{label}: {time.time() - t:.0f} s")
print(pd.concat({k: v[0][cols] for k, v in res.items()}, axis=1).round(4).to_string())
cw, cc = res["weighted bins"][1], res["per cell"][1]
print("max |curve mean difference| weighted vs per cell:", float(np.max(np.abs(cw["cond0_mean"].to_numpy() - cc["cond0_mean"].to_numpy()))).__round__(4),
      "| median band width: weighted", float(np.median(cw["cond0_ci_high"] - cw["cond0_ci_low"])).__round__(3), "per cell", float(np.median(cc["cond0_ci_high"] - cc["cond0_ci_low"])).__round__(3))
