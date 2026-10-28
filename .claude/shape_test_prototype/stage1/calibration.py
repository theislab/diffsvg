"""Calibration through the real PyMC pipeline: rejection rates of the parallel test, per cell vs binned."""
import warnings
import anndata as ad
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import diffsvg as dvg

rng = np.random.default_rng(21)
n = 3000
x = np.where(rng.random(2 * n) < 0.8, rng.beta(1.5, 2.5, 2 * n), rng.random(2 * n))
x[np.argmin(x)], x[np.argmax(x)] = 0.0, 1.0
cond = np.repeat(["ctrl", "treat"], n)
c = (cond == "treat").astype(float)
kinds = ["identical"] * 30 + ["level 0.3"] * 30 + ["crossing 0.4"] * 10 + ["peak shift"] * 10
cols = []
for k in kinds:
    a, b = rng.uniform(0.2, 1.0), rng.uniform(1, 5)
    base = a * np.exp(-b * x)
    d = {"identical": 0 * x, "level 0.3": 0.3 + 0 * x, "crossing 0.4": 0.4 * (x - 0.5),
         "peak shift": 0.8 * np.exp(-3 * np.abs(x - 0.15)) - 0.8 * np.exp(-3 * x)}[k]
    cols.append(base + c * d + rng.normal(0, 1, 2 * n))
genes = [f"{k.split()[0]}_{i}" for i, k in enumerate(kinds)]
adata = ad.AnnData(np.column_stack(cols).astype(np.float32), var=pd.DataFrame(index=genes))
adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
SK = {"draws": 500, "tune": 500, "chains": 2}
kind = pd.Series(kinds, index=genes)
rows = {}
for label, ub, tau in [("per cell, tau=0", False, 0.0), ("per cell, tau=0.1", False, 0.1), ("binned (today), tau=0", True, 0.0), ("binned (today), tau=0.1", True, 0.1)]:
    s, _, _ = dvg.tl.fit_model(adata, setup, inplace=False, use_binned=ub, progressbar=False, sample_kwargs=SK, test="parallel", tau=tau, n_workers=4)
    rows[label] = (s["parallel_pval"] < 0.05).groupby(kind).mean()
    print(f"done: {label}", flush=True)
print("\nrejection rate at 0.05 (genes per kind: 30 identical, 30 level, 10 crossing, 10 peak shift)")
print(pd.DataFrame(rows).loc[["identical", "level 0.3", "crossing 0.4", "peak shift"]].round(2).to_string())
