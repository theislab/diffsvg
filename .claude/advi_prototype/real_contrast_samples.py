"""Ctrl vs SHH on legnini, pooled vs slide-aware (NUTS on bins), a few strong genes: does the slide-aware test keep its power?"""

import time

import anndata as ad
import pandas as pd

import diffsvg as dvg
import lab

GENES = ["GLI1", "PAX7", "HHIP", "VAMP2", "FOXA2", "DBX1"]
adata = ad.read_h5ad(lab.DATA)[:, GENES].copy()
kw = {"n_workers": 3, "test": "parallel", "tau": 0.1, "inplace": False, "progressbar": False, "chains": 4, "draws": 500, "tune": 500}
out = {}
for label, sample in [("pooled", None), ("slide-aware", "sample")]:
    t = time.perf_counter()
    dvg.pp.bin_obs(adata, anchor="True", condition="condition", genes=GENES, layer="log1p_norm", n_bins=50, window_radius=None, sample=sample)
    stats, _, _ = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), **kw)
    out[label] = stats[["shape_diff", "slope_diff", "parallel_stat", "parallel_pval", "rhat_max", "ess_bulk_min"]]
    print(f"{label}: {time.perf_counter() - t:.0f}s", flush=True)
res = pd.concat(out, axis=1)
pd.set_option("display.width", 250)
print(res.round(4).to_string())
res.to_csv("results/real_contrast_samples.csv")
