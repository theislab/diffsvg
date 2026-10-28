"""Empirical-Bayes slides with ADVI on cells, on legnini: the within-SHH split null (same genes/halves as before) and Ctrl vs SHH."""

import sys
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd

import diffsvg as dvg
import lab

warnings.filterwarnings("ignore")
N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
KW = {"n_workers": 3, "test": "parallel", "inference": "advi", "use_binned": False, "inplace": False, "progressbar": False}
COLS = ["shape_diff", "parallel_stat", "parallel_pval", "sample_level_sd", "sample_curve_sd", "vi_khat", "vi_ess"]

rng = np.random.default_rng(0)  # same stream as slide_split_null.py: genes first, then the halves
all_genes = list(ad.read_h5ad(lab.DATA, backed="r").var_names)
GENES = ["GLI1", "VAMP2", *[str(g) for g in rng.choice([g for g in all_genes if g not in ("GLI1", "VAMP2")], 8, replace=False)]]
full = ad.read_h5ad(lab.DATA)

# (1) split null
keep = (full.obs["condition"].astype(str) == "SHH").to_numpy() & full.obsm["design_matrix"]["True"].notna().to_numpy()
adata = full[keep, GENES].copy()
slides = np.array(sorted(adata.obs["sample"].astype(str).unique()))
rows = []
for split in range(N_SPLITS):
    half_a = set(rng.choice(slides, len(slides) // 2, replace=False))
    d = adata.obsm["design_matrix"].copy()
    d["condition"] = adata.obs["sample"].astype(str).map(lambda s: "A" if s in half_a else "B").to_numpy()
    adata.obsm["design_matrix"] = d
    dvg.pp.bin_obs(adata, anchor="True", condition="condition", genes=GENES, layer="log1p_norm", n_bins=50, window_radius=None, sample="sample")
    setup = dvg.tl.setup_model(adata)
    for tau in (0.0, 0.1):
        t = time.perf_counter()
        stats, _, _ = dvg.tl.fit_model(adata, setup, tau=tau, **KW)
        rows.append(stats[COLS].assign(split=split, tau=tau).reset_index())
        print(f"split {split} tau {tau}: {time.perf_counter() - t:.0f}s, p<0.05 in {(stats.parallel_pval < 0.05).sum()}/{len(stats)}, "
              f"khat max {stats.vi_khat.max():.2f}, ess min {stats.vi_ess.min():.3f}", flush=True)
null = pd.concat(rows)
null.to_csv("results/slide_eb_split_null.csv", index=False)
print(null.groupby("tau").agg(rejected=("parallel_pval", lambda p: (p < 0.05).mean()), p_mean=("parallel_pval", "mean"),
                               khat_max=("vi_khat", "max"), ess_min=("vi_ess", "min")).round(3).to_string(), flush=True)

# (2) Ctrl vs SHH, strong genes
strong = ["GLI1", "PAX7", "HHIP", "VAMP2", "FOXA2", "DBX1"]
adata = full[:, strong].copy()
dvg.pp.bin_obs(adata, anchor="True", condition="condition", genes=strong, layer="log1p_norm", n_bins=50, window_radius=None, sample="sample")
t = time.perf_counter()
stats, _, _ = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), tau=0.1, **KW)
print(f"Ctrl vs SHH, ADVI empirical Bayes on cells: {time.perf_counter() - t:.0f}s")
pd.set_option("display.width", 250)
print(stats[COLS].round(4).to_string())
stats[COLS].to_csv("results/slide_eb_real_contrast.csv")
