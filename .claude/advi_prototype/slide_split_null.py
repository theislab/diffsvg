"""Within-condition null through fit_model: split the SHH slides at random into two halves and test 'condition' = half.

The halves share a condition, so any rejection is false. Uses the package path (per cell, inference='advi').
Small on purpose: GENES x N_SPLITS fits per tau, 3 workers.
"""

import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

import lab
import diffsvg as dvg

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rng = np.random.default_rng(0)
all_genes = list(ad.read_h5ad(lab.DATA, backed="r").var_names)
GENES = ["GLI1", "VAMP2", *rng.choice([g for g in all_genes if g not in ("GLI1", "VAMP2")], 8, replace=False)]

full = ad.read_h5ad(lab.DATA)
dm = full.obsm["design_matrix"]
keep = (full.obs["condition"].astype(str) == "SHH").to_numpy() & dm["True"].notna().to_numpy()
adata = full[keep, GENES].copy()
slides = np.array(sorted(adata.obs["sample"].astype(str).unique()))
print(f"{len(slides)} SHH slides with an anchor, {adata.n_obs} cells, genes: {GENES}", flush=True)

rows = []
for split in range(N_SPLITS):
    half_a = set(rng.choice(slides, len(slides) // 2, replace=False))
    label = adata.obs["sample"].astype(str).map(lambda s: "A" if s in half_a else "B").to_numpy()
    d = adata.obsm["design_matrix"].copy()
    d["condition"] = label
    adata.obsm["design_matrix"] = d
    dvg.pp.bin_obs(adata, anchor="True", condition="condition", genes=GENES, layer="log1p_norm", n_bins=50, window_radius=None)
    setup = dvg.tl.setup_model(adata)
    for tau in (0.0, 0.1):
        t = time.perf_counter()
        stats, _, _ = dvg.tl.fit_model(
            adata, setup, n_workers=3, use_binned=False, inference="advi", test="parallel", tau=tau, inplace=False, progressbar=False
        )
        stats = stats[["shape_diff", "parallel_stat", "parallel_pval", "vi_khat", "vi_ess"]].assign(split=split, tau=tau)
        rows.append(stats.reset_index())
        print(f"split {split} tau {tau}: {time.perf_counter() - t:.0f}s, p<0.05 in {(stats.parallel_pval < 0.05).sum()}/{len(stats)}", flush=True)
out = pd.concat(rows)
out.to_csv("results/slide_split_null.csv", index=False)
print(out.groupby("tau").agg(rejected=("parallel_pval", lambda p: (p < 0.05).mean()), median_shape_diff=("shape_diff", "median"),
                              khat_max=("vi_khat", "max"), ess_min=("vi_ess", "min")).round(3).to_string())
