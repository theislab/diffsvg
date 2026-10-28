"""Same within-SHH split null as slide_split_null.py (same genes and halves), with slides modelled: sample='sample', NUTS on bins."""

import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

import diffsvg as dvg
import lab

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rng = np.random.default_rng(0)  # same stream as slide_split_null.py: genes first, then the halves
all_genes = list(ad.read_h5ad(lab.DATA, backed="r").var_names)
GENES = ["GLI1", "VAMP2", *rng.choice([g for g in all_genes if g not in ("GLI1", "VAMP2")], 8, replace=False)]

full = ad.read_h5ad(lab.DATA)
dm = full.obsm["design_matrix"]
keep = (full.obs["condition"].astype(str) == "SHH").to_numpy() & dm["True"].notna().to_numpy()
adata = full[keep, GENES].copy()
slides = np.array(sorted(adata.obs["sample"].astype(str).unique()))
print(f"{len(slides)} SHH slides, {adata.n_obs} cells, genes {[str(g) for g in GENES]}", flush=True)

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
        stats, _, _ = dvg.tl.fit_model(
            adata, setup, n_workers=3, test="parallel", tau=tau, inplace=False, progressbar=False, chains=4, draws=500, tune=500
        )
        keep_cols = ["shape_diff", "parallel_stat", "parallel_pval", "n_divergent", "rhat_max", "ess_bulk_min"]
        rows.append(stats[keep_cols].assign(split=split, tau=tau).reset_index())
        print(f"split {split} tau {tau}: {time.perf_counter() - t:.0f}s, p<0.05 in {(stats.parallel_pval < 0.05).sum()}/{len(stats)}, "
              f"divergent {int(stats.n_divergent.sum())}, rhat max {stats.rhat_max.max():.3f}", flush=True)
out = pd.concat(rows)
out.to_csv("results/slide_split_null_samples.csv", index=False)
print(out.groupby("tau").agg(rejected=("parallel_pval", lambda p: (p < 0.05).mean()), median_shape_diff=("shape_diff", "median"),
                              divergent=("n_divergent", "sum"), rhat_max=("rhat_max", "max")).round(3).to_string())
