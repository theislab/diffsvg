"""Rerun the Legnini, Xenium and synthetic experiments with the v0.2.0 statistics.

    python run_v020.py --exp NAME --data DIR --out DIR     # NAME from --list
Legnini replicates are organoids (17 sections from 8 organoids, 4 per condition); null splits shuffle whole organoids.
"""

import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd

TESTS = ("parallel", "level", "slope", "wald")


def experiments():
    exps = {}
    for t in TESTS:
        exps[f"legnini_pooled_{t}"] = ("legnini", None, None, t, False)
        exps[f"xenium_pooled_{t}"] = ("xenium", None, None, t, False)
        exps[f"synth_pooled_{t}"] = ("synth", None, None, t, False)
    for t in ("parallel", "level", "slope"):
        exps[f"legnini_organoid_{t}"] = ("legnini", "organoid", None, t, False)
    for t in ("level", "slope"):
        exps[f"legnini_organoid_{t}_twostage"] = ("legnini", "organoid", None, t, True)
    for s in range(10):
        for t in ("parallel", "level", "slope"):
            exps[f"legnini_organoid_null{s}_{t}"] = ("legnini", "organoid", s, t, False)
        for t in ("level", "slope"):
            exps[f"legnini_organoid_null{s}_{t}_twostage"] = ("legnini", "organoid", s, t, True)
        for t in ("parallel", "slope"):
            exps[f"legnini_pooled_null{s}_{t}"] = ("legnini", None, s, t, False)
    return exps


p = argparse.ArgumentParser()
p.add_argument("--exp")
p.add_argument("--data")
p.add_argument("--out")
p.add_argument("--list", action="store_true")
args = p.parse_args()
EXPS = experiments()
if args.list:
    print("\n".join(EXPS))
    raise SystemExit

import scanpy as sc
import squidpy as sq

import diffsvg as dvg

data, sample, null_seed, test, two_stage = EXPS[args.exp]
LAYER = "log1p_norm"
if data == "legnini":
    a = sc.read_h5ad(f"{args.data}/legnini23.h5ad")
    shh = a[:, "SHH"].X
    shh = (shh.toarray() if hasattr(shh, "toarray") else np.asarray(shh)).ravel()
    a.obs["SHH+"] = (shh > np.percentile(shh, 99)).astype(str)  # Laurens' anchor, kept for comparability
    sq.tl.var_by_distance(a, cluster_key="SHH+", groups="True", library_key="sample", covariates=["condition", "organoid"])
    dm = a.obsm["design_matrix"]
    dm["condition"] = a.obs["condition"].astype(str).values
    dm["organoid"] = a.obs["organoid"].astype(str).values
    if null_seed is not None:
        orgs = np.array(sorted(set(dm["organoid"])))
        pick = set(np.random.default_rng(null_seed).choice(orgs, 4, replace=False))
        dm["condition"] = np.where(dm["organoid"].isin(pick), "A", "B")
    anchor = "True"
elif data == "xenium":
    a = sc.read_h5ad(f"{args.data}/both_conditions_5_7_without_noise.h5ad")
    sc.pp.filter_cells(a, min_counts=10)
    a.layers[LAYER] = a.X.copy()
    sc.pp.normalize_total(a, layer=LAYER)
    sc.pp.log1p(a, layer=LAYER)
    sq.tl.var_by_distance(a, cluster_key="Dentate_Gyrus", groups="anchor", library_key="condition", covariates="condition")
    anchor = "anchor"
else:  # one section per condition, identical true curves per gene
    import anndata as ad

    rng = np.random.default_rng(0)
    n_cells, n_genes = 20000, 300
    x = rng.uniform(0, 1, n_cells) ** 1.5
    cond = rng.choice(["c0", "c1"], n_cells)
    truth = rng.uniform(0.2, 2.0, n_genes) + rng.uniform(0, 1.0, n_genes) * np.sin(np.pi * rng.uniform(0.5, 2.5, n_genes) * x[:, None])
    X = truth + rng.uniform(0.2, 1.0, n_genes) * rng.standard_normal((n_cells, n_genes))
    a = ad.AnnData(X.astype(np.float32))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.layers[LAYER] = a.X.copy()
    a.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=a.obs_names)
    anchor = "anchor"

t0 = time.perf_counter()
dvg.pp.bin_obs(a, anchor=anchor, condition="condition", layer=LAYER, sample=sample)
stats, _, _ = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, test=test, two_stage=two_stage)
dt = time.perf_counter() - t0
os.makedirs(args.out, exist_ok=True)
stats.to_csv(f"{args.out}/{args.exp}.csv")
q = stats[f"{test}_pval_adj"]
print(f"{args.exp}: {dt:.0f} s, BH<.05 {int((q < .05).sum())}/{len(stats)}, p<.05 {(stats[f'{test}_pval'] < .05).mean():.3f}", flush=True)
