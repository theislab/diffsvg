"""diffsvg on cuprizone-rapamycin vs control mouse forebrain Xenium (GEO GSE266689 via SpatialDataDB; 5 vs 5 sections).

Anchor: distance (um) to the nearest ependymal cell (ventricle wall), capped at --cap um and scaled to [0, 1].
Population: one cell type (--celltype). Replicate unit: section (one section per mouse).
    python run_cuprizone.py [--cap 1000] --celltype oligo|microglia|all --contrast disease|null --seed S [--pooled] --test slope|level
"""

import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

import diffsvg as dvg
from celltypes import CELLTYPE

p = argparse.ArgumentParser()
p.add_argument("--celltype", default="oligo")
p.add_argument("--contrast", default="disease", choices=["disease", "null"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pooled", action="store_true")
p.add_argument("--test", default="slope", choices=["parallel", "level", "slope"])
p.add_argument("--tau", type=float, default=0.0)
p.add_argument("--cap", type=float, default=1000.0)
p.add_argument("--data", default=os.path.expanduser("~/Documents/data/diffsvg/cuprizone_xenium.h5ad"))
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = p.parse_args()

a = ad.read_h5ad(args.data)
ctype = a.obs["cluster"].astype(str).map(CELLTYPE).fillna("other").to_numpy()
dist = np.full(a.n_obs, np.nan)
for sec in a.obs["section"].unique():
    m = (a.obs["section"] == sec).to_numpy()
    xy = a.obsm["spatial"][m]
    ep = ctype[m] == "ependymal"
    d, _ = NearestNeighbors(n_neighbors=1).fit(xy[ep]).kneighbors(xy)
    dist[m] = d[:, 0]
keep = np.ones(a.n_obs, bool) if args.celltype == "all" else ctype == args.celltype
a = a[keep].copy()
dist = dist[keep]
a.layers["lognorm"] = a.X.astype(np.float32)  # X holds raw counts
sc.pp.normalize_total(a, layer="lognorm")
sc.pp.log1p(a, layer="lognorm")

section = a.obs["section"].astype(str).to_numpy()
disease = a.obs["condition"].astype(str).to_numpy()
if args.contrast == "disease":
    cond = disease
else:
    secs = np.array(sorted(set(section)))
    pick = set(np.random.default_rng(args.seed).choice(secs, size=len(set(section[disease == "cuprizone"])), replace=False))
    cond = np.where(np.isin(section, list(pick)), "A", "B")

a.obsm["design_matrix"] = pd.DataFrame(
    {"vent_dist": np.clip(dist / args.cap, 0, 1), "condition": cond, "section": section}, index=a.obs_names
)
t0 = time.perf_counter()
dvg.pp.bin_obs(a, anchor="vent_dist", condition="condition", layer="lognorm", sample=None if args.pooled else "section")
stats, preds, curves = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, test=args.test, tau=args.tau)
dt = time.perf_counter() - t0
tag = f"{args.celltype}_{args.contrast}{'' if args.contrast == 'disease' else args.seed}_{'pooled' if args.pooled else 'section'}_{args.test}"
os.makedirs(args.out, exist_ok=True)
stats.to_csv(os.path.join(args.out, f"{tag}.csv"))
col = f"{args.test}_pval_adj"
top = stats.sort_values(f"{args.test}_pval").head(8)
print(f"{tag}: {a.n_obs} cells, cap {args.cap:g} um, {dt:.1f} s; BH<.05 {int((stats[col] < .05).sum())}/{len(stats)}; "
      f"nominal<.05 {(stats[f'{args.test}_pval'] < .05).mean():.2f}; edge {stats.exact_edge_mass.max():.1e}; top: "
      + ", ".join(f"{g}({q:.1e})" for g, q in top[col].items()), flush=True)
