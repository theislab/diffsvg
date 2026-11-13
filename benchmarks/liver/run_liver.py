"""diffsvg on human liver fibrosis MERFISH (SpatialDataDB study zenodo:10634153, 7 donors: 4 fibrotic, 3 healthy).

Hepatocytes only; distance = distance to the nearest cholangiocyte (portal tract), in um, capped at --cap um
and scaled to [0, 1] (the same physical scale in every donor). Replicate unit: donor.
    python run_liver.py --contrast disease|null --seed S [--pooled] --test parallel|level [--tau T]
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

p = argparse.ArgumentParser()
p.add_argument("--contrast", default="disease", choices=["disease", "null"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pooled", action="store_true")
p.add_argument("--test", default="parallel", choices=["parallel", "level", "slope", "wald"])
p.add_argument("--tau", type=float, default=0.0)
p.add_argument("--cap", type=float, default=500.0)
p.add_argument("--joint", action="store_true", help="joint-model test instead of the two-stage test")
p.add_argument("--data", default=os.path.expanduser("~/Documents/data/diffsvg/liver_fibrosis_merfish.h5ad"))
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = p.parse_args()

a = ad.read_h5ad(args.data)
a.layers["lognorm"] = a.layers["csc_counts"].astype(np.float32) if "csc_counts" in a.layers else a.X.copy()
sc.pp.normalize_total(a, layer="lognorm")
sc.pp.log1p(a, layer="lognorm")

dist = np.full(a.n_obs, np.nan)
ctype = a.obs["Cell_Type_final"].astype(str).to_numpy()
for sec in a.obs["section_uid"].unique():
    m = (a.obs["section_uid"] == sec).to_numpy()
    xy = a.obsm["spatial"][m]
    chol = ctype[m] == "Cholangiocyte"
    d, _ = NearestNeighbors(n_neighbors=1).fit(xy[chol]).kneighbors(xy)
    dist[m] = d[:, 0]
hep = np.isin(ctype, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
a = a[hep].copy()
dist = dist[hep]

donor = a.obs["donor_id"].astype(str).to_numpy()
disease = a.obs["condition"].astype(str).to_numpy()
if args.contrast == "disease":
    cond = disease
else:
    donors = np.array(sorted(set(donor)))
    n_dis = len(set(donor[disease == "diseased"]))
    pick = set(np.random.default_rng(args.seed).choice(donors, size=n_dis, replace=False))
    cond = np.where(np.isin(donor, list(pick)), "A", "B")

a.obsm["design_matrix"] = pd.DataFrame(
    {"portal_dist": np.clip(dist / args.cap, 0, 1), "condition": cond, "donor": donor}, index=a.obs_names
)
t0 = time.perf_counter()
dvg.pp.bin_obs(a, anchor="portal_dist", condition="condition", layer="lognorm", sample=None if args.pooled else "donor")
stats, preds, curves = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, test=args.test, tau=args.tau, two_stage=not args.joint)
dt = time.perf_counter() - t0
tag = ("joint_" if args.joint else "") + f"{args.contrast}{'' if args.contrast == 'disease' else args.seed}_{'pooled' if args.pooled else 'donor'}_{args.test}_tau{args.tau:g}"
os.makedirs(args.out, exist_ok=True)
stats.to_csv(os.path.join(args.out, f"{tag}.csv"))
for k in ("cond0_mean", "cond0_ci_low", "cond0_ci_high", "cond1_mean", "cond1_ci_low", "cond1_ci_high"):
    curves[k].to_csv(os.path.join(args.out, f"{tag}_{k}.csv"))
donors_per = pd.DataFrame({"c": cond, "d": donor}).drop_duplicates().groupby("c").size().to_dict()
col = f"{args.test}_pval_adj"
top = stats.sort_values(f"{args.test}_pval").head(8)
print(f"{tag}: {a.n_obs} hepatocytes, donors {donors_per}, cap {args.cap:g} um, {dt:.1f} s; BH<.05 {int((stats[col] < .05).sum())}/{len(stats)}; "
      f"edge {stats.exact_edge_mass.max():.1e}; top: " + ", ".join(f"{g}({q:.1e})" for g, q in top[col].items()), flush=True)
