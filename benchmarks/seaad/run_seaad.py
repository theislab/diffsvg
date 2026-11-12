"""diffsvg on SEA-AD MTG MERFISH (SpatialDataDB study seaad:MTG-MERFISH): cortical depth curves by AD pathology.

Distance axis: Normalized_depth_from_pia (0 = pia, 1 = white matter). Replicate unit: donor (sections pooled per donor).
    python run_seaad.py --contrast adnc|dementia|null --seed N [--pooled] [--celltype CLASS]
The h5ad is built from the SpatialDataDB tables (see README in this folder).
"""

import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import anndata as ad
import numpy as np
import pandas as pd

import diffsvg as dvg

p = argparse.ArgumentParser()
p.add_argument("--contrast", default="adnc", choices=["adnc", "dementia", "null"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pooled", action="store_true", help="no donor effects (cells as replicates)")
p.add_argument("--celltype", default=None, help="restrict to one seaad_class")
p.add_argument("--tau", type=float, default=0.1)
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
p.add_argument("--data", default=os.path.expanduser("~/Documents/data/diffsvg/seaad_mtg_merfish.h5ad"))
args = p.parse_args()

a = ad.read_h5ad(args.data)
a = a[a.obs["Normalized_depth_from_pia"].notna()].copy()
if args.celltype:
    a = a[a.obs["seaad_class"] == args.celltype].copy()
a.var_names = a.var["symbol"].astype(str).values
a.var_names_make_unique()

donor = a.obs["Donor_ID"].astype(str)
adnc = a.obs["Overall_AD_neuropathological_Change"].astype(str)
if args.contrast == "adnc":
    cond = np.where(adnc.isin(["High", "Intermediate"]), "AD", "lowAD")
elif args.contrast == "dementia":
    cond = np.where(a.obs["Cognitive_Status"].astype(str) == "Dementia", "dementia", "nodementia")
else:  # random donor split, same group sizes as the ADNC contrast: no true difference
    donors = np.array(sorted(donor.unique()))
    n_ad = donor[adnc.isin(["High", "Intermediate"])].nunique()
    ad_donors = set(np.random.default_rng(args.seed).choice(donors, size=n_ad, replace=False))
    cond = np.where(donor.isin(ad_donors), "A", "B")

a.obsm["design_matrix"] = pd.DataFrame(
    {"depth": a.obs["Normalized_depth_from_pia"].clip(0, 1).to_numpy(), "condition": cond, "donor": donor.to_numpy()},
    index=a.obs_names,
)
t0 = time.perf_counter()
dvg.pp.bin_obs(a, anchor="depth", condition="condition", layer="lognorm", sample=None if args.pooled else "donor")
stats, _, _ = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, tau=args.tau)
dt = time.perf_counter() - t0
tag = f"{args.contrast}{'' if args.contrast != 'null' else args.seed}_{'pooled' if args.pooled else 'donor'}{'_' + args.celltype.split(':')[-1].strip().replace(' ', '') if args.celltype else ''}{'' if args.tau == 0.1 else f'_tau{args.tau:g}'}"
os.makedirs(args.out, exist_ok=True)
stats.to_csv(os.path.join(args.out, f"{tag}.csv"))
n_groups = pd.Series(cond).groupby(cond).size().to_dict()
donors_per = pd.DataFrame({"c": cond, "d": donor.to_numpy()}).drop_duplicates().groupby("c").size().to_dict()
print(f"{tag}: {a.n_obs} cells, cells per group {n_groups}, donors per group {donors_per}; {dt:.1f} s; "
      f"BH<.05 {int((stats.parallel_pval_adj < .05).sum())}/{len(stats)}; p<.05 {int((stats.parallel_pval < .05).sum())}; "
      f"max edge mass {stats.exact_edge_mass.max():.2e}", flush=True)
