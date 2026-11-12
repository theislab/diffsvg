"""SEA-AD per cell type: depth curves within one seaad_subclass, donor effects, real contrasts and random splits.

    python per_celltype.py --index I --out DIR      # I = row of the subclass list (SLURM array task id)
Writes DIR/<subclass>__<contrast>__<test>.csv and prints one summary line per fit.
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
p.add_argument("--index", type=int, required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--min-cells-per-donor", type=int, default=20)
p.add_argument("--rescale", action="store_true", help="map the cell type's 2-98% depth range to [0, 1]")
p.add_argument("--joint", action="store_true", help="joint-model test instead of the two-stage test")
args = p.parse_args()

full = ad.read_h5ad(args.data)
full = full[full.obs["Normalized_depth_from_pia"].notna()]
subclasses = sorted(full.obs["seaad_subclass"].astype(str).unique())
sub = subclasses[args.index]
a = full[full.obs["seaad_subclass"].astype(str) == sub].copy()
del full
a.var_names = a.var["symbol"].astype(str).values
a.var_names_make_unique()

donor = a.obs["Donor_ID"].astype(str)
keep = donor.map(donor.value_counts()) >= args.min_cells_per_donor  # donors with too few cells of this type add no curve
a = a[keep.to_numpy()].copy()
donor = a.obs["Donor_ID"].astype(str).to_numpy()
adnc = a.obs["Overall_AD_neuropathological_Change"].astype(str).to_numpy()
dementia = a.obs["Cognitive_Status"].astype(str).to_numpy()
depth = a.obs["Normalized_depth_from_pia"].clip(0, 1).to_numpy()
if args.rescale:  # the test averages over [0, 1]; without this a layer-restricted type leaves most of it empty
    lo, hi = np.quantile(depth, [0.02, 0.98])
    depth = np.clip((depth - lo) / (hi - lo), 0, 1)

contrasts = {
    "adnc": np.where(np.isin(adnc, ["High", "Intermediate"]), "AD", "lowAD"),
    "dementia": np.where(dementia == "Dementia", "dementia", "nodementia"),
}
donors = np.array(sorted(set(donor)))
ad_donors = sorted(set(donor[np.isin(adnc, ["High", "Intermediate"])]))
for s in range(3):
    pick = set(np.random.default_rng(s).choice(donors, size=len(ad_donors), replace=False))
    contrasts[f"null{s}"] = np.where(np.isin(donor, list(pick)), "A", "B")

os.makedirs(args.out, exist_ok=True)
safe = sub.replace("/", "-").replace(" ", "_")
for name, cond in contrasts.items():
    per = pd.DataFrame({"c": cond, "d": donor}).drop_duplicates().groupby("c").size()
    if len(per) < 2 or per.min() < 3:
        print(f"rc=skip {sub} {name}: donors per group {per.to_dict()}", flush=True)
        continue
    a.obsm["design_matrix"] = pd.DataFrame({"depth": depth, "condition": cond, "donor": donor}, index=a.obs_names)
    dvg.pp.bin_obs(a, anchor="depth", condition="condition", layer="log1p", sample="donor")
    setup = dvg.tl.setup_model(a)
    for test in ("level", "slope", "parallel"):
        t0 = time.perf_counter()
        stats, _, _ = dvg.tl.fit_model(a, setup, inplace=False, progressbar=False, test=test, tau=0.0, two_stage=not args.joint)
        stats.to_csv(os.path.join(args.out, f"{safe}__{name}__{test}.csv"))
        print(f"rc=0 {sub} {name} {test}: {a.n_obs} cells, donors {per.to_dict()}, {time.perf_counter() - t0:.0f} s, "
              f"BH<.05 {int((stats[test + '_pval_adj'] < .05).sum())}, p<.05 {int((stats[test + '_pval'] < .05).sum())}, "
              f"edge {stats.exact_edge_mass.max():.2e}", flush=True)
