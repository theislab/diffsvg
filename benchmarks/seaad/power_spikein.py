"""Power of diffsvg on SEA-AD with real donor variation: spike known effects into a random donor split.

Each fit: a random split of the 27 donors (14 vs 13), five effect sizes spiked into 20 genes each in group A's
cells (lognorm layer), 40 genes left as null controls. Shape effect: E * (depth - 0.5), shape_diff = E.
Level effect: a constant L.
    python power_spikein.py --kind shape|level --tau T --seed S
"""

import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

import diffsvg as dvg

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["shape", "level"], required=True)
p.add_argument("--tau", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_power"))
p.add_argument("--data", default=os.path.expanduser("~/Documents/data/diffsvg/seaad_mtg_merfish.h5ad"))
args = p.parse_args()
SIZES = {"shape": [0.1, 0.2, 0.3, 0.5, 0.8], "level": [0.05, 0.1, 0.2, 0.3, 0.5]}[args.kind]

rng = np.random.default_rng(args.seed)
a = ad.read_h5ad(args.data)
a = a[a.obs["Normalized_depth_from_pia"].notna()].copy()
a.var_names = a.var["symbol"].astype(str).values
a.var_names_make_unique()
donor = a.obs["Donor_ID"].astype(str).to_numpy()
donors = np.array(sorted(set(donor)))
group_a = set(rng.choice(donors, size=14, replace=False))
in_a = np.isin(donor, list(group_a))
depth = a.obs["Normalized_depth_from_pia"].clip(0, 1).to_numpy()

genes = np.array(a.var_names)
perm = rng.permutation(len(genes))
effect = pd.Series(0.0, index=genes)
for k, size in enumerate(SIZES):
    effect.iloc[perm[20 * k : 20 * (k + 1)]] = size
X = a.layers["lognorm"]
X = np.asarray(X.toarray() if issparse(X) else X, dtype=float)
profile = (depth - 0.5) if args.kind == "shape" else np.ones_like(depth)
X[in_a] += profile[in_a, None] * effect.to_numpy()[None, :]
a.layers["spiked"] = X

a.obsm["design_matrix"] = pd.DataFrame(
    {"depth": depth, "condition": np.where(in_a, "A", "B"), "donor": donor}, index=a.obs_names
)
t0 = time.perf_counter()
dvg.pp.bin_obs(a, anchor="depth", condition="condition", layer="spiked", sample="donor")
test = "parallel" if args.kind == "shape" else "level"
stats, _, _ = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, test=test, tau=args.tau)
dt = time.perf_counter() - t0
col = f"{test}_pval_adj"
stats["effect"] = effect.reindex(stats.index)
summary = stats.groupby("effect")[col].agg(lambda s: float((s < 0.05).mean())).rename("detected")
os.makedirs(args.out, exist_ok=True)
stats.to_csv(os.path.join(args.out, f"{args.kind}_tau{args.tau:g}_seed{args.seed}.csv"))
print(f"{args.kind} tau={args.tau:g} seed={args.seed} ({dt:.0f} s): " + ", ".join(f"{e:g}: {d:.2f}" for e, d in summary.items()), flush=True)
