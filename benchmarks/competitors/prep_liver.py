"""Export the liver hepatocyte table + donor splits for the R competitor benchmarks (DESpace2, tradeSeq).

Same anchor as benchmarks/liver/run_liver.py: distance to the nearest cholangiocyte in the same section,
capped at 500 um, scaled to [0, 1]. Hepatocytes only. Splits: seed 0..9, group A = rng(seed).choice(sorted donors, 4).
Writes to results/: cells.csv (all hepatocytes: donor, condition, dist, bin), counts_pb.csv (pseudobulk donor x bin x gene),
sub_cells.csv + sub_counts.mtx (2,000 hepatocytes/donor, raw counts), genes.txt, splits.csv.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

CAP, NBIN, NSUB = 500.0, 5, 2000
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out, exist_ok=True)
a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/liver_fibrosis_merfish.h5ad"))

dist = np.full(a.n_obs, np.nan)
ctype = a.obs["Cell_Type_final"].astype(str).to_numpy()
for sec in a.obs["section_uid"].unique():
    m = (a.obs["section_uid"] == sec).to_numpy()
    xy = a.obsm["spatial"][m]
    d, _ = NearestNeighbors(n_neighbors=1).fit(xy[ctype[m] == "Cholangiocyte"]).kneighbors(xy)
    dist[m] = d[:, 0]
hep = np.isin(ctype, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
a = a[hep].copy()
x = np.clip(dist[hep] / CAP, 0, 1)
counts = sp.csr_matrix(a.layers["csc_counts"])
genes = a.var["symbol"].astype(str).to_numpy()
cells = pd.DataFrame(
    {
        "cell": a.obs_names,
        "donor": a.obs["donor_id"].astype(str).to_numpy(),
        "condition": a.obs["condition"].astype(str).to_numpy(),
        "dist": x,
        "bin": np.minimum((x * NBIN).astype(int), NBIN - 1) + 1,  # 5 equal-width bins on [0, 1]
    }
)
pd.Series(genes).to_csv(os.path.join(out, "genes.txt"), index=False, header=False)

# Pseudobulk donor x bin (all hepatocytes)
key = cells["donor"] + "|" + cells["bin"].astype(str)
codes, uniq = pd.factorize(key)
ind = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
pb = pd.DataFrame((ind @ counts).toarray(), index=uniq, columns=genes)
pb.index.name = "donor_bin"
pb.to_csv(os.path.join(out, "counts_pb.csv"))

# Subsample for tradeSeq
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(cells["donor"] == d), NSUB, replace=False) for d in sorted(cells["donor"].unique())])
cells.iloc[idx].to_csv(os.path.join(out, "sub_cells.csv"), index=False)
scipy.io.mmwrite(os.path.join(out, "sub_counts.mtx"), counts[idx].T.tocsc().astype(np.int64))  # genes x cells
cells.to_csv(os.path.join(out, "cells.csv"), index=False)

donors = np.array(sorted(cells["donor"].unique()))
n_dis = cells.loc[cells.condition == "diseased", "donor"].nunique()
rows = [{"split": "real", "donor": d, "group": cells.loc[cells.donor == d, "condition"].iloc[0]} for d in donors]
for s in range(10):
    pick = set(np.random.default_rng(s).choice(donors, size=n_dis, replace=False))
    rows += [{"split": f"null{s}", "donor": d, "group": "A" if d in pick else "B"} for d in donors]
pd.DataFrame(rows).to_csv(os.path.join(out, "splits.csv"), index=False)
print(f"{len(cells)} hepatocytes, {len(donors)} donors, bins per donor:\n{pd.crosstab(cells.donor, cells.bin)}")
print(pd.DataFrame(rows).pivot(index="split", columns="donor", values="group"))
