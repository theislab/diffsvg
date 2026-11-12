"""Per-donor zonation slopes in the liver data: each donor's hepatocyte expression regressed on portal distance."""
import os
import anndata as ad, numpy as np, pandas as pd, scanpy as sc
from sklearn.neighbors import NearestNeighbors

a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/liver_fibrosis_merfish.h5ad"))
a.X = a.layers["csc_counts"].astype(np.float32)
sc.pp.normalize_total(a); sc.pp.log1p(a)
ct = a.obs["Cell_Type_final"].astype(str).to_numpy()
dist = np.full(a.n_obs, np.nan)
for s in a.obs["section_uid"].unique():
    m = (a.obs["section_uid"] == s).to_numpy(); xy = a.obsm["spatial"][m]
    dist[m] = NearestNeighbors(n_neighbors=1).fit(xy[ct[m] == "Cholangiocyte"]).kneighbors(xy)[0][:, 0]
hep = np.isin(ct, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
genes = ["CYP2E1", "CYP1A2", "CYP3A4", "SDS", "ASS1"]
rows = []
for d in sorted(a.obs["donor_id"].astype(str).unique()):
    m = hep & (a.obs["donor_id"].astype(str) == d).to_numpy()
    x = np.clip(dist[m] / 500, 0, 1)
    X = a[m, genes].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    slopes = np.polyfit(x, X, 1)[0]
    rows.append({"donor": d, "condition": a.obs.loc[m, "condition"].iat[0], **dict(zip(genes, slopes))})
df = pd.DataFrame(rows).sort_values(["condition", "donor"])
os.makedirs("results", exist_ok=True); df.to_csv("results/donor_slopes.csv", index=False)
print(df.round(3).to_string(index=False))
