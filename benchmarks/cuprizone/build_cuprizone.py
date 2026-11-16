"""Build the compact cuprizone-rapamycin vs control Xenium h5ad (GEO GSE266689 via SpatialDataDB).

10 sections (5 CupRap, 5 control) from two batches with different panels (forebrain Biol1/2: 348 genes;
NoRecovery 1-3: 347 genes); only shared genes kept. Raw counts in layers['counts']. Cell types from Leiden
clusters (fit on a 200k subsample, transferred by kNN) labelled by marker genes.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier
from sddb import Catalog

OUT = os.path.expanduser("~/Documents/data/diffsvg/cuprizone_xenium.h5ad")
coh = Catalog(version="2026-11-08").collection("geo-gse266689").filter(
    sample_title=[f"Adult_{c}_{b}" for c in ("CupRap", "Control") for b in
                  ("NoRecovery_Xenium_1", "NoRecovery_Xenium_2", "NoRecovery_Xenium_3", "Forebrain_Xenium_Biol1", "Forebrain_Xenium_Biol2")])
m = coh.members.reset_index(drop=True)
assert len(m) == 10, m.sample_title.tolist()
sds = coh.to_list(elements=["tables/table"])
tabs = []
for (_, r), sd in zip(m.iterrows(), sds):
    t = sd.tables["table"]
    X = t.layers["csc_counts"] if "csc_counts" in t.layers else t.X
    X = sp.csr_matrix(X).astype(np.float32)
    assert np.all(X.data % 1 == 0)
    a = ad.AnnData(X, obs=pd.DataFrame({"cell_area": t.obs["cell_area"].to_numpy()}, index=[f"{r.sample_id}_{i}" for i in t.obs_names]), var=pd.DataFrame(index=t.var_names))
    a.obsm["spatial"] = np.asarray(t.obsm["spatial"], dtype=np.float32)
    a.obs["section"] = r.sample_title.replace("Adult_", "").replace("_Xenium", "")
    a.obs["condition"] = "cuprizone" if "CupRap" in r.sample_title else "control"
    a.obs["batch"] = "forebrain" if "Biol" in r.sample_title else "norecovery"
    tabs.append(a)
a = ad.concat(tabs, join="inner")
a.layers["counts"] = a.X.copy()
print(a, a.obs.groupby(["batch", "condition"], observed=True).section.nunique(), flush=True)

sc.pp.filter_cells(a, min_counts=20)
sc.pp.normalize_total(a)
sc.pp.log1p(a)
sc.pp.pca(a, n_comps=30)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(a.n_obs, 200_000, replace=False))
s = a[idx].copy()
from sklearn.cluster import KMeans
s.obs["leiden"] = KMeans(n_clusters=30, n_init=3, random_state=0).fit_predict(s.obsm["X_pca"]).astype(str)
knn = KNeighborsClassifier(n_neighbors=15).fit(s.obsm["X_pca"], s.obs["leiden"].to_numpy())
lab = np.empty(a.n_obs, dtype=object)
for i in range(0, a.n_obs, 200_000):
    lab[i:i + 200_000] = knn.predict(a.obsm["X_pca"][i:i + 200_000])
a.obs["cluster"] = pd.Categorical(lab)
a.X = a.layers["counts"]  # keep raw counts only; lognorm rebuilt downstream
del a.obsm["X_pca"]
a.uns = {}
a.write_h5ad(OUT, compression="gzip")
print("wrote", OUT, flush=True)
