"""Build the legnini design matrix exactly as the morphogen notebook does and cache it."""
import warnings
import numpy as np
import scanpy as sc
import squidpy as sq

warnings.filterwarnings("ignore")
adata = sc.read_h5ad("/home/lehnerl/Arbeit/data/legnini23.h5ad")
print(adata)
print("layers:", list(adata.layers))
X = adata.X
print("X type", type(X), "dtype", X.dtype)
for name, M in [("X", adata.X), *[(k, adata.layers[k]) for k in adata.layers]]:
    A = M[:2000].toarray() if hasattr(M, "toarray") else np.asarray(M[:2000])
    print(f"{name}: min {A.min():.3f} max {A.max():.3f} frac0 {(A == 0).mean():.3f} integer {np.allclose(A, np.round(A))}")
arr = adata.X[:, adata.var_names.get_loc("SHH")]
arr = arr.toarray().ravel() if hasattr(arr, "toarray") else np.asarray(arr).ravel()
adata.obs["SHH+"] = (arr > np.percentile(arr, 99)).astype(str)
sq.tl.var_by_distance(adata, cluster_key="SHH+", groups="True", library_key="sample", covariates="condition")
dm = adata.obsm["design_matrix"]
print(dm.head()); print(dm.describe(include="all").T)
adata.write_h5ad("legnini_dm.h5ad")
