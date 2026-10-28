import numpy as np, scanpy as sc, squidpy as sq, pandas as pd, warnings
warnings.filterwarnings("ignore")
adata = sc.read_h5ad("/home/lehnerl/Arbeit/data/legnini23.h5ad")
print(adata)
X = adata.X
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
print("X dtype", X.dtype, "min/max", X.min(), X.max(), "frac zero", (X == 0).mean().round(3), "integer-valued", np.allclose(X, np.round(X)))
print("per-gene mean range", X.mean(0).min().round(3), X.mean(0).max().round(3), " sd range", X.std(0).min().round(3), X.std(0).max().round(3))
print(adata.obs.groupby("condition", observed=True)["sample"].nunique())
print(adata.obs.groupby(["condition", "sample"], observed=True).size().to_string())
# anchor as in the notebook
arr = X[:, adata.var_names.get_loc("SHH")]
adata.obs["SHH+"] = (arr > np.percentile(arr, 99)).astype(str)
sq.tl.var_by_distance(adata, cluster_key="SHH+", groups="True", library_key="sample", covariates="condition")
dm = adata.obsm["design_matrix"]
print("anchor quantiles per sample (non-anchor cells):")
q = dm.dropna(subset=["True"]).groupby("sample", observed=True)["True"].describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]]
q["condition"] = adata.obs.groupby("sample", observed=True)["condition"].first()
print(q.round(3).to_string())
adata.write_h5ad("legnini_dm.h5ad")
