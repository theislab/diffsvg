"""Baseline: per-donor linear zonation slope for every gene, then a two-sample t-test (5 df). Compare to diffsvg's slope test."""
import os
import anndata as ad, numpy as np, pandas as pd, scanpy as sc
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors
from diffsvg.tl.model import _benjamini_hochberg as bh

a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/liver_fibrosis_merfish.h5ad"))
a.X = a.layers["csc_counts"].astype(np.float32); sc.pp.normalize_total(a); sc.pp.log1p(a)
a.var_names = a.var["symbol"].astype(str).values
ct = a.obs["Cell_Type_final"].astype(str).to_numpy(); dist = np.full(a.n_obs, np.nan)
for s in a.obs["section_uid"].unique():
    m = (a.obs["section_uid"] == s).to_numpy(); xy = a.obsm["spatial"][m]
    dist[m] = NearestNeighbors(n_neighbors=1).fit(xy[ct[m] == "Cholangiocyte"]).kneighbors(xy)[0][:, 0]
hep = np.isin(ct, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
S, cond = [], []
for d in sorted(a.obs["donor_id"].astype(str).unique()):
    m = hep & (a.obs["donor_id"].astype(str) == d).to_numpy()
    X = a.X[m]; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    S.append(np.polyfit(np.clip(dist[m] / 500, 0, 1), X, 1)[0]); cond.append(a.obs.loc[m, "condition"].iat[0])
S, cond = np.array(S), np.array(cond)
t, p = ttest_ind(S[cond == "healthy"], S[cond == "diseased"])
base = pd.DataFrame({"t_pval": p, "t_fdr": bh(p)}, index=a.var_names)
dv = pd.read_csv("results/disease_donor_slope_tau0.csv", index_col=0)
j = base.join(dv[["slope_pval", "slope_pval_adj"]])
print("t-test on donor slopes: FDR<.05", int((j.t_fdr < .05).sum()), "| FDR<.1", int((j.t_fdr < .1).sum()))
print("diffsvg slope test:     FDR<.05", int((j.slope_pval_adj < .05).sum()), "| FDR<.1", int((j.slope_pval_adj < .1).sum()))
print("spearman of p-values:", round(j.t_pval.rank().corr(j.slope_pval.rank()), 3))
print(j.sort_values("t_pval").head(12).map(lambda v: f"{v:.1e}").to_string())
print("\nminimum attainable FDR with 313 genes if the top gene had p=1e-4:", round(1e-4 * 313, 3))
