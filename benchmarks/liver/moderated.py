"""Does limma-style variance moderation help with 7 donors? Per-donor slopes -> plain vs moderated t, real contrast + all 35 donor splits."""
import itertools, os
import anndata as ad, numpy as np, pandas as pd, scanpy as sc
from scipy.special import digamma, polygamma
from scipy.stats import t as tdist
from sklearn.neighbors import NearestNeighbors
from diffsvg.tl.model import _benjamini_hochberg as bh

def trigamma_inv(y):
    x = 0.5 + 1 / y
    for _ in range(50):
        x += polygamma(1, x) * (1 - polygamma(1, x) / y) / polygamma(2, x)
    return x

def squeeze(s2, d):  # Smyth 2004 fitFDist by moments
    z = np.log(s2); e = z - digamma(d / 2) + np.log(d / 2); evar = e.var(ddof=1) - polygamma(1, d / 2)
    if evar <= 0: return np.full_like(s2, np.exp(e.mean())), np.inf
    d0 = 2 * trigamma_inv(evar); s02 = np.exp(e.mean() + digamma(d0 / 2) - np.log(d0 / 2))
    return (d0 * s02 + d * s2) / (d0 + d), d0

def tests(S, g):
    a, b = S[g], S[~g]; n1, n2 = len(a), len(b); d = n1 + n2 - 2
    diff = a.mean(0) - b.mean(0); s2 = (((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)) / d
    se = np.sqrt(1 / n1 + 1 / n2)
    p_plain = 2 * tdist.sf(np.abs(diff / np.sqrt(s2) / se), d)
    s2m, d0 = squeeze(s2, d); p_mod = 2 * tdist.sf(np.abs(diff / np.sqrt(s2m) / se), d + min(d0, 1e6))
    return diff, p_plain, p_mod, d0

a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/liver_fibrosis_merfish.h5ad"))
a.X = a.layers["csc_counts"].astype(np.float32); sc.pp.normalize_total(a); sc.pp.log1p(a)
a.var_names = a.var["symbol"].astype(str).values
ct = a.obs["Cell_Type_final"].astype(str).to_numpy(); dist = np.full(a.n_obs, np.nan)
for s in a.obs["section_uid"].unique():
    m = (a.obs["section_uid"] == s).to_numpy(); xy = a.obsm["spatial"][m]
    dist[m] = NearestNeighbors(n_neighbors=1).fit(xy[ct[m] == "Cholangiocyte"]).kneighbors(xy)[0][:, 0]
hep = np.isin(ct, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
S, cond = [], []
for dn in sorted(a.obs["donor_id"].astype(str).unique()):
    m = hep & (a.obs["donor_id"].astype(str) == dn).to_numpy()
    X = a.X[m]; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    S.append(np.polyfit(np.clip(dist[m] / 500, 0, 1), X, 1)[0]); cond.append(a.obs.loc[m, "condition"].iat[0])
S, cond = np.array(S), np.array(cond)
real = cond == "healthy"
diff, pp, pm, d0 = tests(S, real)
print(f"REAL healthy vs fibrotic: prior df d0 = {d0:.1f} (test df 5 -> {5 + d0:.1f})")
print(f"  FDR<.05  plain t: {int((bh(pp) < .05).sum())}   moderated t: {int((bh(pm) < .05).sum())}")
res = pd.DataFrame({"slope_diff": diff, "p_plain": pp, "fdr_plain": bh(pp), "p_mod": pm, "fdr_mod": bh(pm)}, index=a.var_names)
res.to_csv("results/moderated_slopes.csv")
print(res.sort_values("p_mod").head(15).round(5).to_string())
rows = []
for idx in itertools.combinations(range(7), 3):
    g = np.zeros(7, bool); g[list(idx)] = True
    if (g == real).all() or (g == ~real).all(): continue
    _, p1, p2, dd = tests(S, g)
    rows.append(dict(nominal_plain=(p1 < .05).mean(), nominal_mod=(p2 < .05).mean(), fdr_hits_plain=int((bh(p1) < .05).sum()), fdr_hits_mod=int((bh(p2) < .05).sum()), d0=dd))
n = pd.DataFrame(rows)
print(f"\n{len(n)} random 3-vs-4 donor splits (real split excluded):")
print(n.describe().loc[["mean", "max"]].round(3).to_string())
