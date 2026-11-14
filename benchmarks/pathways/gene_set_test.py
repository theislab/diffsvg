"""Prototype: pathway-level tests on diffsvg's per-replicate summaries (level and slope per donor/organoid and gene).

Per replicate and gene: a weighted least-squares line of the bin means on the anchor support -> level (at the
support midpoint) and slope. Gene-set tests with the replicate as the unit:
  * self-contained rotation test (ROAST-style, Wu et al. 2010): mean moderated-free t over the set, the joint
    rotation of all genes' residuals keeps the inter-gene correlation; directions "up", "down", "mixed" (mean |t|).
  * competitive rank test (CAMERA-style, Wu & Smyth 2012): Wilcoxon of the set's t against the other genes,
    variance inflated by 1 + (m - 1) * mean inter-gene residual correlation.
    python gene_set_test.py --mode liver|legnini|seaad_spike --data DIR --out DIR [--seed S]
"""

import argparse
import itertools
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import diffsvg as dvg
from diffsvg.tl.model import _anchor_support


def replicate_summaries(binned):
    """(replicates x genes) level and slope, and the condition of each replicate."""
    lo, hi = _anchor_support(binned)
    x_all = np.asarray(binned.bin_centers)[np.asarray(binned.bin_assignment)]
    w_all = np.asarray(binned.n_obs, dtype=float)
    s_idx = np.asarray(binned.sample_index)
    cond = np.asarray(binned.condition_fit)
    Y = np.column_stack([np.asarray(binned.Y[g], dtype=float) for g in binned.genes])
    reps = np.unique(s_idx)
    level, slope, rcond = np.empty((len(reps), Y.shape[1])), np.empty((len(reps), Y.shape[1])), np.empty(len(reps))
    for i, s in enumerate(reps):
        r = (s_idx == s) & (x_all >= lo) & (x_all <= hi) & (w_all > 0)
        X = np.column_stack([np.ones(r.sum()), x_all[r] - (lo + hi) / 2])
        W = w_all[r][:, None]
        beta = np.linalg.solve(X.T @ (W * X), X.T @ (W * Y[r]))
        level[i], slope[i], rcond[i] = beta[0], beta[1], cond[r][0]
    return pd.DataFrame(level, columns=binned.genes), pd.DataFrame(slope, columns=binned.genes), rcond


def _t_stats(U):
    """U: rotated (n-1 x G) matrix, row 0 = contrast, rest = residual directions -> per-gene t."""
    resid = U[1:]
    s = np.sqrt((resid**2).mean(axis=0))
    return U[0] / np.where(s > 0, s, np.inf)


def rotation_test(Ymat, group, sets, n_rot=9999, seed=0):
    """Self-contained ROAST-style test. Ymat: replicates x genes, group: 0/1. Returns {set: {up, down, mixed}}."""
    n = len(group)
    X = np.column_stack([np.ones(n), group])
    Q, _ = np.linalg.qr(X, mode="complete")
    U = Q.T @ Ymat  # row 0 intercept, row 1 contrast (sign up to Q), rows 2.. residuals
    sign = np.sign(Q[:, 1] @ (group - group.mean()))
    U = U[1:] * np.r_[sign, np.ones(n - 2)][:, None]
    t_obs = _t_stats(U)
    rng = np.random.default_rng(seed)
    out = {}
    stats_obs = {k: (t_obs[idx].mean(), np.abs(t_obs[idx]).mean()) for k, idx in sets.items()}
    counts = {k: np.zeros(3) for k in sets}
    for _ in range(n_rot):
        R, upper = np.linalg.qr(rng.standard_normal((n - 1, n - 1)))
        R = R * np.sign(np.diag(upper))  # Haar-uniform rotation (plain QR is not)
        t_r = _t_stats(R @ U)
        for k, idx in sets.items():
            m, a = t_r[idx].mean(), np.abs(t_r[idx]).mean()
            counts[k] += [m >= stats_obs[k][0], m <= stats_obs[k][0], a >= stats_obs[k][1]]
    for k in sets:
        up, down, mixed = (counts[k] + 1) / (n_rot + 1)
        out[k] = {"mean_t": stats_obs[k][0], "up": up, "down": down, "mixed": mixed, "size": len(sets[k])}
    return out, t_obs


def camera_rank(t_obs, Ymat, group, idx):
    """Competitive rank test with a variance inflation factor from the inter-gene residual correlation."""
    G, m = len(t_obs), len(idx)
    X = np.column_stack([np.ones(len(group)), group])
    resid = Ymat - X @ np.linalg.lstsq(X, Ymat, rcond=None)[0]
    C = np.corrcoef(resid[:, idx], rowvar=False)
    rho = (C.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
    vif = 1 + (m - 1) * rho
    r = rankdata(t_obs)
    W = r[idx].sum()
    mu = m * (G + 1) / 2
    var = m * (G - m) * (G + 1) / 12 * vif
    z = (W - mu) / np.sqrt(var)
    return {"z": z, "p_up": float(norm.sf(z)), "p_down": float(norm.cdf(z)), "rho": rho}


def two_sample_t(M, group):
    """Per-gene two-sample t (pooled variance) of the replicate summaries: the input decoupler recommends."""
    a, b = M[group == 1], M[group == 0]
    n1, n0 = len(a), len(b)
    sp = (((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)) / (n1 + n0 - 2)
    return (a.mean(0) - b.mean(0)) / np.sqrt(sp * (1 / n1 + 1 / n0))


def decoupler_tests(M, group, net, tag, summary):
    """decoupler ULM on the per-gene t vector, and ORA on the top 10% of genes by |t|."""
    import decoupler as dc

    t = pd.DataFrame([two_sample_t(M.to_numpy(), group)], index=[tag], columns=M.columns).fillna(0.0)
    rows = []
    es, pv = dc.mt.ulm(data=t, net=net, tmin=3, verbose=False)
    for s in es.columns:
        rows.append({"tag": tag, "summary": summary, "set": s, "method": "ulm", "score": es.iloc[0][s], "pval": pv.iloc[0][s]})
    k = max(5, int(0.1 * t.shape[1]))
    absval = t.abs()
    es, pv = dc.mt.ora(data=absval, net=net.assign(weight=1.0), tmin=3, n_up=k, n_bm=0, n_bg=t.shape[1], verbose=False)  # background = the panel, not 20,000 genes
    for s in es.columns:
        rows.append({"tag": tag, "summary": summary, "set": s, "method": "ora_top10pct", "score": es.iloc[0][s], "pval": pv.iloc[0][s]})
    return rows


NET_EXTRA = None  # decoupler resources (hallmark, progeny), loaded once from data/


def run(level, slope, group, sets, tag, out, seed):
    rows = []
    net = pd.DataFrame([(k, g, 1.0) for k, v in sets.items() for g in v], columns=["source", "target", "weight"])
    if NET_EXTRA is not None:
        net = pd.concat([net, NET_EXTRA], ignore_index=True)
    dec = []
    for name, M in (("level", level), ("slope", slope)):
        try:
            dec += decoupler_tests(M, group, net, tag, name)
        except Exception as e:  # noqa: BLE001 - report and keep the rotation results
            print(f"decoupler failed for {tag} {name}: {e}", flush=True)
    if dec:
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(dec).to_csv(f"{out}/{tag}_decoupler.csv", index=False)
    for name, M in (("level", level), ("slope", slope)):
        genes = list(M.columns)
        idx = {k: np.array([genes.index(g) for g in v if g in genes]) for k, v in sets.items()}
        idx = {k: v for k, v in idx.items() if len(v) >= 3}
        rot, t_obs = rotation_test(M.to_numpy(), group, idx, seed=seed)
        for k, v in idx.items():
            cam = camera_rank(t_obs, M.to_numpy(), group, v)
            rows.append({"tag": tag, "summary": name, "set": k, **rot[k], "camera_z": cam["z"], "camera_p_up": cam["p_up"],
                         "camera_p_down": cam["p_down"], "rho": cam["rho"]})
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/{tag}.csv", index=False)
    return df


p = argparse.ArgumentParser()
p.add_argument("--mode", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
_extra = [f for f in (f"{args.data}/decoupler_hallmark.csv", f"{args.data}/decoupler_progeny.csv") if os.path.exists(f)]
if _extra:
    NET_EXTRA = pd.concat([pd.read_csv(f)[["source", "target", "weight"]] for f in _extra], ignore_index=True)

if args.mode == "liver":
    import anndata as ad
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors

    a = ad.read_h5ad(f"{args.data}/liver_fibrosis_merfish.h5ad")
    a.X = a.layers["csc_counts"].astype(np.float32)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.var_names = a.var["symbol"].astype(str).values
    a.var_names_make_unique()
    ct = a.obs["Cell_Type_final"].astype(str).to_numpy()
    dist = np.full(a.n_obs, np.nan)
    for s in a.obs["section_uid"].unique():
        m = (a.obs["section_uid"] == s).to_numpy()
        xy = a.obsm["spatial"][m]
        dist[m] = NearestNeighbors(n_neighbors=1).fit(xy[ct[m] == "Cholangiocyte"]).kneighbors(xy)[0][:, 0]
    hep = np.isin(ct, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"])
    a = a[hep].copy()
    donor = a.obs["donor_id"].astype(str).to_numpy()
    real = a.obs["condition"].astype(str).to_numpy()
    pc = ["CYP2E1", "CYP1A2", "CYP3A4", "GLUL", "OAT", "CYP7A1"]
    pp = ["SDS", "HAL", "ASS1", "CPS1", "PCK1", "HAMP", "ARG1", "G6PC"]
    sets = {"pericentral": pc, "periportal": pp}
    rng = np.random.default_rng(1)
    genes_all = list(a.var_names)
    for i in range(20):  # random sets of size 10: competitive-test null
        sets[f"random{i:02d}"] = list(rng.choice(genes_all, 10, replace=False))
    donors = sorted(set(donor))
    healthy = sorted(set(donor[real == "healthy"]))
    splits = [("real", set(healthy))] + [(f"null{j:02d}", set(c)) for j, c in enumerate(itertools.combinations(donors, 3)) if set(c) != set(healthy)]
    a.obsm["design_matrix"] = pd.DataFrame({"anchor": np.clip(dist[hep] / 500, 0, 1), "donor": donor}, index=a.obs_names)
    for tag, grp in splits:
        a.obsm["design_matrix"]["condition"] = np.where(np.isin(donor, list(grp)), "A_healthy", "B_fibrotic")  # A first -> 0
        dvg.pp.bin_obs(a, anchor="anchor", condition="condition", sample="donor")
        lv, sl, rc = replicate_summaries(a.uns["binned"])
        group = 1 - rc  # 1 = the "healthy" group, so positive t = higher/steeper in healthy
        run(lv, sl, group, sets, f"liver_{tag}", args.out, args.seed)
        print(f"rc=0 liver {tag}", flush=True)

elif args.mode == "legnini":
    import scanpy as sc
    import squidpy as sq

    a = sc.read_h5ad(f"{args.data}/legnini23.h5ad")
    shh = a[:, "SHH"].X
    shh = (shh.toarray() if hasattr(shh, "toarray") else np.asarray(shh)).ravel()
    a.obs["SHH+"] = (shh > np.percentile(shh, 99)).astype(str)
    sq.tl.var_by_distance(a, cluster_key="SHH+", groups="True", library_key="sample", covariates=["condition", "organoid"])
    dm = a.obsm["design_matrix"]
    dm["organoid"] = a.obs["organoid"].astype(str).values
    real = a.obs["condition"].astype(str).values
    genes = list(a.var_names)
    shh_targets = [g for g in ["GLI1", "PTCH1", "PTCH2", "HHIP", "FOXA2", "NKX2-2", "NKX6-1", "NKX6.1", "OLIG2", "NKX2-1", "NKX6.2", "NKX6-2"] if g in genes]
    dorsal = [g for g in ["PAX7", "PAX6", "PAX3", "IRX3", "MSX1", "MSX2", "DBX1", "DBX2", "GLI3", "BMP4", "WNT1"] if g in genes]
    sets = {"shh_targets": shh_targets, "dorsal": dorsal}
    rng = np.random.default_rng(1)
    for i in range(20):
        sets[f"random{i:02d}"] = list(rng.choice(genes, 8, replace=False))
    print("sets:", {k: v for k, v in sets.items() if not k.startswith("random")}, flush=True)
    orgs = sorted(set(dm["organoid"]))
    shh_orgs = sorted(set(dm["organoid"][real == "SHH"]))
    splits = [("real", set(shh_orgs))] + [(f"null{j:02d}", set(c)) for j, c in enumerate(itertools.combinations(orgs, 4)) if set(c) != set(shh_orgs) and set(orgs) - set(c) != set(shh_orgs)]
    for tag, grp in splits:
        dm["condition"] = np.where(dm["organoid"].isin(grp), "A_shh", "B_ctrl")
        dvg.pp.bin_obs(a, anchor="True", condition="condition", layer="log1p_norm", sample="organoid")
        lv, sl, rc = replicate_summaries(a.uns["binned"])
        run(lv, sl, 1 - rc, sets, f"legnini_{tag}", args.out, args.seed)
        print(f"rc=0 legnini {tag}", flush=True)

elif args.mode == "seaad_spike":
    import anndata as ad
    from scipy.sparse import issparse

    rng = np.random.default_rng(args.seed)
    a = ad.read_h5ad(f"{args.data}/seaad_mtg_merfish.h5ad")
    a = a[a.obs["Normalized_depth_from_pia"].notna()].copy()
    a.var_names = a.var["symbol"].astype(str).values
    a.var_names_make_unique()
    donor = a.obs["Donor_ID"].astype(str).to_numpy()
    group_a = set(rng.choice(np.array(sorted(set(donor))), size=14, replace=False))
    in_a = np.isin(donor, list(group_a))
    depth = a.obs["Normalized_depth_from_pia"].clip(0, 1).to_numpy()
    genes = np.array(a.var_names)
    perm = rng.permutation(len(genes))
    sizes = [0.02, 0.05, 0.1]  # small per-gene level shifts, below per-gene FDR power
    sets, effect = {}, pd.Series(0.0, index=genes)
    for k, e in enumerate(sizes):
        for j in range(2):  # two sets of 10 per size
            g = genes[perm[20 * k + 10 * j: 20 * k + 10 * (j + 1)]]
            effect[g] = e
            sets[f"spike{e:g}_{j}"] = list(g)
    for j in range(6):
        sets[f"null_{j}"] = list(genes[perm[60 + 10 * j: 70 + 10 * j]])
    X = a.layers["log1p"]
    X = np.asarray(X.toarray() if issparse(X) else X, dtype=float)
    X[in_a] += effect.to_numpy()[None, :]
    a.layers["spiked"] = X
    a.obsm["design_matrix"] = pd.DataFrame({"depth": depth, "condition": np.where(in_a, "A", "B"), "donor": donor}, index=a.obs_names)
    dvg.pp.bin_obs(a, anchor="depth", condition="condition", layer="spiked", sample="donor")
    lv, sl, rc = replicate_summaries(a.uns["binned"])
    run(lv, sl, 1 - rc, sets, f"seaad_spike_seed{args.seed}", args.out, args.seed)
    print(f"rc=0 seaad_spike {args.seed}", flush=True)
