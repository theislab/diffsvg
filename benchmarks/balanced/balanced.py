"""Joint slide model on balanced replicate splits (true nulls) and the real contrast; saves stats with effective df.

    python balanced.py --data liver|legnini --split I|real --test parallel|level|slope --datadir D --out O
Balanced: Legnini group A = 2 SHH + 2 Ctrl organoids (18 splits); liver group A = 2 fibrotic + 2 healthy donors (18).
"""
import argparse, itertools, os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, pandas as pd
import diffsvg as dvg

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--split", required=True); p.add_argument("--test", required=True)
p.add_argument("--datadir", required=True); p.add_argument("--out", required=True)
p.add_argument("--n-hyper3", type=int, default=None, help="grid points per replicate scale (default: package default)")
args = p.parse_args()

if args.data == "legnini":
    import scanpy as sc, squidpy as sq
    a = sc.read_h5ad(f"{args.datadir}/legnini23.h5ad")
    shh = a[:, "SHH"].X; shh = (shh.toarray() if hasattr(shh, "toarray") else np.asarray(shh)).ravel()
    a.obs["SHH+"] = (shh > np.percentile(shh, 99)).astype(str)
    sq.tl.var_by_distance(a, cluster_key="SHH+", groups="True", library_key="sample", covariates=["condition", "organoid"])
    dm = a.obsm["design_matrix"]; dm["organoid"] = a.obs["organoid"].astype(str).values
    real = a.obs["condition"].astype(str).values
    unit, pos, neg, anchor, layer = "organoid", sorted(set(dm["organoid"][real == "SHH"])), sorted(set(dm["organoid"][real == "Ctrl"])), "True", "log1p_norm"
    labels = dm["organoid"].to_numpy()
else:
    import anndata as ad, scanpy as sc
    from sklearn.neighbors import NearestNeighbors
    a = ad.read_h5ad(f"{args.datadir}/liver_fibrosis_merfish.h5ad")
    a.X = a.layers["csc_counts"].astype(np.float32); sc.pp.normalize_total(a); sc.pp.log1p(a)
    a.var_names = a.var["symbol"].astype(str).values; a.var_names_make_unique()
    ct = a.obs["Cell_Type_final"].astype(str).to_numpy(); dist = np.full(a.n_obs, np.nan)
    for s in a.obs["section_uid"].unique():
        m = (a.obs["section_uid"] == s).to_numpy(); xy = a.obsm["spatial"][m]
        dist[m] = NearestNeighbors(n_neighbors=1).fit(xy[ct[m] == "Cholangiocyte"]).kneighbors(xy)[0][:, 0]
    hep = np.isin(ct, ["Central Hep", "Portal Hep", "Fibrotic Hep 1", "Fibrotic Hep 2"]); a = a[hep].copy()
    labels = a.obs["donor_id"].astype(str).to_numpy(); real = a.obs["condition"].astype(str).to_numpy()
    a.obsm["design_matrix"] = pd.DataFrame({"anchor": np.clip(dist[hep] / 500, 0, 1), "donor": labels}, index=a.obs_names)
    dm = a.obsm["design_matrix"]
    unit, pos, neg, anchor, layer = "donor", sorted(set(labels[real == "diseased"])), sorted(set(labels[real == "healthy"])), "anchor", None

splits = [set(x) | set(y) for x in itertools.combinations(pos, 2) for y in itertools.combinations(neg, 2)]
if args.data == "legnini":  # 4 vs 4: a split and its complement are the same partition
    uniq = []
    for s_ in splits:
        if not any(s_ == u or s_ == set(pos) | set(neg) - u for u in uniq):
            uniq.append(s_)
    splits = uniq
grp = set(pos) if args.split == "real" else splits[int(args.split)]
dm["condition"] = np.where(np.isin(labels, list(grp)), "A", "B")
dvg.pp.bin_obs(a, anchor=anchor, condition="condition", layer=layer, sample=unit)
kw = {} if args.n_hyper3 is None else {"sample_kwargs": {"n_hyper3": args.n_hyper3}}
st, _, _ = dvg.tl.fit_model(a, dvg.tl.setup_model(a), inplace=False, progressbar=False, test=args.test, **kw)
os.makedirs(args.out, exist_ok=True)
tag = f"{args.data}_{args.split}_{args.test}" + (f"_h{args.n_hyper3}" if args.n_hyper3 else "")
st.to_csv(f"{args.out}/{tag}.csv")
print(f"rc=0 {tag}: n_splits {len(splits)}, FDR {int((st[args.test + '_pval_adj'] < .05).sum())}", flush=True)
