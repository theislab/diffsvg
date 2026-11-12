"""Summarise old-vs-new validation runs (see run.py) into report tables. Run with the repo's venv python."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

R = Path(__file__).parent / "results"
SHH_TARGETS = ["GLI1", "PTCH1", "HHIP", "FOXA2", "NKX2-2", "NKX6-1", "OLIG2", "FOXA1"]
DORSAL = ["PAX7", "PAX3", "MSX1", "MSX2", "OLIG3", "ZIC1"]


def load(code, exp):
    f = R / f"{code}_{exp}.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def calls(df):
    """Hit counts per p-value column present (raw < 0.05 and BH < 0.05)."""
    out = {}
    for col in ("parallel_pval", "wald_pval", "empirical_pval", "gamma_pval"):
        if col in df:
            out[f"{col} < .05"] = int((df[col] < 0.05).sum())
            adj = col + "_adj"
            if adj in df:
                out[f"{adj} < .05"] = int((df[adj] < 0.05).sum())
    out["genes"] = int(df.filter(like="pval").notna().any(axis=1).sum())
    return out


def section(title):
    print(f"\n## {title}\n")


section("Hit counts (no true differences: synth_null*, legnini_null*)")
rows = {}
for f in sorted(R.glob("*.csv")):
    code, exp = f.stem.split("_", 1)
    rows[(exp, code)] = calls(pd.read_csv(f, index_col=0))
print(pd.DataFrame(rows).T.sort_index().fillna("").to_markdown())

section("Regression: same explicit configuration, old vs new (parallel test)")
for exp in (
    "legnini_parallel_pooled",
    "legnini_parallel_slides",
    "xenium_parallel",
    "synth_null_parallel",
    "legnini_null0_parallel_slides",
):
    o, n = load("old", exp), load("new", exp)
    if o is None or n is None:
        continue
    common = o.index.intersection(n.index)
    for col in ("parallel_stat", "parallel_pval", "shape_diff", "slope_diff", "integral"):
        a, b = o.loc[common, col], n.loc[common, col]
        ok = a.notna() & b.notna()
        print(
            f"{exp:32s} {col:14s} max|diff| {np.abs(a[ok] - b[ok]).max():.3g}  "
            f"spearman {a[ok].corr(b[ok], method='spearman'):.4f}  n={ok.sum()}"
        )

section("Old default vs new default: rank agreement and known biology (Legnini)")
o, n = load("old", "legnini_default"), load("new", "legnini_default")
if o is not None and n is not None:
    o_rank = o["T"].rank(ascending=False)  # what the old default ranked on (empirical/gamma null of T)
    n_rank = n["parallel_stat"].rank(ascending=False)
    print("spearman(old T rank, new parallel_stat rank):", round(o_rank.corr(n_rank, method="spearman"), 3))
    present = [g for g in SHH_TARGETS + DORSAL if g in n.index]
    tab = pd.DataFrame(
        {
            "old rank (T)": o_rank[present],
            "new rank": n_rank[present],
            "new BH": n["parallel_pval_adj"][present].round(4),
        }
    )
    print(tab.to_markdown())
    print("old top 15:", ", ".join(o_rank.nsmallest(15).index))
    print("new top 15:", ", ".join(n_rank.nsmallest(15).index))

section("Xenium: expression bias and far-end concentration")
xen = os.environ.get("DIFFSVG_XENIUM")
if xen:
    a = sc.read_h5ad(xen)
    sc.pp.filter_cells(a, min_counts=10)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    mean_expr = pd.Series(np.asarray(a.X.mean(axis=0)).ravel(), index=a.var_names)
    quint = pd.qcut(mean_expr.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    for code, exp, col in (
        ("old", "xenium_default", "empirical_pval_adj"),
        ("old", "xenium_default", "gamma_pval_adj"),
        ("old", "xenium_parallel", "parallel_pval_adj"),
        ("new", "xenium_default", "parallel_pval_adj"),
        ("new", "xenium_parallel", "parallel_pval_adj"),
    ):
        df = load(code, exp)
        if df is None or col not in df:
            continue
        hit = (df[col] < 0.05).reindex(mean_expr.index)
        share = hit.groupby(quint, observed=True).mean().round(2).tolist()
        print(f"{code} {exp:18s} {col:20s} hits {int(hit.sum()):4d}  share by expression quintile {share}")
    for code, exp in (("old", "xenium_default"), ("old", "xenium_parallel"), ("new", "xenium_default")):
        f = R / f"{code}_{exp}_curves.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        d = z["dsvg_preds"]
        grid = np.linspace(0, 1, d.shape[1])
        shape = d - np.nanmean(d, axis=1, keepdims=True)
        far = (shape[:, grid > 0.9] ** 2).sum(axis=1) / (shape**2).sum(axis=1)
        print(f"{code} {exp:18s} median share of squared shape difference at x > 0.9: {np.nanmedian(far):.2f}")
