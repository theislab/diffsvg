"""Summarise DESpace / tradeSeq competitor runs: hits at BH FDR < .05, nominal p < .05, marker ranks."""

import os

import pandas as pd

res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MARKERS = ["CYP3A4", "ASS1", "CYP2E1", "CYP1A2", "SDS", "HAL"]
splits = pd.read_csv(os.path.join(res, "splits.csv"))
real = splits[splits.split == "real"].set_index("donor")["group"]
nulls = [f"null{s}" for s in range(10)]

# How far each random split is from the disease grouping (diseased donors in group A, of 4 in A).
ov = {s: int((real[splits[(splits.split == s) & (splits.group == "A")].donor] == "diseased").sum()) for s in nulls}
print("diseased donors in group A (of 4):", ov)

methods = {"DESpace default (~cond*cluster)": ("despace_default_{}.csv", "PValue", "FDR", "gene_id"),
           "DESpace + donor term": ("despace_donor_{}.csv", "PValue", "FDR", "gene_id"),
           "tradeSeq conditionTest": ("tradeseq_{}.csv", "pvalue", "FDR", "gene")}
rows, ranks, per = [], {}, {}
for name, (pat, p, q, g) in methods.items():
    def load(s):
        f = os.path.join(res, pat.format(s))
        return pd.read_csv(f) if os.path.exists(f) else None
    r = load("real")
    if r is None:
        continue
    nl = [d for d in map(load, nulls) if d is not None]
    hits = [int((d[q] < .05).sum()) for d in nl]
    nom = [float((d[p].dropna() < .05).mean()) for d in nl]
    per[name] = hits
    rows.append({"method": name, "n_null": len(nl), "real_FDR": int((r[q] < .05).sum()), "null_FDR_mean": sum(hits) / len(hits),
                 "null_FDR_max": max(hits), "real_p05": float((r[p].dropna() < .05).mean()),
                 "null_p05_mean": sum(nom) / len(nom), "null_p05_max": max(nom)})
    rk = r.assign(rank=r[p].rank(method="min")).set_index(g)
    ranks[name] = {m: (int(rk.loc[m, "rank"]), float(rk.loc[m, q])) for m in MARKERS if m in rk.index}
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nper-split FDR hits:", per)
print("\nmarker (rank by p, FDR) on real contrast:")
for k, v in ranks.items():
    print(f"  {k}: " + ", ".join(f"{m} {a} ({b:.2g})" for m, (a, b) in v.items()))
