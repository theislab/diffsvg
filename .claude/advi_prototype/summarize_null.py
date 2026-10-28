"""Summarise results/null_permutation.csv: ADVI and Laplace against the exact posterior."""

import pandas as pd

d = pd.read_csv("results/null_permutation.csv")
pd.set_option("display.width", 200)
print("=== permutation null (identical curves by construction): share of p < 0.05")
perm = d[d.kind == "perm"]
print(perm.groupby(["gene", "method"])[["p0.0", "p0.1"]].agg(lambda p: (p < 0.05).mean()).round(3).unstack().to_string())
print(f"({perm.rep.nunique()} permutations per gene)")

print("\n=== agreement with the exact posterior on the same permutation")
ex = perm[perm.method == "exact"].set_index(["gene", "rep"])
rows = []
for (g, m), grp in perm[perm.method != "exact"].groupby(["gene", "method"]):
    grp = grp.set_index(["gene", "rep"])
    ref = ex.loc[grp.index]
    ratio = grp.W / ref.W
    rows.append({"gene": g, "method": m, "W/exact median": ratio.median(), "W/exact 5%": ratio.quantile(0.05), "W/exact 95%": ratio.quantile(0.95),
                 "|dE| max": (grp.E - ref.E).abs().max(), "max |p0.0 diff|": (grp["p0.0"] - ref["p0.0"]).abs().max(),
                 "max |p0.1 diff|": (grp["p0.1"] - ref["p0.1"]).abs().max(), "khat max": grp.vi_khat.max(), "ess min": grp.vi_ess.min()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== real data, ADVI over seeds (exact in the first row)")
real = d[d.kind == "real"]
rows = []
for g, grp in real.groupby("gene"):
    e = grp[grp.method == "exact"].iloc[0]
    for m in ("laplace", "advi"):
        s = grp[grp.method == m]
        rows.append({"gene": g, "method": m, "n": len(s), "exact W": e.W, "W mean": s.W.mean(), "W cv": s.W.std() / s.W.mean(),
                     "exact p0.1": e["p0.1"], "p0.1 min": s["p0.1"].min(), "p0.1 max": s["p0.1"].max(),
                     "khat max": s.vi_khat.max(), "ess min": s.vi_ess.min()})
print(pd.DataFrame(rows).round(4).to_string(index=False))
