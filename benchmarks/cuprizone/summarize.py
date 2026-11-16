"""Summarize results/*.csv: hits at BH<.05 and nominal p<.05 rate per run; top genes on the real contrast."""
import glob, os, re
import pandas as pd
R = os.path.join(os.path.dirname(__file__), "results")
rows = []
for f in sorted(glob.glob(os.path.join(R, "*.csv"))):
    ct, con, mode, test = re.match(r"(\w+?)_(disease|null\d+)_(pooled|section)_(\w+)\.csv", os.path.basename(f)).groups()
    s = pd.read_csv(f, index_col=0)
    rows.append(dict(ct=ct, test=test, mode=mode, real=con == "disease", con=con, bh=int((s[f"{test}_pval_adj"] < .05).sum()), nom=(s[f"{test}_pval"] < .05).mean()))
d = pd.DataFrame(rows)
print(d[d.real].to_string())
print(d[~d.real].groupby(["ct", "test", "mode"]).agg(n=("bh", "size"), bh_mean=("bh", "mean"), bh_max=("bh", "max"), splits_with_hits=("bh", lambda x: (x > 0).sum()), nom_mean=("nom", "mean")).round(3).to_string())
for ct in ("oligo", "microglia"):
    for test in ("slope", "level"):
        s = pd.read_csv(os.path.join(R, f"{ct}_disease_section_{test}.csv"), index_col=0).sort_values(f"{test}_pval").head(15)
        print(f"\n{ct} {test}"); print(s[[f"{test}_diff", f"{test}_pval", f"{test}_pval_adj"]].round(4).to_string())
