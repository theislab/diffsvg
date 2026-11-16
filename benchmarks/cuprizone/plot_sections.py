import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import anndata as ad, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from celltypes import CELLTYPE
a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/cuprizone_xenium.h5ad"), backed="r")
ct = a.obs["cluster"].astype(str).map(CELLTYPE).fillna("other").to_numpy(); xy = a.obsm["spatial"]; sec = a.obs["section"].to_numpy()
fig, ax = plt.subplots(2, 5, figsize=(25, 10))
for i, s in enumerate(sorted(set(sec), key=lambda x: ("CupRap" in x, x))):
    m = sec == s; o = ax.flat[i]
    o.scatter(*xy[m].T, s=0.05, c="0.85", rasterized=True)
    for t, c in (("oligo", "tab:blue"), ("microglia", "tab:green"), ("ependymal", "red"), ("choroid", "orange")):
        mm = m & (ct == t); o.scatter(*xy[mm].T, s=0.3 if t != "ependymal" else 1, c=c, rasterized=True)
    o.set_title(f"{s} n={m.sum()}"); o.set_aspect("equal"); o.invert_yaxis()
fig.savefig(os.path.join(os.path.dirname(__file__), "results", "sections.png"), dpi=60)
