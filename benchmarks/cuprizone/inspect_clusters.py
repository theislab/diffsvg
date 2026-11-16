import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import anndata as ad, numpy as np, pandas as pd, scanpy as sc
a = ad.read_h5ad(os.path.expanduser("~/Documents/data/diffsvg/cuprizone_xenium.h5ad"))
sc.pp.normalize_total(a); sc.pp.log1p(a)
mk = ["Mog","Mag","Opalin","Sox10","Enpp6","Pdgfra","Cspg4","Gpr17","Cx3cr1","Tmem119","Siglech","Aif1","Cd68","Trem2","Itgax","Spp1","Lgals3","Cd52","Laptm5","Foxj1","Spag16","Ttr","Clic6","Aqp4","Gfap","Slc17a7","Gad1","Cldn5","Acta2","Dcn","Ascl1","Dlx2","Sp9","Mki67","Egfr"]
mk=[g for g in mk if g in a.var_names]
df = pd.DataFrame(a[:, mk].X.toarray(), columns=mk); df["cl"]=a.obs["cluster"].to_numpy()
mean = df.groupby("cl").mean().round(2)
mean["n"]=df.groupby("cl").size()
ct = pd.crosstab(a.obs["cluster"], a.obs["condition"])
mean["frac_cup"]=(ct["cuprizone"]/ct.sum(1)).round(2)
pd.set_option("display.width",400); pd.set_option("display.max_columns",60)
print(mean.to_string())
