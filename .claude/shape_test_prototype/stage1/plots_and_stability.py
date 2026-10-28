import sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import holoviews as hv
warnings.filterwarnings("ignore")
import harness  # noqa: F401
import diffsvg as dvg
from make_data import make_adata

SK = {"draws": 500, "tune": 500, "chains": 2}
adata = make_adata(n_per_cond=3000, nan_frac=0.03)
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
dvg.tl.fit_model(adata, setup, use_binned=False, progressbar=False, sample_kwargs=SK, test="parallel", tau=0.1, n_workers=4)
print("uns params:", adata.uns["dsvg_stats_params"])
print(adata.varm["dsvg_stats"][["slope_diff", "shape_diff", "parallel_pval", "parallel_pval_adj", "integral"]].round(4).to_string())

ax = dvg.pl.plot_volcano(adata.varm["dsvg_stats"], n_label=4, figsize=(4, 4))
ax.get_figure().savefig("volcano_parallel.png", dpi=110, bbox_inches="tight")
print("volcano x/y labels:", repr(ax.get_xlabel()), repr(ax.get_ylabel()))
ax = dvg.pl.plot_volcano(adata.varm["dsvg_stats"], x="integral", n_label=4, figsize=(4, 4))
ax.get_figure().savefig("volcano_level_vs_shape.png", dpi=110, bbox_inches="tight")
ax = dvg.pl.plot_pvalue_histogram(adata.varm["dsvg_stats"])
print("histogram x label:", ax.get_xlabel())
hv.save(dvg.pl.plot_prediction_curve(adata, list(adata.var_names), ncols=4, figsize=(4, 3)), "curves_parallel.png", dpi=70)

res = dvg.tl.check_stability(adata, "anchor", "condition", grid={"n_bsplines": [8, 10]}, genes=None, n_seeds=1,
                             rank_by="parallel_stat", test="parallel", tau=0.1, sample_kwargs={"draws": 200, "tune": 200, "chains": 2},
                             progressbar=False)
print("check_stability with the parallel test:\n", res.stats.round(2).to_string())
