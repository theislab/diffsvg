"""Wall-clock of exact inference (fit_model, default parallel test) on real and synthetic data.

DIFFSVG_LEGNINI=... DIFFSVG_XENIUM=... python exact_speed.py
"""

import os
import time

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import diffsvg as dvg


def timed(adata, anchor, use_binned, **bin_kwargs):
    t0 = time.perf_counter()
    dvg.pp.bin_obs(adata, anchor=anchor, condition="condition", layer="log1p_norm", **bin_kwargs)
    t1 = time.perf_counter()
    stats, _, _ = dvg.tl.fit_model(
        adata, dvg.tl.setup_model(adata), inplace=False, progressbar=False, use_binned=use_binned
    )
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, float(stats["exact_edge_mass"].max())


def legnini():
    a = sc.read_h5ad(os.environ["DIFFSVG_LEGNINI"])
    x = a[:, "SHH"].X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x).ravel()
    a.obs["SHH+"] = (x > np.percentile(x, 99)).astype(str)
    sq.tl.var_by_distance(a, cluster_key="SHH+", groups="True", library_key="sample", covariates="condition")
    return a, "True"


def xenium():
    a = sc.read_h5ad(os.environ["DIFFSVG_XENIUM"])
    sc.pp.filter_cells(a, min_counts=10)
    a.layers["log1p_norm"] = a.X.copy()
    sc.pp.normalize_total(a, layer="log1p_norm")
    sc.pp.log1p(a, layer="log1p_norm")
    sq.tl.var_by_distance(
        a, cluster_key="Dentate_Gyrus", groups="anchor", library_key="condition", covariates="condition"
    )
    return a, "anchor"


def synthetic(n_cells, n_genes, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_cells) ** 1.5
    X = (1 + np.sin(3 * x)[:, None] + 0.5 * rng.standard_normal((n_cells, n_genes))).astype(np.float32)
    a = ad.AnnData(X, layers={"log1p_norm": X})
    a.obsm["design_matrix"] = pd.DataFrame(
        {"anchor": x, "condition": rng.choice(["a", "b"], n_cells)}, index=a.obs_names
    )
    return a, "anchor"


rows = []
for name, loader in (("legnini (44k cells x 88 genes)", legnini), ("xenium (113k cells x 347 genes)", xenium)):
    a, anchor = loader()
    for use_binned in (True, False):
        b, f, edge = timed(a, anchor, use_binned)
        rows.append((name, "bins" if use_binned else "cells", b, f, edge))
for n_cells, n_genes in ((100_000, 1000), (200_000, 1000)):
    a, anchor = synthetic(n_cells, n_genes)
    for use_binned in (True, False):
        b, f, edge = timed(a, anchor, use_binned)
        rows.append(
            (f"synthetic ({n_cells // 1000}k cells x {n_genes} genes)", "bins" if use_binned else "cells", b, f, edge)
        )

table = pd.DataFrame(rows, columns=["data", "fit on", "bin_obs s", "fit_model s", "max edge mass"])
print(table.to_markdown(index=False, floatfmt=".3g"))
