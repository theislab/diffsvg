"""Small simulated AnnData shaped like the real inputs (design_matrix with anchor in [0, 1] and two conditions)."""
import anndata as ad
import numpy as np
import pandas as pd


def make_adata(n_per_cond=2000, nan_frac=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_cond
    x = np.where(rng.random(n) < 0.8, rng.beta(1.5, 2.5, n), rng.random(n))
    x[np.argmin(x)], x[np.argmax(x)] = 0.0, 1.0  # anchor is min-max normalized in the real data
    cond = np.repeat(["ctrl", "treat"], n_per_cond)
    c = (cond == "treat").astype(float)
    curves = {
        "same": 0.6 * np.exp(-3 * x),
        "level": 0.6 * np.exp(-3 * x) + 0.3 * c,
        "crossing": 0.6 * np.exp(-3 * x) + 0.4 * c * (x - 0.5),
        "peakshift": 0.8 * np.exp(-3 * np.abs(x - 0.15 * c)),
    }
    X = np.column_stack(list(curves.values())) + rng.normal(0, 1, (n, len(curves)))
    if nan_frac:
        x[rng.random(n) < nan_frac] = np.nan  # e.g. the anchor cells themselves (legnini)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X.astype(np.float32), obs=obs, var=pd.DataFrame(index=list(curves)))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=obs.index)
    return adata
