import anndata as ad
import numpy as np
import pandas as pd
import pytest

import diffsvg as dvg

N_BINS = 20


def _cells(curves, n=3000, noise=0.5, seed=0, slides=None, slide_curve_sd=0.0):
    """One gene per entry of `curves` (functions of the anchor in [0, 1]); optional slides with their own curves."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    x[:2] = 0.0, 1.0
    cols = []
    slide = None if slides is None else rng.integers(0, slides, n)
    for f in curves:
        y = f(x) + rng.normal(0, noise, n)
        if slides is not None:
            amp = rng.normal(0, slide_curve_sd, (slides, 2))
            y += amp[slide, 0] * np.cos(np.pi * x) + amp[slide, 1] * np.cos(2 * np.pi * x) + rng.normal(0, 0.3, slides)[slide]
        cols.append(y)
    adata = ad.AnnData(np.column_stack(cols).astype(np.float32), var=pd.DataFrame(index=[f"g{i}" for i in range(len(curves))]))
    dm = {"anchor": x}
    if slides is not None:
        dm["slide"] = np.array([f"s{j}" for j in slide])
    adata.obsm["design_matrix"] = pd.DataFrame(dm, index=adata.obs_names)
    return adata


def _fit(adata, sample=None, n_bsplines=6, **kw):
    dvg.pp.bin_obs(adata, anchor="anchor", condition=None, n_bins=N_BINS, n_bsplines=n_bsplines, window_radius=None, sample=sample)
    dvg.tl.fit_profiles(adata, dvg.tl.setup_model(adata), progressbar=False, inference="exact", **kw)
    return dvg.tl.flatness_test(adata, **({"tau": 0.0} if sample is None else {}))


def test_flat_genes_are_not_inflated_and_trends_are_found():
    flat = [lambda x: np.zeros_like(x)] * 60
    adata = _cells(flat + [lambda x: 0.8 * x, lambda x: 0.6 * np.sin(np.pi * x)], seed=1)
    res = _fit(adata)
    p_flat = res["pval"].iloc[:60].to_numpy()
    assert (p_flat < 0.05).mean() <= 0.15  # 60 null genes: about 3 expected
    assert res["pval"].iloc[60:].max() < 1e-6  # a slope and an arch
    assert set(res.columns) == {"shape", "stat", "df", "df_denominator", "pval", "pval_adj"}
    assert "dsvg_profiles_flat_stat" in adata.var


def test_shape_of_a_straight_line_is_its_slope_and_level_drops_out():
    adata = _cells([lambda x: 0.5 * x, lambda x: 0.5 * x + 3.0], noise=0.05, seed=2)
    res = _fit(adata)
    np.testing.assert_allclose(res["shape"], 0.5, atol=0.03)
    np.testing.assert_allclose(res["stat"].iloc[0], res["stat"].iloc[1], rtol=0.05)


def test_tau_protects_small_shapes():
    adata = _cells([lambda x: 0.05 * x], n=20000, noise=0.05, seed=3)
    dvg.pp.bin_obs(adata, anchor="anchor", condition=None, n_bins=N_BINS, n_bsplines=6, window_radius=None)
    dvg.tl.fit_profiles(adata, dvg.tl.setup_model(adata), progressbar=False, inference="exact")
    assert dvg.tl.flatness_test(adata, tau=0.0)["pval"].iloc[0] < 1e-6
    assert dvg.tl.flatness_test(adata, tau=0.1)["pval"].iloc[0] > 0.5


def test_slides_absorb_slide_specific_curves():
    """Zero-mean slide curves: pooled, the curve looks non-flat; with slide effects it does not."""
    flat = [lambda x: np.zeros_like(x)] * 20
    adata = _cells(flat, n=4000, noise=0.3, seed=4, slides=6, slide_curve_sd=0.5)
    pooled = _fit(adata.copy())
    slides = _fit(adata, sample="slide", sample_kwargs={"n_hyper": 24})
    assert (pooled["pval"] < 0.05).mean() > 0.5
    assert (slides["pval"] < 0.05).mean() <= 0.2
    assert (slides["df_denominator"] == 5).all()
    assert "sample_level_sd" in adata.varm["dsvg_profiles_diagnostics"]


def test_slides_keep_a_shared_curve():
    adata = _cells([lambda x: 1.0 * x] * 3, n=4000, noise=0.3, seed=5, slides=6, slide_curve_sd=0.1)
    res = _fit(adata, sample="slide", sample_kwargs={"n_hyper": 24})
    assert res["pval"].max() < 0.01


def test_needs_a_stored_fit():
    adata = _cells([lambda x: x])
    with pytest.raises(KeyError, match="fit_profiles"):
        dvg.tl.flatness_test(adata)
