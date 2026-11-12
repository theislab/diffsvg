"""Exact inference matches the sampled posterior of the same model."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import diffsvg as dvg
from diffsvg.tl import _exact


def _adata(n=3000, seed=0, n_genes=4):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    x[:4] = 0.0, 0.0, 1.0, 1.0
    cond = np.where(np.arange(n) % 2 == 0, "a", "b")
    shift = np.where(cond == "b", 1.0, 0.0)[:, None] * np.array([0.0, 0.3, 0.0, 0.0])[:n_genes]
    slope = np.where(cond == "b", 1.0, 0.0)[:, None] * np.array([0.0, 0.0, 0.8, 0.0])[:n_genes]
    X = 1.0 + np.sin(3 * x)[:, None] + shift + slope * x[:, None] + 0.5 * rng.normal(size=(n, n_genes))
    adata = ad.AnnData(X.astype(np.float32), var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


def test_exact_posterior_matches_a_direct_computation_at_fixed_sigma():
    """With sigma's prior a narrow spike, the posterior is the textbook conjugate one at that sigma."""
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    y = X @ np.array([0.5, 1.0, -1.0, 0.2]) + 0.3 * rng.normal(size=200)
    XtWX, XtWY, yWy = _exact.sufficient_statistics(X, y[:, None])
    post = _exact.posterior(XtWX, XtWY, yWy, 200, np.zeros(4), np.ones(4), ("halfnormal", 1.0))
    sigma2 = post.sigma2_mean[0]
    precision = XtWX / sigma2 + np.eye(4)
    mean = np.linalg.solve(precision, XtWY[:, 0] / sigma2)
    np.testing.assert_allclose(post.mean[0], mean, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(post.cov[0], np.linalg.inv(precision), rtol=0.05)
    assert post.edge_mass[0] < 1e-6


def test_exact_draws_have_the_exact_moments():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 2))])
    y = X @ np.array([0.0, 1.0, 0.5]) + rng.normal(size=100)
    post = _exact.posterior(*_exact.sufficient_statistics(X, y[:, None]), 100, np.zeros(3), np.ones(3), ("halfnormal", 1.0))
    theta = _exact.draws(post, 0, 200000, np.random.default_rng(2))
    np.testing.assert_allclose(theta.mean(axis=0), post.mean[0], atol=0.01)
    np.testing.assert_allclose(np.cov(theta, rowvar=False), post.cov[0], atol=0.005)


@pytest.mark.parametrize("use_binned", [True, False])
def test_exact_fit_matches_nuts(use_binned):
    adata = _adata()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20)
    setup = dvg.tl.setup_model(adata)
    exact, _, _ = dvg.tl.fit_model(adata, setup, inference="exact", inplace=False, progressbar=False, use_binned=use_binned)
    nuts, _, _ = dvg.tl.fit_model(
        adata, setup, inference="nuts", inplace=False, progressbar=False, use_binned=use_binned,
        sample_kwargs={"draws": 2000, "tune": 1000, "chains": 2},
    )  # fmt: skip
    for col in ("shape_diff", "slope_diff", "integral", "categorical_beta_mean"):
        np.testing.assert_allclose(exact[col], nuts[col], atol=0.02)
    np.testing.assert_allclose(exact["parallel_stat"], nuts["parallel_stat"], rtol=0.15, atol=1.0)
    assert (exact["parallel_pval_adj"] < 0.05).tolist() == (nuts["parallel_pval_adj"] < 0.05).tolist()


def test_auto_inference_picks_exact_and_falls_back():
    adata = _adata(n=800)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=10)
    stats, _, _ = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), inplace=False, progressbar=False)
    assert "exact_edge_mass" in stats
    laplace = dvg.tl.setup_model(adata, priors={"beta": ("laplace", 0.0, 1.0)})
    with pytest.raises(ValueError, match="exact"):
        dvg.tl.fit_model(adata, laplace, inference="exact", inplace=False, progressbar=False)


def test_exact_profiles_match_nuts():
    adata = _adata()
    dvg.pp.bin_obs(adata, anchor="anchor", condition=None, n_bins=20)
    setup = dvg.tl.setup_model(adata)
    dvg.tl.fit_profiles(adata, setup, inference="exact", progressbar=False, profiles_key="ex")
    dvg.tl.fit_profiles(adata, setup, inference="nuts", progressbar=False, profiles_key="nu", sample_kwargs={"draws": 2000})
    np.testing.assert_allclose(adata.varm["ex_mean"], adata.varm["nu_mean"], atol=0.02)
