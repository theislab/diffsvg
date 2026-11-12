"""Regression tests for the statistical fixes: each fails on the behaviour it replaced."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diffsvg as dvg
from diffsvg.pp.utils import _condition_levels
from diffsvg.tl.model import _benjamini_hochberg, _fittable_genes, _gamma_null_pvalues, pattern_wald_test
from diffsvg.tl.stability import _format_value


def _adata(x, cond=None, n_genes=2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    cond = np.where(np.arange(n) % 2 == 0, "a", "b") if cond is None else cond
    X = (np.sin(3 * np.nan_to_num(x))[:, None] + rng.normal(size=(n, n_genes))).astype(np.float32)
    adata = ad.AnnData(X, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


def test_bh_ignores_missing_pvalues():
    p = np.array([0.01, np.nan, 0.04, 0.5])
    adj = _benjamini_hochberg(p)
    assert np.isnan(adj[1])
    np.testing.assert_allclose(adj[[0, 2, 3]], _benjamini_hochberg(p[[0, 2, 3]]))


def test_gamma_null_is_calibrated_on_a_pure_null():
    """Fitting the truncated pool as if untruncated gave ~10-13% p < 0.05 here."""
    T = stats.chi2.rvs(10, size=20000, random_state=np.random.default_rng(0))
    p = _gamma_null_pvalues(T, null_quantile=0.9)
    assert 0.04 < (p < 0.05).mean() < 0.06


def test_gamma_null_does_not_flag_a_fixed_top_fraction():
    """The empirical null gave the top 10% the minimum p-value whatever the data."""
    T = stats.chi2.rvs(10, size=2000, random_state=np.random.default_rng(1))
    adj = _benjamini_hochberg(_gamma_null_pvalues(T))
    assert (adj < 0.05).sum() < 20


def test_wald_test_degrees_of_freedom_are_the_rank():
    """Draws of a difference curve spanned by 3 functions: df 3, not the number of grid points."""
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(3, 40))
    coef = rng.normal(size=(4000, 3))
    stat, pval, df, _ = pattern_wald_test(np.zeros((4000, 40)), coef @ basis)
    assert df == 3
    assert 0 < pval <= 1


def test_wald_test_pvalue_keeps_small_values():
    """``1 - chi2.cdf`` rounds p-values below ~1e-16 to 0, which ties the strongest genes."""
    rng = np.random.default_rng(0)
    d = np.array([6.3, 6.3]) + rng.normal(size=(20000, 2))  # W ~ 80 on 2 df, p ~ 1e-17
    _, pval, df, _ = pattern_wald_test(np.zeros((20000, 2)), d)
    assert df == 2
    assert 0 < pval < 1e-12


def test_condition_levels_follow_categorical_order_and_skip_missing():
    values = pd.Categorical(["ctrl", "treated", None, "ctrl"], categories=["treated", "ctrl", "unused"])
    assert _condition_levels(values) == ["treated", "ctrl"]
    assert _condition_levels(pd.Series(["b", "a", np.nan])) == ["a", "b"]


def test_bin_obs_leaves_out_cells_without_a_condition():
    x = np.random.default_rng(0).random(2000)
    x[[0, 1, 2, 3]] = 0.0, 0.0, 1.0, 1.0
    cond = np.where(np.arange(2000) % 2 == 0, "a", "b").astype(object)
    cond[5:15] = None
    b = dvg.pp.bin_obs(_adata(x, cond), anchor="anchor", condition="condition", n_bins=20, inplace=False)
    assert b.n_obs.sum() == 2000 - 10


def test_bin_obs_rejects_anchor_values_outside_unit_interval():
    x = np.random.default_rng(0).random(500) * 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dvg.pp.bin_obs(_adata(x), anchor="anchor", condition="condition", inplace=False)


def test_bin_obs_works_when_no_cell_sits_near_zero():
    """patsy took the boundary knots from the data, so a gap at x = 0 crashed the grid basis."""
    x = 0.2 + 0.8 * np.random.default_rng(0).random(2000)
    b = dvg.pp.bin_obs(_adata(x), anchor="anchor", condition="condition", n_bins=20, inplace=False)
    assert b.B_pred.shape == (21, b.n_bsplines)


def test_binning_defaults_are_contiguous_and_overlapping_bins_are_weighted():
    x = np.random.default_rng(0).random(2000)
    x[[0, 1, 2, 3]] = 0.0, 0.0, 1.0, 1.0
    adata = _adata(x)
    assert dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", inplace=False).weighted
    sliding = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", window_radius=0.02, inplace=False)
    assert not sliding.weighted
    assert sliding.n_obs is not None and (sliding.n_obs > 0).all()


def test_constant_genes_are_not_fitted():
    Y = {"ok": np.array([0.0, 1.0, 2.0]), "zero": np.zeros(3), "nan": np.array([0.0, np.nan, 1.0])}
    with pytest.warns(UserWarning, match="2 gene"):
        assert _fittable_genes(list(Y), Y) == ["ok"]


def test_prior_overrides_get_distinct_labels():
    a = _format_value({"beta": ("normal", 0.0, 0.5)})
    b = _format_value({"beta": ("normal", 0.0, 2.0)})
    assert a != b


def test_volcano_leaves_out_unfitted_genes():
    import matplotlib

    matplotlib.use("Agg")
    df = pd.DataFrame(
        {"slope_diff": [0.5, -0.3, np.nan], "parallel_pval": [1e-4, 0.2, np.nan], "parallel_pval_adj": [1e-3, 0.3, np.nan]},
        index=["a", "b", "unfitted"],
    )
    ax = dvg.pl.plot_volcano(df, n_label=None)
    labels = {t.get_text() for t in ax.texts}
    assert "unfitted" not in labels


def test_default_test_is_parallel():
    import inspect

    assert inspect.signature(dvg.tl.fit_model).parameters["test"].default == "parallel"
    assert inspect.signature(dvg.tl.check_stability).parameters["test"].default == "parallel"
