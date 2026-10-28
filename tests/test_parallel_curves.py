import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diffsvg as dvg
from diffsvg.pp.utils import _bspline_basis_01
from diffsvg.tl.model import _slope_weights

K = 10


def _coefficients(curve):
    """Spline coefficients of `curve` on [0, 1] (least squares on a fine grid)."""
    x = np.linspace(0, 1, 2001)
    return np.linalg.lstsq(_bspline_basis_01(x, K, 3), curve(x), rcond=None)[0]


def _posterior(seed, scale=0.01):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(K, K))
    return rng.normal(size=K) * 0.1, A @ A.T / K * scale


def test_shape_diff_is_the_slope_difference_for_a_linear_gap():
    b = _coefficients(lambda x: 0.4 - 0.3 * x)
    assert np.sqrt(b @ dvg.tl.shape_metric(K, 3) @ b) == pytest.approx(0.3, rel=1e-6)
    assert _slope_weights(K, 3) @ b == pytest.approx(-0.3, rel=1e-6)


def test_level_differences_do_not_count():
    Q = dvg.tl.shape_metric(K, 3)
    b = _coefficients(lambda x: 0.7 + 0 * x)
    assert np.abs(Q @ np.ones(K)).max() < 1e-10
    assert b @ Q @ b == pytest.approx(0, abs=1e-12)
    assert _slope_weights(K, 3) @ b == pytest.approx(0, abs=1e-12)


def test_tau_zero_is_the_wald_test_on_first_differences():
    mean, cov = _posterior(0)
    _, stat, df, pval = dvg.tl.parallel_curves_test(mean, dvg.tl.shape_metric(K, 3), tau=0, cov=cov)
    D = np.diff(np.eye(K), axis=0)  # zero exactly for constant coefficients, i.e. parallel curves
    W = (D @ mean) @ np.linalg.solve(D @ cov @ D.T, D @ mean)
    assert df == K - 1
    assert stat == pytest.approx(W, rel=1e-8)
    assert pval == pytest.approx(stats.chi2.sf(W, K - 1), rel=1e-8)


def test_pvalue_is_continuous_at_zero_and_grows_with_tau():
    mean, cov = _posterior(1)
    Q = dvg.tl.shape_metric(K, 3)
    pvals = [dvg.tl.parallel_curves_test(mean, Q, tau=t, cov=cov)[3] for t in (0, 1e-9, 0.05, 0.1, 0.3)]
    assert pvals[1] == pytest.approx(pvals[0], rel=1e-6)
    assert np.all(np.diff(pvals) >= 0)


def test_draws_and_moments_give_the_same_result():
    mean, cov = _posterior(2)
    draws = np.random.default_rng(3).multivariate_normal(mean, cov, size=4000)
    Q = dvg.tl.shape_metric(K, 3)
    from_draws = dvg.tl.parallel_curves_test(draws, Q, tau=0.1)
    from_moments = dvg.tl.parallel_curves_test(draws.mean(0), Q, tau=0.1, cov=np.cov(draws, rowvar=False))
    assert from_draws == pytest.approx(from_moments)


def test_interval_null_is_exact_at_its_least_favourable_point():
    """A true shape difference of exactly tau along the best-measured direction rejects at the nominal rate."""
    _, cov = _posterior(4)
    Q = dvg.tl.shape_metric(K, 3)
    tau = 0.1
    evals, evecs = np.linalg.eigh(Q)
    L = evecs[:, 1:] * np.sqrt(evals[1:])  # Q = L L' on its K - 1 shape directions
    omega_vals, omega_vecs = np.linalg.eigh(L.T @ cov @ L)
    worst = L @ np.linalg.solve(L.T @ L, tau * omega_vecs[:, 0])  # shape_diff == tau, best-measured direction
    assert np.sqrt(worst @ Q @ worst) == pytest.approx(tau)

    rng = np.random.default_rng(5)
    estimates = rng.multivariate_normal(worst, cov, size=4000)
    pvals = np.array([dvg.tl.parallel_curves_test(b, Q, tau=tau, cov=cov)[3] for b in estimates])
    assert abs(np.mean(pvals < 0.05) - 0.05) < 0.015
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_spline_basis_at_reproduces_bin_obs():
    rng = np.random.default_rng(6)
    n = 3000
    x = rng.random(n)
    x[[0, 1]] = 0.0, 1.0
    cond = np.where(rng.random(n) < 0.5, "a", "b")
    adata = ad.AnnData(rng.normal(size=(n, 2)).astype(np.float32))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    binned = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", inplace=False)
    centers = binned.bin_centers
    assert binned.B.shape[0] == 2 * len(centers)  # no empty bins, so the rows are the grid twice
    np.testing.assert_allclose(dvg.pp.spline_basis_at(binned, centers), binned.B_pred, atol=1e-12)
    np.testing.assert_allclose(dvg.pp.spline_basis_at(binned, np.r_[centers, centers]), binned.B, atol=1e-12)
    with pytest.raises(ValueError, match="missing"):
        dvg.pp.spline_basis_at(binned, [0.5, np.nan])


def test_fit_model_rejects_unknown_test_and_negative_tau():
    with pytest.raises(ValueError, match="Unknown test"):
        dvg.tl.fit_model(None, None, test="nope")
    with pytest.raises(ValueError, match="tau"):
        dvg.tl.fit_model(None, None, test="parallel", tau=-0.1)


def test_plots_default_to_the_parallel_columns():
    genes = [f"g{i}" for i in range(6)]
    stats_df = pd.DataFrame(
        {
            "slope_diff": [-0.4, -0.1, 0.0, 0.05, 0.2, 0.5],
            "shape_diff": [0.4, 0.1, 0.02, 0.05, 0.2, 0.5],
            "integral": np.zeros(6),
            "parallel_pval": [0.0, 0.3, 0.9, 0.5, 0.01, 1e-8],  # 0.0 = underflow, drawn at the smallest p
            "parallel_pval_adj": [0.0, 0.4, 0.9, 0.6, 0.03, 1e-7],
        },
        index=genes,
    )
    ax = dvg.pl.plot_volcano(stats_df, n_label=2)
    assert "slope" in ax.get_xlabel()
    assert np.isfinite(np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])).all()
    assert dvg.pl.plot_pvalue_histogram(stats_df).get_xlabel() == "parallel_pval"
