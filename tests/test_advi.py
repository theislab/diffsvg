import anndata as ad
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import diffsvg as dvg
from diffsvg.tl import model as M

K = 10


class _Gaussian:
    """A Gaussian 'density' with the interface of `_FlatDensity`, for checking the diagnostics without PyMC."""

    def __init__(self, mean, cov):
        self.mean, self.precision = mean, np.linalg.inv(cov)

    def logp(self, x):
        d = x - self.mean
        return -0.5 * d @ self.precision @ d


def _target(p=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    return rng.normal(size=p), A @ A.T / p + 0.1 * np.eye(p)


def test_pareto_k_follows_the_tail_of_the_weights():
    rng = np.random.default_rng(0)
    # p = N(0, 1), q = N(0, s^2): the ratios p / q have a Pareto tail with k = 1 - s^2 for s < 1, and are bounded for s >= 1
    ks = []
    for s in (0.9, 0.7, 0.5):
        z = s * rng.standard_normal(20000)
        ks.append(M._pareto_k(np.exp(-0.5 * z**2 * (1 - 1 / s**2) + np.log(s))))
    assert ks[0] == pytest.approx(1 - 0.9**2, abs=0.1)
    assert ks[0] < ks[1] < ks[2]
    assert np.isnan(M._pareto_k(np.ones(500)))  # equal weights: nothing to fit, the approximation is exact
    assert M._pareto_k(np.r_[np.zeros(990), np.logspace(0, 3, 10)]) == np.inf  # a few draws carry all the weight


def test_pareto_k_matches_arviz():
    base = pytest.importorskip("arviz_stats.base")
    w = np.random.default_rng(1).pareto(3.0, size=3000) + 1
    expected = float(base.array_stats.pareto_khat(w[None, :], r_eff=1.0, tail="right", log_weights=False))
    assert M._pareto_k(w) == pytest.approx(expected, rel=1e-8)


def test_importance_diagnostics_see_a_wrong_covariance():
    mean, cov = _target()
    density = _Gaussian(mean, cov)
    _, ess = M._importance_diagnostics(density, mean, cov, 2000, seed=0)
    assert ess > 0.99  # exact: all ratios equal (the Gaussian's constant shifts every log-weight alike)
    k, ess = M._importance_diagnostics(density, mean, 0.1 * cov, 2000, seed=0)
    assert k > 0.7 and ess < 0.1  # too narrow: heavy-tailed ratios
    _, ess = M._importance_diagnostics(density, mean, 6.0 * cov, 2000, seed=0)
    assert ess < 0.1  # too wide: the ratios are bounded, so only the effective sample size sees it
    _, ess = M._importance_diagnostics(density, mean, 1.5 * cov, 2000, seed=0)
    assert 0.5 < ess < 0.9  # moderately too wide: visible, not alarming


def test_tril_params_round_trip_through_pymc():
    _, cov = _target(p=4, seed=2)
    with pm.Model() as model:
        pm.Normal("a", shape=2)
        pm.Normal("b", shape=2)
        approx = pm.FullRankADVI(model=model).approx
    approx.groups[0].shared_params["L_tril"].set_value(M._tril_params(cov))
    np.testing.assert_allclose(approx.cov.eval(), cov, rtol=1e-8, atol=1e-10)


def _adata(n=3000, seed=0):
    """Two genes on [0, 1]: one with a shape difference between the conditions, one without."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    x[:4] = 0.0, 0.0, 1.0, 1.0
    cond = np.where(np.arange(n) % 2 == 0, "a", "b")
    c = (cond == "b").astype(float)
    shared = np.sin(3 * x)
    X = np.column_stack([shared + 0.8 * c * (x - 0.5), shared]) + rng.normal(0, 0.6, (n, 2))
    adata = ad.AnnData(X.astype(np.float32), var=pd.DataFrame(index=["shaped", "parallel"]))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


def _closed_form_beta_1(adata, setup, gene):
    """Posterior mean and covariance of beta_1 for the per-cell Normal model, with sigma set to its residual estimate."""
    B, _, c, Y, _ = M._fit_inputs(adata, setup.binned, use_binned=False)
    X = np.column_stack([np.ones(len(c)), B, c, B * c[:, None]])
    y = Y[gene]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma2 = np.mean((y - X @ coef) ** 2)
    V = np.linalg.inv(X.T @ X / sigma2 + np.eye(X.shape[1]))
    m = V @ X.T @ y / sigma2
    block = slice(K + 2, 2 * K + 2)
    return m[block], V[block, block]


def test_laplace_start_matches_the_closed_form():
    adata = _adata()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20)
    setup = dvg.tl.setup_model(adata)
    B, Bm, c, Y, init = M._fit_inputs(adata, setup.binned, use_binned=False)
    model = M._build_gene_model(Y["shaped"], B, Bm, c, K, init["shaped"], record_mu=False)
    density = M._FlatDensity(model)
    assert density.names == ["intercept", "beta", "beta_1", "categorical_beta", "sigma_log__"]

    mode, cov = M._laplace(density)
    block = slice(1 + K, 1 + 2 * K)  # beta_1 comes after intercept and beta
    mean_exact, cov_exact = _closed_form_beta_1(adata, setup, "shaped")
    sd = np.sqrt(np.diag(cov_exact))
    assert np.abs(mode[block] - mean_exact).max() < 0.02 * sd.max()
    np.testing.assert_allclose(np.diag(cov[block, block]), np.diag(cov_exact), rtol=0.02)
    assert np.abs(density.dlogp(mode)).max() < 1e-3  # a mode


@pytest.mark.parametrize("gene", ["shaped", "parallel"])
def test_advi_agrees_with_the_closed_form_on_cells(gene):
    adata = _adata()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20)
    setup = dvg.tl.setup_model(adata)
    stats, preds, curves = dvg.tl.fit_model(
        adata, setup, use_binned=False, test="parallel", tau=0.1, inference="advi", inplace=False, progressbar=False
    )
    row = stats.loc[gene]

    mean, cov = _closed_form_beta_1(adata, setup, gene)
    shape_diff, stat, _, pval = dvg.tl.parallel_curves_test(mean, dvg.tl.shape_metric(K, 3), tau=0.1, cov=cov)
    assert row["shape_diff"] == pytest.approx(shape_diff, abs=0.03)
    assert row["parallel_stat"] == pytest.approx(stat, rel=0.15)
    assert row["vi_khat"] < 0.7 and row["vi_ess"] > 0.5
    # the variational fit reports its own diagnostics instead of the MCMC ones
    assert not {"n_divergent", "rhat_max", "ess_bulk_min"} & set(stats.columns)
    assert np.isfinite(curves["cond0_mean"].to_numpy()).all()


def test_advi_on_weighted_bins_stays_at_its_laplace_start():
    adata = _adata()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20, window_radius=None)
    setup = dvg.tl.setup_model(adata)
    kwargs = {"test": "parallel", "inference": "advi", "inplace": False, "progressbar": False}
    polished, _, _ = dvg.tl.fit_model(adata, setup, **kwargs)
    laplace, _, _ = dvg.tl.fit_model(adata, setup, n=0, **kwargs)
    np.testing.assert_allclose(polished["shape_diff"], laplace["shape_diff"], atol=0.03)
    np.testing.assert_allclose(polished["parallel_stat"], laplace["parallel_stat"], rtol=0.2)
    assert (polished["vi_ess"] > 0.5).all() and (laplace["vi_ess"] > 0.5).all()


def test_fit_model_rejects_unknown_inference_and_settings():
    with pytest.raises(ValueError, match="Unknown inference"):
        dvg.tl.fit_model(None, None, inference="svgd")
    with pytest.raises(ValueError, match="Unknown setting"):
        dvg.tl.fit_model(None, None, inference="advi", chains=2)  # a NUTS setting
    with pytest.raises(ValueError, match="init"):
        dvg.tl.fit_model(None, None, inference="advi", init="map")
    with pytest.raises(ValueError, match="learning rates"):
        dvg.tl.fit_model(None, None, inference="advi", final_learning_rate=0.0)
    with pytest.raises(ValueError, match="slide_scales"):
        dvg.tl.fit_model(None, None, inference="advi", slide_scales="joint")


def test_mu_is_only_recorded_when_asked_for():
    x = np.linspace(0, 1, 30)
    B, c = np.random.default_rng(0).normal(size=(30, K)), (np.arange(30) % 2).astype(float)
    kept = M._build_gene_model(x, B, B * c[:, None], c, K, 0.0)
    left_out = M._build_gene_model(x, B, B * c[:, None], c, K, 0.0, record_mu=False)
    assert "mu" in [v.name for v in kept.deterministics]
    assert "mu" not in [v.name for v in left_out.deterministics]
