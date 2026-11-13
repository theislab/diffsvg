"""Per-gene GAM (P-spline) models for differential spatially/pseudotemporally variable genes.

`setup_model` reads the output of `diffsvg.pp.bin_obs` from ``adata.uns["binned"]``; `fit_model` fits one
model per gene in parallel, predicts both condition curves on a fixed grid, tests them against each other
and writes the result to ``adata.varm``.

Parallelization: workers receive only plain arrays (the shared design matrices plus the gene's `Y`) and
build their own `pm.Model` with `_build_gene_model`. A built model cannot be shipped instead (PyTensor
graph nodes do not pickle), and every worker is a separate process, so each compiles its own.
"""

import contextlib
import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize
from scipy.stats import chi2, f, gamma, ncf, ncx2
from scipy.stats import t as student_t
from tqdm.auto import tqdm

from diffsvg.pp.utils import BinnedData, _bspline_basis_01, _expression, spline_basis_at
from diffsvg.tl import _exact

try:  # the `pymc` extra: only needed for inference="nuts"
    import arviz as az
    import pymc as pm
except ImportError:
    az = pm = None


def _require_pymc(inference):
    if pm is None:
        raise ImportError(
            f"inference={inference!r} needs PyMC, which is an optional dependency: pip install 'diffsvg[pymc]'. "
            "The default inference='auto' uses the exact posterior wherever the model allows it."
        )


# ============================================================
# 1) Priors and likelihoods
# ============================================================
# Registries, so alternatives can be added without touching the model builders. A prior is a tuple
# ``(distribution, *params)``, parameters in the order listed below.

_PRIOR_DISTRIBUTIONS = {
    "normal": ("Normal", ("mu", "sigma")),
    "laplace": ("Laplace", ("mu", "b")),
    "studentt": ("StudentT", ("nu", "mu", "sigma")),
    "halfnormal": ("HalfNormal", ("sigma",)),
    "halfcauchy": ("HalfCauchy", ("beta",)),
    "exponential": ("Exponential", ("lam",)),
}

#: Default priors; ``setup_model(priors=...)`` overrides any of them.
DEFAULT_PRIORS = {
    "intercept": ("normal", 0.0, 1.0),
    "beta": ("normal", 0.0, 1.0),
    "beta_1": ("normal", 0.0, 1.0),
    "categorical_beta": ("normal", 0.0, 1.0),
    "sigma": ("halfnormal", 1.0),
    # Only used by fits with `sample` (see `bin_obs`): how far a slide's level / curve strays from its condition's.
    "sample_level_sd": ("halfnormal", 1.0),
    "sample_curve_sd": ("halfnormal", 1.0),
}


def _resolve_priors(priors):
    """Merge a partial prior spec (e.g. ``{"beta": ("normal", 0.0, 2.0)}``) over `DEFAULT_PRIORS` and validate it."""
    merged = dict(DEFAULT_PRIORS)
    for name, spec in (priors or {}).items():
        if name not in DEFAULT_PRIORS:
            raise ValueError(f"Unknown prior {name!r}. Available: {sorted(DEFAULT_PRIORS)}")
        merged[name] = tuple(spec)

    for name, spec in merged.items():
        distribution, *params = spec
        if distribution not in _PRIOR_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {distribution!r} for prior {name!r}. Available: {sorted(_PRIOR_DISTRIBUTIONS)}"
            )
        expected = _PRIOR_DISTRIBUTIONS[distribution][1]
        if len(params) != len(expected):
            raise ValueError(
                f"Prior {name!r}: {distribution!r} takes {len(expected)} parameter(s) {expected}, got {len(params)}."
            )
    return merged


def _draw_prior(name, spec, shape=None, initval=None):
    """Instantiate one prior from its ``(distribution, *params)`` tuple."""
    distribution, *params = spec
    dist_name, param_names = _PRIOR_DISTRIBUTIONS[distribution]
    dist = getattr(pm, dist_name)
    kwargs = dict(zip(param_names, params, strict=True))
    if shape is not None:
        kwargs["shape"] = shape
    if initval is not None:
        kwargs["initval"] = initval
    return dist(name, **kwargs)


def _normal_likelihood(mu, observed, total_size, priors, weights=None, ss_within=None):
    sigma = _draw_prior("sigma", priors["sigma"], initval=0.5)
    if weights is None:
        return pm.Normal("D", mu=mu, sigma=sigma, observed=observed, total_size=total_size)
    # A row that averages n cells has variance sigma^2 / n, sigma being the cell-level noise.
    D = pm.Normal("D", mu=mu, sigma=sigma / pm.math.sqrt(weights), observed=observed, total_size=total_size)
    if ss_within is not None:
        # What the per-cell likelihood has on top of the row means: the spread of
        # the cells around their row's mean, `ss` over `df` = n_cells - n_rows.
        ss, df = ss_within
        pm.Potential("within_rows", -0.5 * (2.0 * df * pm.math.log(sigma) + ss / sigma**2))
    return D


_LIKELIHOODS = {
    "normal": _normal_likelihood,
}


# ============================================================
# 2) Model setup
# ============================================================


@dataclass
class DiffSVGModelSetup:
    """Config bundle returned by `setup_model` and consumed by `fit_model`.

    Holds no `pm.Model`: `fit_model` builds each gene's model inside its worker. To inspect one gene's
    model, call `_build_gene_model` directly.

    Attributes
    ----------
    binned
        The `diffsvg.pp.utils.BinnedData` this setup was built from.
    likelihood
        Observation model every gene's model is built with.
    priors
        Prior spec passed to `setup_model`, merged over `DEFAULT_PRIORS`.
    """

    binned: BinnedData
    likelihood: str
    priors: dict | None = None

    def __post_init__(self):
        self.priors = _resolve_priors(self.priors)


def _build_gene_model(
    Y,
    B,
    B_masked,
    condition_arr,
    n_bsplines,
    intercept_initval,
    likelihood="normal",
    priors=None,
    weights=None,
    ss_within=None,
    record_mu=True,
    sample_index=None,
    n_samples=None,
    sample_scales=None,
):
    """Build the P-spline GAM for a single gene.

    `B`/`B_masked`/`condition_arr` are shared across genes; only `Y` and `intercept_initval` change per
    gene. Building is cheap (graph construction only), so every worker builds its own.

    - `weights` (cells per row) and `ss_within` (``(within-row sum of squares, n_cells - n_rows)``) come
      from contiguous bins: rows are weighted by their cell count and the noise is the cell-level one.
      `weights` lives in a ``"w_data"`` container so predictions can set it per grid point.
    - `record_mu` keeps the row-wise mean as the ``"mu"`` deterministic (one value per row and draw; off
      in `fit_model`, nothing reads it).
    - `sample_index` (the slide of every row) adds a non-centred random level and random curve per slide,
      ``sample_level_sd * z_level[slide] + B @ (sample_curve_sd * z_curve[slide])``, multiplied by a
      ``"sample_on"`` container that a prediction grid sets to 0 for the population-level curves.
      `sample_scales` ``(level_sd, curve_sd)`` fixes the two scales instead (empirical Bayes).
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    likelihood_fn = _LIKELIHOODS[likelihood]
    priors = _resolve_priors(priors)

    with pm.Model() as gene_model:
        Y_data = pm.Data("Y_data", np.asarray(Y, dtype=float))
        B_data = pm.Data("B_data", B)
        condition_data = pm.Data("condition_data", np.asarray(condition_arr, dtype=float))
        B_masked_data = pm.Data("B_masked_data", B_masked)

        intercept = _draw_prior("intercept", priors["intercept"], initval=intercept_initval)
        beta = _draw_prior("beta", priors["beta"], shape=n_bsplines, initval=np.zeros(n_bsplines))
        beta_1 = _draw_prior("beta_1", priors["beta_1"], shape=n_bsplines, initval=np.zeros(n_bsplines))
        categorical_beta = _draw_prior("categorical_beta", priors["categorical_beta"], initval=0.0)

        mu = (
            intercept
            + pm.math.dot(B_data, beta)
            + categorical_beta * condition_data
            + pm.math.dot(B_masked_data, beta_1)
        )
        if sample_index is not None:
            if sample_scales is None:
                level_sd = _draw_prior("sample_level_sd", priors["sample_level_sd"], initval=0.1)
                curve_sd = _draw_prior("sample_curve_sd", priors["sample_curve_sd"], initval=0.1)
            else:
                level_sd, curve_sd = (float(v) for v in sample_scales)
            level_z = pm.Normal("sample_level_z", 0.0, 1.0, shape=n_samples)
            curve_z = pm.Normal("sample_curve_z", 0.0, 1.0, shape=(n_samples, n_bsplines))
            slide = pm.Data("sample_data", np.asarray(sample_index, dtype="int64"))
            on = pm.Data("sample_on", np.ones(len(sample_index)))
            mu = mu + on * (level_sd * level_z[slide] + pm.math.sum(B_data * (curve_sd * curve_z)[slide], axis=1))
        if record_mu:
            mu = pm.Deterministic("mu", mu)

        w_data = None if weights is None else pm.Data("w_data", np.asarray(weights, dtype=float))
        likelihood_fn(mu, Y_data, len(Y), priors, weights=w_data, ss_within=ss_within)

    return gene_model


def _build_profile_model(
    Y, B, n_bsplines, intercept_initval, likelihood="normal", priors=None, weights=None, ss_within=None
):
    """`_build_gene_model` without the `condition` covariate and its spline interaction (for `fit_profiles`)."""
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    likelihood_fn = _LIKELIHOODS[likelihood]
    priors = _resolve_priors(priors)

    with pm.Model() as gene_model:
        Y_data = pm.Data("Y_data", np.asarray(Y, dtype=float))
        B_data = pm.Data("B_data", B)

        intercept = _draw_prior("intercept", priors["intercept"], initval=intercept_initval)
        beta = _draw_prior("beta", priors["beta"], shape=n_bsplines, initval=np.zeros(n_bsplines))

        mu = pm.Deterministic("mu", intercept + pm.math.dot(B_data, beta))

        w_data = None if weights is None else pm.Data("w_data", np.asarray(weights, dtype=float))
        likelihood_fn(mu, Y_data, len(Y), priors, weights=w_data, ss_within=ss_within)

    return gene_model


def setup_model(adata, likelihood="normal", priors=None):
    """Configure the per-gene GAM, given `adata` pre-processed by `diffsvg.pp.bin_obs`.

    This validates `adata`/`likelihood` and bundles them into a
    `DiffSVGModelSetup` - it does **not** build any `pm.Model`. Actual model
    construction happens lazily, per gene, once `fit_model` starts fitting
    (see `DiffSVGModelSetup`), so this is cheap regardless of gene count.

    Parameters
    ----------
    adata
        Annotated data matrix that `diffsvg.pp.bin_obs` was already run on
        (i.e. ``"binned" in adata.uns``).
    likelihood
        Observation model. Only ``"normal"`` (the default) is implemented.
    priors
        Overrides for any of `DEFAULT_PRIORS`, as ``{name: (distribution,
        *params)}`` - e.g. ``{"beta": ("normal", 0.0, 2.0), "sigma":
        ("exponential", 1.0)}``. Anything not named keeps its default, so
        the models are unchanged unless asked. Validated here rather than
        inside a worker, so a typo fails immediately.

    Returns
    -------
    DiffSVGModelSetup
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    if "binned" not in adata.uns:
        raise KeyError("adata.uns['binned'] not found - run `diffsvg.pp.bin_obs(adata, ...)` first.")

    binned = BinnedData(**adata.uns["binned"])
    return DiffSVGModelSetup(binned=binned, likelihood=likelihood, priors=priors)


# ============================================================
# 3) Fitting (parallel across genes) + posterior predictive
# ============================================================


@dataclass
class _FitShared:
    """The plain-array payload shared by every gene (no PyMC objects, see the module docstring)."""

    B: np.ndarray
    B_masked: np.ndarray
    condition_fit: np.ndarray
    n_bsplines: int
    likelihood: str
    priors: dict
    bin_centers: np.ndarray
    B_pred: np.ndarray
    B_pred_masked_0: np.ndarray
    B_pred_masked_1: np.ndarray
    condition_pred_0: np.ndarray
    condition_pred_1: np.ndarray
    # Which condition test to run, see `_TESTS`. `shape_metric`/`slope_weights`
    # are only filled for the parallel-curves test.
    test: str = "wald"
    tau: float = 0.1
    shape_metric: np.ndarray | None = None
    slope_weights: np.ndarray | None = None
    # Cells per row, and per grid point for predictions, when fitting weighted
    # contiguous bins (see `_weighting`); `None` otherwise.
    weights: np.ndarray | None = None
    # How the posterior is obtained, see `_INFERENCE`.
    inference: str = "nuts"
    # The slide of every fitted row and the denominator degrees of freedom of the test, for fits with `sample`.
    sample_index: np.ndarray | None = None
    n_samples: int | None = None
    df_denominator: int | None = None
    # The anchor range the tests are computed over, see `_anchor_support`.
    support: tuple = (0.0, 1.0)


@dataclass
class _ProfileShared:
    """`_FitShared`'s counterpart for the condition-free model - same contract, fewer arrays."""

    B: np.ndarray
    n_bsplines: int
    likelihood: str
    priors: dict
    B_pred: np.ndarray
    weights: np.ndarray | None = None


def _fitted_cells(design, binned):
    """Which cells a fit on single cells uses: those with an anchor value, a condition and (if any) a sample."""
    keep = np.isfinite(design[binned.anchor].to_numpy(dtype=float))
    if binned.has_condition:
        keep &= design[binned.condition].notna().to_numpy()
    if binned.sample is not None:
        keep &= design[binned.sample].notna().to_numpy()
    return keep


def _sample_inputs(adata, binned, use_binned):
    """The slide of every fitted row and the number of slides, or ``(None, None)`` for a fit without `sample`."""
    if binned.sample is None:
        return None, None
    if use_binned:
        return np.asarray(binned.sample_index, dtype=int), binned.n_samples
    design = adata.obsm[binned.design_matrix]
    position = {label: i for i, label in enumerate(binned.sample_labels)}
    index = design[binned.sample].astype(str).map(position)[_fitted_cells(design, binned)]
    if index.isna().any():
        raise ValueError(
            f"`{binned.sample}` holds samples that were not seen when binning (expected {binned.sample_labels}) - "
            "re-run `diffsvg.pp.bin_obs`."
        )
    return index.to_numpy(dtype=int), binned.n_samples


def _fit_inputs(adata, binned, use_binned):
    """The targets and design matrices to fit on: the binned aggregates, or one row per cell.

    For cells, the basis is the binned one evaluated per cell (`diffsvg.pp.spline_basis_at`), so the
    coefficients stay valid on the prediction grid, and `Y` is read from `adata`. Cells without an anchor
    value or a condition are left out, as in the binning.

    Returns ``(B, B_masked, condition_fit, Y, intercept_initvals)``.
    """
    if use_binned:
        return binned.B, binned.B_masked, binned.condition_fit, binned.Y, binned.intercept_initvals

    design = adata.obsm[binned.design_matrix]
    anchor_vals = design[binned.anchor].to_numpy(dtype=float)
    keep = _fitted_cells(design, binned)
    # The binned basis evaluated per cell. A basis rebuilt from the cells would
    # be centred on the cells rather than the bins, and predicting with
    # `binned.B_pred` would then shift every curve by a constant.
    B = spline_basis_at(binned, anchor_vals[keep])

    condition_fit, B_masked = None, None
    if binned.has_condition:
        code_map = {label: float(code) for code, label in binned.condition_categories.items()}
        condition_fit = design[binned.condition].map(code_map).to_numpy(dtype=float)[keep]
        if np.isnan(condition_fit).any():
            raise ValueError(
                f"`condition` column {binned.condition!r} holds values that were not seen when binning "
                f"(expected {sorted(code_map)}) - re-run `diffsvg.pp.bin_obs`."
            )
        B_masked = B * condition_fit[:, None]

    X = _expression(adata, binned.genes, binned.layer)[keep]
    Y = {gene: X[:, i] for i, gene in enumerate(binned.genes)}
    # As for bins (mean expression at anchor 0), but no cell sits exactly at 0, so the first bin stands in.
    first_bin = np.asarray(binned.bin_assignment)[keep] == 0
    intercept_initvals = {
        gene: float(y[first_bin].mean()) if first_bin.any() else float(y.mean()) for gene, y in Y.items()
    }
    return B, B_masked, condition_fit, Y, intercept_initvals


def _weighting(binned, use_binned):
    """What a fit on weighted contiguous bins needs on top of `_fit_inputs`, or `None`s otherwise.

    Returns ``(weights, ss_within, n_obs_grid)``: the cells behind each row,
    per gene the within-row sum of squares with its degrees of freedom
    (``n_cells - n_rows``), and the cells per grid point. Predictions use the
    latter, so a predicted value is a bin mean just like a fitted row. Fits on
    the cells are unweighted. Overlapping bins are weighted by their cell
    counts too (a window with 2 cells must not count as much as one with
    500, which otherwise bends the curve at the sparse ends of the range),
    but have no within-row term: their rows share cells.
    """
    if not use_binned:
        return None, None, None
    if not binned.weighted:
        if binned.n_obs is None:  # binned before the counts were recorded for overlapping bins
            return None, None, None
        return np.asarray(binned.n_obs, dtype=float), None, None
    weights = np.asarray(binned.n_obs, dtype=float)
    df = float(weights.sum() - len(weights))
    ss_within = {gene: (float(binned.ss_within[gene]), df) for gene in binned.genes}
    return weights, ss_within, np.asarray(binned.n_obs_grid, dtype=float)


@contextlib.contextmanager
def _quiet_fds():
    """Redirect the OS-level stdout/stderr file descriptors to `os.devnull`.

    Unlike `contextlib.redirect_stdout`, this also catches compiled extensions (nutpie's Rust sampler)
    writing straight to the fds. Exceptions still propagate; the fds are restored before anything prints them.
    """
    saved = [os.dup(fd) for fd in (1, 2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        for fd, saved_fd in zip((1, 2), saved, strict=True):
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull)


@contextlib.contextmanager
def _quiet_pymc():
    """Silence PyMC/PyTensor/nutpie's logging, warnings and raw fd output in the process that samples.

    Runs inside each worker; the outer `tqdm` bar lives in the parent process and is unaffected.
    """
    loggers = [logging.getLogger(name) for name in ("pymc", "pytensor", "nutpie")]
    previous_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), _quiet_fds():
            warnings.simplefilter("ignore")
            yield
    finally:
        for logger, level in zip(loggers, previous_levels, strict=True):
            logger.setLevel(level)


# ------------------------------------------------------------
# Posterior inference by sampling (the exact path lives in `diffsvg.tl._exact`)
# ------------------------------------------------------------
# `_INFERENCE` maps a sampler name to a function that runs inside ``with model`` and returns the posterior as
# `InferenceData` plus a dict of diagnostic columns. NUTS is only needed for non-normal coefficient priors.


def _sample_nuts(model, sample_kwargs, diag_vars):
    """Sample the posterior with NUTS. `diag_vars` are the parameters the convergence diagnostics are reported for."""
    idata = pm.sample(**sample_kwargs)
    n_divergent = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    rhat_ds = az.rhat(idata, var_names=diag_vars)
    ess_ds = az.ess(idata, var_names=diag_vars, method="bulk")
    columns = {
        "n_divergent": n_divergent,
        "rhat_max": float(max(np.asarray(rhat_ds[v]).max() for v in rhat_ds.data_vars)),
        "ess_bulk_min": float(min(np.asarray(ess_ds[v]).min() for v in ess_ds.data_vars)),
    }
    return idata, columns


_INFERENCE = {
    "nuts": _sample_nuts,
}


def _posterior_draws(idata, name):
    """Posterior draws of one variable, flattened over chains: ``(n_draws,)`` or ``(n_draws, dim)``.

    `idata` is an `arviz.InferenceData` or, for exact inference, a dict ``{name: draws}``.
    """
    if isinstance(idata, dict):
        return idata[name]
    values = idata.posterior[name].stack(sample=("chain", "draw")).to_numpy()
    return values.T if values.ndim == 2 else values


def _mean_curve_draws(idata, shared):
    """Posterior draws of both condition curves on the grid, ``(n_draws, n_grid)`` each.

    The mean function ``intercept + B beta + condition * categorical_beta + B_masked beta_1`` at every
    draw, without observation noise and without slide effects.
    """
    intercept = _posterior_draws(idata, "intercept")
    beta = _posterior_draws(idata, "beta")
    beta_1 = _posterior_draws(idata, "beta_1")
    cb = _posterior_draws(idata, "categorical_beta")
    base = intercept[:, None] + beta @ shared.B_pred.T
    draws0 = base + cb[:, None] * shared.condition_pred_0[None, :] + beta_1 @ shared.B_pred_masked_0.T
    draws1 = base + cb[:, None] * shared.condition_pred_1[None, :] + beta_1 @ shared.B_pred_masked_1.T
    return draws0, draws1


def _fit_and_predict_gene(gene, Y, intercept_initval, shared, sample_kwargs, ss_within=None):
    """Build and fit one gene's model in this worker, then predict both condition curves on the grid.

    Returns ``(gene, stats_row, curve_row)``: `stats_row` holds the test's columns and the shared ones;
    `curve_row` holds the mean difference curve ``"delta"``, per condition the mean curve and its 94% band
    (``"cond{0,1}_{mean,ci_low,ci_high}"``) and the posterior means of ``"beta"`` and ``"beta_1"``.
    `ss_within` is the gene's within-row term (see `_weighting`).
    """

    def build(sample_scales=None):
        return _build_gene_model(
            Y,
            shared.B,
            shared.B_masked,
            shared.condition_fit,
            shared.n_bsplines,
            intercept_initval,
            shared.likelihood,
            shared.priors,
            weights=shared.weights,
            ss_within=ss_within,
            record_mu=False,
            sample_index=shared.sample_index,
            n_samples=shared.n_samples,
            sample_scales=sample_scales,
        )

    model = build()
    slides = shared.sample_index is not None

    with _quiet_pymc():
        with model:
            diag_vars = ["intercept", "beta", "beta_1", "categorical_beta", "sigma"]
            idata, fit_columns = _INFERENCE[shared.inference](model, sample_kwargs, diag_vars)
            if slides:
                fit_columns = {
                    **fit_columns,
                    "sample_level_sd": float(idata.posterior["sample_level_sd"].mean()),
                    "sample_curve_sd": float(idata.posterior["sample_curve_sd"].mean()),
                }
    return _summarise_gene(gene, idata, shared, fit_columns)


def _condition_maps(shared):
    """Linear maps from ``theta = (intercept, beta, categorical_beta, beta_1)`` to the two grid curves."""
    n_grid, K = shared.B_pred.shape
    ones = np.ones((n_grid, 1))
    A0 = np.hstack([ones, shared.B_pred, shared.condition_pred_0[:, None], shared.B_pred_masked_0])
    A1 = np.hstack([ones, shared.B_pred, shared.condition_pred_1[:, None], shared.B_pred_masked_1])
    return A0, A1


def _summarise_gene(gene, idata, shared, fit_columns, moments=None):
    """Curves, test and summary columns of one gene from its posterior; see `_fit_and_predict_gene`.

    `idata` holds the posterior draws (InferenceData or a dict). `moments` ``(mean, cov)`` of ``theta``
    (exact inference) replace the draw estimates wherever only moments are needed: the means, the tests.
    """
    # The condition curves are the posterior of the mean function on the grid (slide effects off, i.e.
    # population level). No observation noise: the bands are credible bands of the curve, and the Wald
    # test sees the covariance of the coefficients rather than that of new observations.
    draws0, draws1 = _mean_curve_draws(idata, shared)
    test_row, delta_mean = _TESTS[shared.test][0](draws0, draws1, idata, shared, moments)
    if moments is None:
        cond0_mean, cond1_mean = draws0.mean(axis=0), draws1.mean(axis=0)
    else:
        A0, A1 = _condition_maps(shared)
        cond0_mean, cond1_mean = A0 @ moments[0], A1 @ moments[0]
    cond0_ci_low, cond0_ci_high = np.quantile(draws0, [0.03, 0.97], axis=0)
    cond1_ci_low, cond1_ci_high = np.quantile(draws1, [0.03, 0.97], axis=0)
    integral = float(np.trapezoid(delta_mean, x=shared.bin_centers))

    post_cb = _posterior_draws(idata, "categorical_beta")
    ci_low, ci_high = (float(v) for v in np.quantile(post_cb, [0.03, 0.97]))
    K = shared.n_bsplines
    if moments is None:
        cb_mean = float(post_cb.mean())
        beta_mean = _posterior_draws(idata, "beta").mean(axis=0)
        beta_1_mean = _posterior_draws(idata, "beta_1").mean(axis=0)
    else:
        cb_mean = float(moments[0][K + 1])
        beta_mean, beta_1_mean = moments[0][1 : K + 1], moments[0][K + 2 :]

    stats_row = {
        "gene": gene,
        **test_row,
        "integral": integral,
        "categorical_beta_mean": cb_mean,
        "categorical_beta_ci_low": ci_low,
        "categorical_beta_ci_high": ci_high,
        **fit_columns,
    }
    curve_row = {
        "delta": delta_mean,
        "cond0_mean": cond0_mean,
        "cond0_ci_low": cond0_ci_low,
        "cond0_ci_high": cond0_ci_high,
        "cond1_mean": cond1_mean,
        "cond1_ci_low": cond1_ci_low,
        "cond1_ci_high": cond1_ci_high,
        "beta": np.asarray(beta_mean),
        "beta_1": np.asarray(beta_1_mean),
    }
    return gene, stats_row, curve_row


def _exact_settings(sample_kwargs):
    settings = {"draws": 1000, "random_seed": 42, "n_grid": 81, "n_hyper": 72}
    given = dict(sample_kwargs or {})
    ignored = sorted(set(given) - set(settings))
    if ignored:
        warnings.warn(
            f"inference='exact' ignores the sampler settings {ignored} (it uses only draws, random_seed, n_grid, n_hyper).",
            UserWarning,
            stacklevel=3,
        )
    settings.update({k: v for k, v in given.items() if k in settings})
    return settings


def _exact_fit(design, Y, genes, weights, ss_within, priors, prior_names, n_bsplines, settings):
    """`_exact.posterior` for all `genes` on the design `design` (columns ordered as `prior_names`)."""
    Ymat = np.column_stack([np.asarray(Y[g], dtype=float) for g in genes])
    XtWX, XtWY, yWy = _exact.sufficient_statistics(design, Ymat, weights)
    sizes = {"intercept": 1, "categorical_beta": 1, "beta": n_bsplines, "beta_1": n_bsplines}
    m0 = np.concatenate([[priors[n][1]] * sizes[n] for n in prior_names])
    s0 = np.concatenate([[priors[n][2]] * sizes[n] for n in prior_names])
    extra_df = extra_ss = None
    if ss_within is not None:
        extra_df = ss_within[genes[0]][1]
        extra_ss = np.array([ss_within[g][0] for g in genes])
    return _exact.posterior(
        XtWX, XtWY, yWy, design.shape[0], m0, s0, priors["sigma"], extra_df, extra_ss, n_grid=settings["n_grid"]
    )


def _exact_slide_results(shared, Y, genes, ss_within, settings):
    """Exact inference for the two-condition model with slide effects."""
    priors = _resolve_priors(shared.priors)
    K = shared.n_bsplines
    fixed = np.column_stack([np.ones(shared.B.shape[0]), shared.B, shared.condition_fit, shared.B_masked])
    X = _exact.slide_design(fixed, shared.B, np.asarray(shared.sample_index, dtype=int), shared.n_samples)
    sizes = {"intercept": 1, "categorical_beta": 1, "beta": K, "beta_1": K}
    names = ("intercept", "beta", "categorical_beta", "beta_1")
    m0 = np.concatenate([[priors[n][1]] * sizes[n] for n in names])
    s0 = np.concatenate([[priors[n][2]] * sizes[n] for n in names])
    Ymat = np.column_stack([np.asarray(Y[g], dtype=float) for g in genes])
    extra_df = extra_ss = None
    if ss_within is not None:
        extra_df = ss_within[genes[0]][1]
        extra_ss = np.array([ss_within[g][0] for g in genes])
    res = _exact.slide_posterior(
        X, Ymat, shared.weights, fixed.shape[1], shared.n_samples, K, m0, s0, priors["sigma"],
        priors["sample_level_sd"], priors["sample_curve_sd"], extra_df, extra_ss, n_draws=settings["draws"],
        seeds=np.random.SeedSequence(settings["random_seed"]).spawn(len(genes)), n_hyper=settings.get("n_hyper", 72),
    )  # fmt: skip
    results = []
    for i, gene in enumerate(genes):
        theta = res["draws"][i]
        draws = {
            "intercept": theta[:, 0],
            "beta": theta[:, 1 : K + 1],
            "categorical_beta": theta[:, K + 1],
            "beta_1": theta[:, K + 2 :],
        }
        fit_columns = {
            "sigma_mean": float(np.sqrt(res["sigma2_mean"][i])),
            "sample_level_sd": float(res["level_sd_mean"][i]),
            "sample_curve_sd": float(res["curve_sd_mean"][i]),
            "exact_edge_mass": float(res["edge_mass"][i]),
        }
        results.append(_summarise_gene(gene, draws, shared, fit_columns, moments=(res["mean"][i], res["cov"][i])))
    return results


def _exact_gene_results(shared, Y, genes, ss_within, settings):
    """Exact inference for the two-condition model: ``[(gene, stats_row, curve_row), ...]``."""
    if shared.sample_index is not None:
        return _exact_slide_results(shared, Y, genes, ss_within, settings)
    design = np.column_stack([np.ones(shared.B.shape[0]), shared.B, shared.condition_fit, shared.B_masked])
    names = ("intercept", "beta", "categorical_beta", "beta_1")
    post = _exact_fit(
        design, Y, genes, shared.weights, ss_within, _resolve_priors(shared.priors), names, shared.n_bsplines, settings
    )
    K = shared.n_bsplines
    seeds = np.random.SeedSequence(settings["random_seed"]).spawn(len(genes))
    results = []
    for i, gene in enumerate(genes):
        theta = _exact.draws(post, i, settings["draws"], np.random.default_rng(seeds[i]))
        draws = {
            "intercept": theta[:, 0],
            "beta": theta[:, 1 : K + 1],
            "categorical_beta": theta[:, K + 1],
            "beta_1": theta[:, K + 2 :],
        }
        fit_columns = {"sigma_mean": float(np.sqrt(post.sigma2_mean[i])), "exact_edge_mass": float(post.edge_mass[i])}
        results.append(_summarise_gene(gene, draws, shared, fit_columns, moments=(post.mean[i], post.cov[i])))
    return results


def _exact_profile_results(shared, Y, genes, ss_within, settings):
    """Exact inference for the condition-free model: ``[(gene, diag_row, curve_row), ...]``."""
    design = np.column_stack([np.ones(shared.B.shape[0]), shared.B])
    post = _exact_fit(design, Y, genes, shared.weights, ss_within, _resolve_priors(shared.priors),
                      ("intercept", "beta"), shared.n_bsplines, settings)  # fmt: skip
    A = np.column_stack([np.ones(shared.B_pred.shape[0]), shared.B_pred])
    seeds = np.random.SeedSequence(settings["random_seed"]).spawn(len(genes))
    results = []
    for i, gene in enumerate(genes):
        curves = _exact.draws(post, i, settings["draws"], np.random.default_rng(seeds[i])) @ A.T
        ci_low, ci_high = np.quantile(curves, [0.03, 0.97], axis=0)
        diag_row = {
            "gene": gene,
            "sigma_mean": float(np.sqrt(post.sigma2_mean[i])),
            "exact_edge_mass": float(post.edge_mass[i]),
        }
        curve_row = {"mean": A @ post.mean[i], "ci_low": ci_low, "ci_high": ci_high, "beta": post.mean[i][1:],
                     "beta_cov": post.cov[i][1:, 1:]}  # fmt: skip
        results.append((gene, diag_row, curve_row))
    return results


def _fit_and_predict_profile(gene, Y, intercept_initval, shared, sample_kwargs, ss_within=None):
    """`_fit_and_predict_gene` for the condition-free model: one curve, no test. Returns ``(gene, diag_row, curve_row)``."""
    model = _build_profile_model(
        Y,
        shared.B,
        shared.n_bsplines,
        intercept_initval,
        shared.likelihood,
        shared.priors,
        weights=shared.weights,
        ss_within=ss_within,
    )
    with model, _quiet_pymc():
        idata, diag_columns = _sample_nuts(model, sample_kwargs, ["intercept", "beta", "sigma"])

    # Posterior of the mean function on the grid: credible bands, no observation noise (see `_mean_curve_draws`).
    draws = _posterior_draws(idata, "intercept")[:, None] + _posterior_draws(idata, "beta") @ shared.B_pred.T
    mean = draws.mean(axis=0)
    ci_low, ci_high = np.quantile(draws, [0.03, 0.97], axis=0)
    beta_mean = idata.posterior["beta"].mean(("chain", "draw")).to_numpy()

    diag_row = {"gene": gene, **diag_columns}
    beta_cov = np.cov(_posterior_draws(idata, "beta"), rowvar=False)
    curve_row = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "beta": beta_mean, "beta_cov": beta_cov}
    return gene, diag_row, curve_row


def _gamma_null_pvalues(T_values, null_quantile=0.9):
    """Gamma-null p-values for `T_values`, the null fitted to the lower `null_quantile` of them.

    Assumes that at least a `null_quantile` fraction of the values come from the null. The Gamma is fitted
    by maximum likelihood to the values below the cutoff *as a truncated sample* (each term divided by the
    Gamma's mass below the cutoff); fitting it as if untruncated gives too light a tail and p-values that
    are too small. Values that are missing or not finite get a missing p-value.
    """
    T_values = np.asarray(T_values, dtype=float)
    finite = np.isfinite(T_values)
    pvals = np.full(T_values.shape, np.nan)
    if finite.sum() < 3:
        return pvals
    cutoff = np.quantile(T_values[finite], null_quantile)
    pool = T_values[finite & (T_values <= cutoff) & (T_values > 0)]
    if len(pool) < 3:
        return pvals
    shape0, _, scale0 = gamma.fit(pool, floc=0)

    def neg_log_lik(log_params):
        a, scale = np.exp(log_params)
        return -(gamma.logpdf(pool, a, scale=scale).sum() - len(pool) * gamma.logcdf(cutoff, a, scale=scale))

    fit = optimize.minimize(neg_log_lik, np.log([shape0, scale0]), method="Nelder-Mead")
    a, scale = np.exp(fit.x)
    pvals[finite] = gamma.sf(T_values[finite], a, scale=scale)
    return pvals


def _benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values over the non-missing entries; missing ones stay missing."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan)
    present = np.isfinite(pvals)
    p = pvals[present]
    n = len(p)
    if n == 0:
        return out
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    filled = np.empty(n)
    filled[order] = adjusted
    out[present] = filled
    return out


# Condition tests, a registry like the priors/likelihoods. Each has a per-gene
# part, run in the worker, which returns the test's own columns plus the mean
# difference curve on the grid, and a panel part for what needs every gene.


def _wald_gene_stats(draws0, draws1, idata, shared, moments=None):
    """Wald test that the two condition curves are equal (level or shape), plus their `T` distance.

    `draws0`/`draws1` are posterior draws of the mean curves, so the covariance is that of the
    coefficients behind ``d(x) = f1(x) - f0(x)`` (rank ``n_bsplines``: the level plus the shape directions).
    """
    x = np.asarray(shared.bin_centers, dtype=float)
    inside = (x >= shared.support[0] - 1e-9) & (x <= shared.support[1] + 1e-9)
    if moments is None:
        delta = draws1 - draws0
        delta_mean = delta.mean(axis=0)
        cov = np.cov(delta[:, inside], rowvar=False)
    else:
        A0, A1 = _condition_maps(shared)
        A = A1 - A0
        delta_mean, cov = A @ moments[0], A[inside] @ moments[1] @ A[inside].T
    wald_stat, wald_pval, wald_df = _wald_from_moments(delta_mean[inside], cov, df_denominator=shared.df_denominator)
    T = float(np.mean(delta_mean**2))
    row = {"wald_stat": wald_stat, "wald_pval": wald_pval, "wald_df": wald_df, "T": T}
    if shared.df_denominator is not None:
        row["wald_df_denominator"] = shared.df_denominator
    return row, delta_mean


def _anchor_support(binned, coverage=0.96):
    """The anchor range the condition tests are computed over: where the cells are.

    The central `coverage` of the cells (2nd to 98th percentile by default), cut to the grid points where both
    conditions have cells. Outside it a curve is extrapolated from the spline and the prior, so a level, slope
    or shape averaged over all of ``[0, 1]`` would be dominated by the prior there; with slides that makes the
    test far too confident. Returns ``(lo, hi)``, or ``(0, 1)`` when the cell counts per grid point are unknown
    (overlapping bins, fits on cells).
    """
    counts = getattr(binned, "n_obs_grid", None)
    if counts is None:
        return 0.0, 1.0
    counts = np.asarray(counts, dtype=float)
    x = np.asarray(binned.bin_centers, dtype=float)
    both = (counts > 0).all(axis=0) if counts.shape[0] > 1 else counts[0] > 0
    total = counts.sum(axis=0) * both
    if total.sum() == 0:
        return 0.0, 1.0
    cdf = np.cumsum(total) / total.sum()
    tail = (1 - coverage) / 2
    lo, hi = x[np.searchsorted(cdf, tail)], x[min(np.searchsorted(cdf, 1 - tail), len(x) - 1)]
    return (float(lo), float(hi)) if hi > lo else (0.0, 1.0)


def _level_weights(bin_centers, support=(0.0, 1.0)):
    """Trapezoid weights that turn a curve on the grid into its mean over `support`."""
    x = np.asarray(bin_centers, dtype=float)
    inside = (x >= support[0] - 1e-9) & (x <= support[1] + 1e-9)
    xi = x[inside]
    w = np.zeros(len(x))
    wi = np.zeros(len(xi))
    dx = np.diff(xi)
    wi[:-1] += dx / 2
    wi[1:] += dx / 2
    w[inside] = wi / (xi[-1] - xi[0])
    return w


def _level_gene_stats(draws0, draws1, idata, shared, moments=None):
    """Test of the level difference ``mean_x (f1 - f0)``, ``H0: |level| <= tau``, plus its credible interval."""
    a = _level_weights(shared.bin_centers, shared.support)
    level_draws = (draws1 - draws0) @ a
    if moments is None:
        delta_mean = (draws1 - draws0).mean(axis=0)
        level, var = float(level_draws.mean()), float(level_draws.var(ddof=1))
    else:
        A0, A1 = _condition_maps(shared)
        A = A1 - A0
        delta_mean = A @ moments[0]
        level, var = float(a @ delta_mean), float(a @ A @ moments[1] @ A.T @ a)
    stat = level**2 / var
    pval = _one_df_pval(stat, var, shared.tau, shared.df_denominator)
    low, high = (float(v) for v in np.quantile(level_draws, [0.03, 0.97]))
    row = {
        "level_diff": level,
        "level_diff_ci_low": low,
        "level_diff_ci_high": high,
        "level_stat": stat,
        "level_pval": pval,
    }
    if shared.df_denominator is not None:
        row["level_df_denominator"] = shared.df_denominator
    return row, delta_mean


def _two_stage_stats(shared, Y, genes, test, tau, ridge=1e-3):
    """Two-stage test for data with slides: a curve per slide, then a comparison of the slides' summaries.

    Stage 1 fits each slide's curve ``a_s + B(x) b_s`` by weighted least squares on its own rows (a small
    ridge keeps it defined where a slide covers few grid points). Stage 2 compares a summary of these curves
    between the conditions with every slide counted once: the level (mean over ``[0, 1]``) or the slope
    (least-squares slope) by a two-sample t-test, the shape (the spline coefficients) by Hotelling's T^2.
    This is the summary-statistics approach to repeated measures: it makes no assumption on how slides vary,
    which the joint model's random curve (one scale for every spline direction) does.
    """
    n_s = shared.n_samples
    X = np.column_stack([np.ones(len(shared.B)), shared.B])
    w = np.ones(len(X)) if shared.weights is None else np.asarray(shared.weights, dtype=float)
    Ymat = np.column_stack([np.asarray(Y[g], dtype=float) for g in genes])
    coefs, cond = [], np.empty(n_s)
    for s in range(n_s):
        rows = np.asarray(shared.sample_index) == s
        Xs, ws = X[rows], w[rows]
        coefs.append(
            np.linalg.solve(Xs.T @ (ws[:, None] * Xs) + ridge * np.eye(X.shape[1]), Xs.T @ (ws[:, None] * Ymat[rows]))
        )
        cond[s] = np.asarray(shared.condition_fit)[rows][0]
    coefs = np.stack(coefs)  # (slides, 1 + K, genes)
    curves = np.einsum("pk,skg->spg", np.column_stack([np.ones(len(shared.B_pred)), shared.B_pred]), coefs)
    g1, g0 = cond == 1, cond == 0
    n1, n0 = int(g1.sum()), int(g0.sum())
    df = n1 + n0 - 2
    rows = {}
    if test in ("level", "slope"):
        a = _level_weights(shared.bin_centers, shared.support)
        if test == "level":
            func = a
        else:
            xc = shared.bin_centers - a @ shared.bin_centers
            func = a * xc / (a @ xc**2)
        summ = np.einsum("p,spg->sg", func, curves)  # (slides, genes)
        diff = summ[g1].mean(0) - summ[g0].mean(0)
        pooled = (((summ[g1] - summ[g1].mean(0)) ** 2).sum(0) + ((summ[g0] - summ[g0].mean(0)) ** 2).sum(0)) / df
        var = pooled * (1 / n1 + 1 / n0)
        half = student_t.ppf(0.97, df) * np.sqrt(var)
        for i, g in enumerate(genes):
            stat = diff[i] ** 2 / var[i] if var[i] > 0 else np.inf
            rows[g] = {
                f"{test}_diff": diff[i], f"{test}_diff_ci_low": diff[i] - half[i], f"{test}_diff_ci_high": diff[i] + half[i],
                f"{test}_stat": stat, f"{test}_pval": _one_df_pval(stat, var[i], tau, df) if var[i] > 0 else 0.0,
                f"{test}_df_denominator": df,
            }  # fmt: skip
    elif test == "parallel":
        x = np.asarray(shared.bin_centers, dtype=float)
        inside = (x >= shared.support[0] - 1e-9) & (x <= shared.support[1] + 1e-9)
        cols = np.abs(np.asarray(shared.B_pred)[inside]).max(axis=0) > 1e-8  # basis functions seen on the support
        coefs = np.concatenate([coefs[:, :1], coefs[:, 1:][:, cols]], axis=1)
        K = coefs.shape[1] - 1
        if df - K + 1 < 1:
            raise ValueError(f"The two-stage shape test needs more than {K + 1} slides, found {n_s}; use test='slope'.")
        for i, g in enumerate(genes):
            b = coefs[:, 1:, i]
            d = b[g1].mean(0) - b[g0].mean(0)
            S = (
                (b[g1] - b[g1].mean(0)).T @ (b[g1] - b[g1].mean(0))
                + (b[g0] - b[g0].mean(0)).T @ (b[g0] - b[g0].mean(0))
            ) / df
            T2 = float(d @ np.linalg.pinv(S) @ d) * n1 * n0 / (n1 + n0)
            F = T2 * (df - K + 1) / (df * K)
            rows[g] = {"parallel_stat": T2, "parallel_pval": float(f.sf(F, K, df - K + 1)), "parallel_df": K,
                       "parallel_df_denominator": df - K + 1}  # fmt: skip
    else:
        return None
    return pd.DataFrame.from_dict(rows, orient="index")


def _level_panel_stats(stats_df, null_quantile):
    stats_df["level_pval_adj"] = _benjamini_hochberg(stats_df["level_pval"].to_numpy())


def _one_df_pval(stat, var, tau, df_denominator):
    """p-value of ``stat = estimate^2 / var`` for ``H0: |effect| <= tau`` (1 df; F with slides)."""
    noncentrality = tau**2 / var
    if df_denominator is None:
        return float(ncx2.sf(stat, 1, noncentrality)) if tau > 0 else float(chi2.sf(stat, 1))
    if tau > 0:
        return float(ncf.sf(stat, 1, df_denominator, noncentrality))
    return float(f.sf(stat, 1, df_denominator))


def _slope_gene_stats(draws0, draws1, idata, shared, moments=None):
    """Test of the slope difference: the least-squares slope of ``f1 - f0`` over ``[0, 1]``, ``H0: |slope| <= tau``.

    One degree of freedom, so with few replicates it has far more power than the ``K - 1`` df parallel test
    for a gradient that steepens or flattens.
    """
    weights = shared.slope_weights
    beta_1 = _posterior_draws(idata, "beta_1")
    slope_draws = beta_1 @ weights
    if moments is None:
        slope, var = float(slope_draws.mean()), float(slope_draws.var(ddof=1))
        delta_mean = (draws1 - draws0).mean(axis=0)
    else:
        b1 = slice(shared.n_bsplines + 2, 2 * shared.n_bsplines + 2)
        slope, var = float(moments[0][b1] @ weights), float(weights @ moments[1][b1, b1] @ weights)
        A0, A1 = _condition_maps(shared)
        delta_mean = (A1 - A0) @ moments[0]
    stat = slope**2 / var
    low, high = (float(v) for v in np.quantile(slope_draws, [0.03, 0.97]))
    row = {
        "slope_diff": slope,
        "slope_diff_ci_low": low,
        "slope_diff_ci_high": high,
        "slope_stat": stat,
        "slope_pval": _one_df_pval(stat, var, shared.tau, shared.df_denominator),
    }
    if shared.df_denominator is not None:
        row["slope_df_denominator"] = shared.df_denominator
    return row, delta_mean


def _slope_panel_stats(stats_df, null_quantile):
    stats_df["slope_pval_adj"] = _benjamini_hochberg(stats_df["slope_pval"].to_numpy())


def _wald_panel_stats(stats_df, null_quantile):
    # Per-gene p-values; only the FDR adjustment needs the panel. (The panel nulls on `T` that used to be
    # here were circular: they always flagged the top `1 - null_quantile` of the genes.)
    stats_df["wald_pval_adj"] = _benjamini_hochberg(stats_df["wald_pval"].to_numpy())


def _parallel_gene_stats(draws0, draws1, idata, shared, moments=None):
    """`parallel_curves_test` on the posterior of `beta_1`, plus the linear slope of the difference curve."""
    beta_1 = _posterior_draws(idata, "beta_1")  # (n_draws, K)
    if moments is None:
        shape_diff, stat, df, pval = parallel_curves_test(
            beta_1, shared.shape_metric, tau=shared.tau, df_denominator=shared.df_denominator
        )
        delta_mean = (draws1 - draws0).mean(axis=0)
    else:
        K = shared.n_bsplines
        b1 = slice(K + 2, 2 * K + 2)
        shape_diff, stat, df, pval = parallel_curves_test(
            moments[0][b1], shared.shape_metric, tau=shared.tau, cov=moments[1][b1, b1],
            df_denominator=shared.df_denominator,
        )  # fmt: skip
        A0, A1 = _condition_maps(shared)
        delta_mean = (A1 - A0) @ moments[0]
    slope = beta_1 @ shared.slope_weights
    slope_low, slope_high = (float(v) for v in np.quantile(slope, [0.03, 0.97]))
    row = {
        "shape_diff": shape_diff,
        "slope_diff": float(slope.mean()),
        "slope_diff_ci_low": slope_low,
        "slope_diff_ci_high": slope_high,
        "parallel_stat": stat,
        "parallel_df": df,
        "parallel_pval": pval,
    }
    if moments is not None:
        row["slope_diff"] = float(moments[0][slice(shared.n_bsplines + 2, None)] @ shared.slope_weights)
    if shared.df_denominator is not None:
        row["parallel_df_denominator"] = shared.df_denominator
    return row, delta_mean


def _parallel_panel_stats(stats_df, null_quantile):
    # The p-values are per gene; only the FDR adjustment needs the panel.
    stats_df["parallel_pval_adj"] = _benjamini_hochberg(stats_df["parallel_pval"].to_numpy())


#: ``{name: (per-gene stats, panel stats)}`` - the condition tests `fit_model` can run.
_TESTS = {
    "wald": (_wald_gene_stats, _wald_panel_stats),
    "parallel": (_parallel_gene_stats, _parallel_panel_stats),
    "level": (_level_gene_stats, _level_panel_stats),
    "slope": (_slope_gene_stats, _slope_panel_stats),
}


_CURVE_SUFFIXES = ("cond0_mean", "cond0_ci_low", "cond0_ci_high", "cond1_mean", "cond1_ci_low", "cond1_ci_high")
_COEF_NAMES = ("beta", "beta_1")


def _coef_columns(n_bsplines):
    """Column labels of a coefficient table, one per spline basis function (strings, so they round-trip through h5ad)."""
    return [f"B{k}" for k in range(n_bsplines)]


def _resolve_inference(inference, model_setup):
    """``"auto"`` picks exact inference where the model allows it (see `_exact.supported`), NUTS otherwise."""
    if inference == "advi":
        warnings.warn(
            "inference='advi' was removed: the exact posterior replaces it (same model, no approximation, much "
            "faster). Running inference='auto' instead; pass that to silence this warning.",
            FutureWarning,
            stacklevel=3,
        )
        inference = "auto"
    allowed = ("auto", "exact", *_INFERENCE)
    if inference not in allowed:
        raise ValueError(f"Unknown inference {inference!r}. Available: {sorted(allowed)}")
    if model_setup is None:
        return "nuts" if inference == "auto" else inference
    ok = _exact.supported(
        _resolve_priors(model_setup.priors), model_setup.likelihood, model_setup.binned.sample is not None
    )
    if inference == "exact" and not ok:
        raise ValueError(
            "inference='exact' needs the normal likelihood and normal priors on the coefficients; use 'nuts'."
        )
    resolved = ("exact" if ok else "nuts") if inference == "auto" else inference
    if resolved != "exact":
        _require_pymc(resolved)
    return resolved


def _warn_predictive_seed(predictive_seed):
    if predictive_seed is not None:
        warnings.warn(
            "`predictive_seed` no longer has an effect: the curves are now the posterior of the mean function, "
            "computed from the posterior draws without new random draws.",
            DeprecationWarning,
            stacklevel=3,
        )


def _fittable_genes(genes, Y):
    """The genes worth fitting.

    Genes whose fitted rows are constant (e.g. all zero) are left out with a warning: the model reproduces
    them exactly, so the noise scale has no lower bound and the posterior is improper (the sampler diverges
    and the test would be meaningless). Their stats stay missing.
    """
    degenerate = [g for g in genes if not np.isfinite(Y[g]).all() or np.ptp(np.asarray(Y[g], dtype=float)) == 0]
    if degenerate:
        shown = ", ".join(degenerate[:10]) + (" ..." if len(degenerate) > 10 else "")
        warnings.warn(
            f"{len(degenerate)} gene(s) are constant or non-finite over the fitted rows and were not fitted "
            f"(their stats are missing): {shown}",
            UserWarning,
            stacklevel=3,
        )
    return [g for g in genes if g not in set(degenerate)]


def _check_slides_per_condition(binned, sample_index, condition_fit):
    """The F reference uses ``n_slides - 2`` degrees of freedom, which assumes at least two slides per condition."""
    codes = np.asarray(condition_fit, dtype=int)
    per_condition = [len(np.unique(np.asarray(sample_index)[codes == c])) for c in (0, 1)]
    if min(per_condition) < 2:
        labels = binned.condition_categories
        raise ValueError(
            "Comparing conditions across slides needs at least 2 slides per condition, found "
            f"{per_condition[0]} for {labels['0']!r} and {per_condition[1]} for {labels['1']!r}."
        )


def _warn_unsupported_range(binned):
    """Warn when one condition has no cells over part of ``[0, 1]``: the curve there is extrapolated."""
    empty = (np.asarray(binned.n_obs_grid) == 0).mean(axis=1)
    if empty.max() > 0.1:
        labels = binned.condition_categories
        detail = ", ".join(f"{labels[str(c)]!r}: {100 * empty[c]:.0f}%" for c in (0, 1))
        warnings.warn(
            f"Part of the anchor range has no cells in one condition ({detail} of the grid empty). The curves "
            "there are extrapolated from the spline and the prior, which inflates `shape_diff`/`integral`.",
            UserWarning,
            stacklevel=3,
        )


def fit_model(
    adata,
    model_setup,
    n_workers=1,
    progressbar=True,
    inplace=True,
    use_binned=True,
    stats_key="dsvg_stats",
    preds_key="dsvg_preds",
    curves_key="dsvg_curves",
    null_quantile=0.9,
    sample_kwargs=None,
    test="parallel",
    tau=None,
    predictive_seed=None,
    inference="auto",
    coefs_key="dsvg_coefs",
    two_stage=True,
    **pymc_kwargs,
):
    """Fit every per-gene model from `setup_model`, in parallel, and summarize the result.

    For each gene: (re-)builds its own private model and obtains the
    posterior (see `inference`), then samples the posterior predictive on the
    fixed `anchor` grid under both levels of `condition`, and tests the two
    curves against each other (see `test`).

    Parameters
    ----------
    adata
        The same `AnnData` object `model_setup.binned` was built from; only
        used as the write target when `inplace` is `True`.
    model_setup
        Output of `setup_model`.
    n_workers
        Number of worker processes (`joblib`, `loky` backend). Each gene's
        model lives entirely within one worker for its whole lifetime -
        genes are never split across workers or re-synced, so there is
        nothing to race. `n_workers=1` runs sequentially in-process (no
        extra pickling at all).
    use_binned
        Fit on the per-bin averages from `diffsvg.pp.bin_obs` (default), or
        on the raw per-cell values when `False`. Raw keeps every
        observation instead of collapsing each window to its mean, at the
        cost of a far larger design matrix - with spatial data that is
        mostly zero counts, which is what the binning is there to avoid,
        this is typically much slower per gene. Knots, degree and the
        condition encoding are identical either way, so the two are
        directly comparable. Bins built with ``window_radius=None`` are
        weighted by their cell counts, which makes the binned fit carry the
        same information as the per-cell one at the cost of the binned fit.
    progressbar
        Show a `tqdm` progress bar over completed genes. Per-worker MCMC
        progress bars are always disabled (they would interleave garbage
        across processes); pass e.g. ``sample_kwargs={"progressbar": True}``
        to override that only when `n_workers=1`.
    inplace
        If `True` (default), write results to `adata.varm`/`adata.uns`
        (aligned to ``adata.var_names``, with genes not in `model_setup`
        left as `NaN`) and return `None`. If `False`, leave `adata` untouched
        and return ``(stats_df, preds_df, curve_dfs)``; `curve_dfs` then also
        holds the coefficient tables under ``"beta"`` and ``"beta_1"``.
    stats_key
        `adata.varm` key for the per-gene summary statistics (`n_genes x n`;
        see `diffsvg.pl.plot_volcano`/`diffsvg.pl.plot_pvalue_histogram`).
        Its columns depend on `test`; which test ran (and its `tau`) is
        recorded in ``adata.uns[f"{stats_key}_params"]``.
    preds_key
        `adata.varm` key for the per-gene, per-bin predicted
        ``condition==1`` minus ``condition==0`` difference curve
        (`n_genes x (n_bins + 1)`).
    curves_key
        Prefix for six `adata.varm` entries (`n_genes x (n_bins + 1)` each),
        one per ``{"cond0", "cond1"} x {"mean", "ci_low", "ci_high"}``: the
        per-condition predicted curves with a credible band, consumed by
        `diffsvg.pl.plot_prediction_curve`. The condition labels are also
        written to ``adata.uns[f"{curves_key}_condition_categories"]``.
    null_quantile
        No longer used by `fit_model` (kept for backwards compatibility).
        The panel-based empirical/Gamma nulls it configured were circular;
        ``test="wald"`` now has a per-gene p-value.
    sample_kwargs
        Extra keyword arguments merged into the `pymc.sample` call (applied
        to every gene). Any extra keyword arguments to `fit_model` itself are
        forwarded the same way, so ``fit_model(..., chains=2)`` also works.
        With ``inference="exact"`` only ``draws``, ``random_seed``,
        ``n_grid`` and ``n_hyper`` apply.
    test
        Which test to run between the two condition curves:

        - ``"parallel"`` (default): whether the curves are parallel, i.e. differ only
          in level, via `parallel_curves_test` on the posterior of the
          interaction coefficients. The p-value is per gene, so it does not
          depend on the panel and works for a single gene. Columns
          ``shape_diff`` (how far from parallel), ``slope_diff`` (the linear
          trend of ``f1 - f0`` over the anchor range, with a 94% credible
          interval ``slope_diff_ci_low``/``slope_diff_ci_high``),
          ``parallel_stat``, ``parallel_df``, ``parallel_pval`` and
          ``parallel_pval_adj`` (Benjamini-Hochberg).
        - ``"wald"``: whether the curves are equal at all (level or shape),
          a tradeSeq-style functional Wald test on the posterior of the
          difference curve, chi-square with ``n_bsplines`` degrees of
          freedom (columns ``wald_stat``, ``wald_pval``, ``wald_df``,
          ``wald_pval_adj`` and ``T``, the mean squared difference). It has
          no `tau`, so with many cells nearly every gene with any difference
          becomes significant; rank by ``wald_stat`` rather than thresholding.
        - ``"level"``: whether the curves differ in level, i.e. in the mean
          of ``f1 - f0`` over the anchor range, beyond `tau` (``H0: |level|
          <= tau``; noncentral chi-square with 1 degree of freedom). The
          complement of ``"parallel"``, which ignores the level. Columns
          ``level_diff`` (with a 94% credible interval), ``level_stat``,
          ``level_pval`` and ``level_pval_adj``.
        - ``"slope"``: whether the gap between the curves changes linearly
          along the anchor, i.e. whether one gradient is steeper than the
          other: the least-squares slope of ``f1 - f0`` over ``[0, 1]``,
          ``H0: |slope| <= tau``, 1 degree of freedom. With few replicates it
          has far more power than ``"parallel"`` (``n_bsplines - 1`` df) for
          gradients that steepen or flatten, and it misses non-monotone
          changes, which ``"parallel"`` covers. Columns ``slope_diff`` (with
          a 94% credible interval), ``slope_stat``, ``slope_pval`` and
          ``slope_pval_adj``.

        With data binned with `sample` (several slides per condition) the model has a random level and a
        random curve per slide, so the covariance of the interaction
        coefficients holds the variation between slides, and the p-value
        uses an F reference with ``n_slides - 2`` denominator degrees of
        freedom for every test (column ``*_df_denominator``). The fitted curves are
        the population-level ones. Fit on bins (NUTS is fine) or on cells
        (``use_binned=False``).

        Every test is computed over the anchor range where the cells are:
        the central 96% of them, cut to where both conditions have cells
        (``anchor_support`` in the stored test parameters). Outside it the
        curves are extrapolated from the spline and the prior, and averaging
        over that part would make the test overconfident, most of all for
        cell types that occupy only part of the anchor range. Slopes and
        shape differences stay in units of the full ``[0, 1]`` range.

        Both tests also write ``integral`` (the area between the curves),
        the ``categorical_beta`` summary and the sampler diagnostics.
        Binned fits with overlapping windows (`diffsvg.pp.bin_obs`'s default
        `window_radius`) treat neighbouring bins as independent although they
        share cells, which makes the model over-confident and the
        ``"parallel"`` p-values too small (a warning says so); contiguous
        bins (``bin_obs(..., window_radius=None)``) and fits on the cells
        (``use_binned=False``) do not have that problem.
    tau
        For the ``"parallel"``, ``"level"`` and ``"slope"`` tests: the
        smallest difference that counts (``shape_diff``, ``level_diff`` or
        ``slope_diff``). The p-value tests ``|difference| <= tau``, so smaller
        differences do not become significant however many cells there are.
        Defaults to ``0.1`` for pooled data and to ``0`` for data binned with
        `sample`, where the variation between slides already calibrates the
        test and a positive `tau` only costs power.
    two_stage
        With data binned with `sample`: compute the test from one curve per
        slide, comparing the slides' summaries between the conditions (level
        or slope: t-test; shape: Hotelling's T^2; ``n_slides - 2``
        denominator degrees of freedom). Every slide counts once and nothing
        is assumed about how slides differ, so the error rate holds when the
        joint model's random curve is misspecified. The fitted curves and
        bands still come from the joint model. ``False`` uses the joint
        model's posterior for the test as well.
    predictive_seed
        Deprecated, has no effect. The curves, their bands, ``integral`` and
        the Wald test are now computed from the posterior draws of the mean
        function (no observation noise), so they need no extra random draws.
    inference
        How each gene's posterior is obtained:

        - ``"nuts"``: MCMC with `pymc.sample` (needs the ``pymc`` extra), configured by
          `sample_kwargs`. Reports ``n_divergent``, ``rhat_max`` and
          ``ess_bulk_min``.
        - ``"exact"``: the exact posterior (see `diffsvg.tl._exact`): the
          coefficients are conjugate given the noise scale, which is
          integrated out numerically, as are the two slide scales. No Monte
          Carlo error and no sampler; needs the normal likelihood and normal
          coefficient priors. Reports ``exact_edge_mass``.
        - ``"auto"`` (default): ``"exact"`` where the model allows it,
          ``"nuts"`` otherwise.
    coefs_key
        Prefix for two `adata.varm` tables (`n_genes x n_bsplines`) with the
        posterior means of the spline coefficients, consumed by
        `diffsvg.tl.cluster_curves`: ``f"{coefs_key}_beta"`` (the main spline
        term; with the 0/1 condition coding this is the curve of condition 0,
        not a pooled one) and ``f"{coefs_key}_beta_1"`` (the interaction, how
        the curve of condition 1 differs from it).

    Returns
    -------
    None or tuple[pandas.DataFrame, pandas.DataFrame, dict[str, pandas.DataFrame]]
    """
    if test not in _TESTS:
        raise ValueError(f"Unknown test {test!r}. Available: {sorted(_TESTS)}")
    if tau is not None and tau < 0:
        raise ValueError(f"`tau` must be >= 0, got {tau}.")
    inference = _resolve_inference(inference, model_setup)
    if inference == "exact":
        sample_defaults = _exact_settings({**(sample_kwargs or {}), **pymc_kwargs})
    elif inference == "nuts":
        sample_defaults = {
            "nuts_sampler": "nutpie",
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "random_seed": 42,
            "compute_convergence_checks": False,
            "init": "adapt_diag",
            "progressbar": False,
        }
    if inference != "exact":
        sample_defaults.update(sample_kwargs or {})
        sample_defaults.update(pymc_kwargs)
    binned = model_setup.binned
    if not binned.has_condition:
        raise ValueError(
            "adata.uns['binned'] was built without a `condition`, so there are no two curves to compare. "
            "Re-run `diffsvg.pp.bin_obs` with one, or use `diffsvg.tl.fit_profiles` + "
            "`diffsvg.tl.one_vs_rest` for the condition-free one-vs-rest test."
        )
    B, B_masked, condition_fit, Y, intercept_initvals = _fit_inputs(adata, binned, use_binned)
    weights, ss_within, n_obs_grid = _weighting(binned, use_binned)
    sample_index, n_samples = _sample_inputs(adata, binned, use_binned)
    if tau is None:
        tau = 0.0 if n_samples is not None else 0.1
    if n_samples is not None:
        if n_samples < 3:
            raise ValueError(f"Comparing conditions across slides needs at least 3 slides, found {n_samples}.")
    if test == "parallel" and use_binned and not binned.weighted:
        warnings.warn(
            "test='parallel' on overlapping bins: neighbouring bins share cells, so the p-values come out too "
            "small. Re-run `diffsvg.pp.bin_obs(..., window_radius=None)`, or fit with `use_binned=False`.",
            UserWarning,
            stacklevel=2,
        )
    _warn_predictive_seed(predictive_seed)
    if n_samples is not None:
        _check_slides_per_condition(binned, sample_index, condition_fit)
    if use_binned and binned.condition is not None and binned.n_obs_grid is not None:
        _warn_unsupported_range(binned)
    fit_genes = _fittable_genes(binned.genes, Y)
    support = _anchor_support(binned) if use_binned else (0.0, 1.0)
    shared = _FitShared(
        B=B,
        B_masked=B_masked,
        condition_fit=condition_fit,
        n_bsplines=binned.n_bsplines,
        likelihood=model_setup.likelihood,
        priors=model_setup.priors,
        bin_centers=binned.bin_centers,
        B_pred=binned.B_pred,
        B_pred_masked_0=binned.B_pred_masked_0,
        B_pred_masked_1=binned.B_pred_masked_1,
        condition_pred_0=binned.condition_pred_0,
        condition_pred_1=binned.condition_pred_1,
        test=test,
        tau=float(tau),
        shape_metric=shape_metric(binned.n_bsplines, binned.degree, support=support) if test == "parallel" else None,
        slope_weights=(
            _slope_weights(binned.n_bsplines, binned.degree, support=support) if test in ("parallel", "slope") else None
        ),
        support=support,
        weights=weights,
        inference=inference,
        sample_index=sample_index,
        n_samples=n_samples,
        df_denominator=None if n_samples is None else n_samples - 2,
    )

    n_jobs = 1 if n_workers is None else n_workers
    if inference == "exact":
        results = _exact_gene_results(shared, Y, fit_genes, ss_within, sample_defaults)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_fit_and_predict_gene)(
                gene,
                Y[gene],
                intercept_initvals[gene],
                shared,
                sample_defaults,
                ss_within=None if ss_within is None else ss_within[gene],
            )
            for gene in fit_genes
        )
    if progressbar:
        results = tqdm(results, total=len(fit_genes), desc="Fitting DiffSVG models")

    stats_rows = []
    curve_rows = {}
    for gene, stats_row, curve_row in results:
        stats_rows.append(stats_row)
        curve_rows[gene] = curve_row

    stats_df = pd.DataFrame(stats_rows).set_index("gene").reindex(binned.genes)
    if two_stage and n_samples is not None:
        two = _two_stage_stats(shared, Y, list(fit_genes), test, tau)
        if two is not None:
            stats_df.loc[two.index, two.columns] = two
    _TESTS[test][1](stats_df, null_quantile)
    test_params = {"test": test, "tau": float(tau)} if test in ("parallel", "level", "slope") else {"test": test}
    test_params["two_stage"] = bool(two_stage and n_samples is not None)
    test_params["anchor_support"] = [float(v) for v in support]

    # Column labels must be strings (not e.g. plain floats) so that
    # `adata.varm[...]` round-trips through h5ad, whose DataFrame writer
    # requires string column names.
    bin_cols = [str(round(float(c), 4)) for c in binned.bin_centers]
    preds_df = pd.DataFrame.from_dict(
        {gene: row["delta"] for gene, row in curve_rows.items()}, orient="index", columns=bin_cols
    ).reindex(binned.genes)
    curve_dfs = {
        suffix: pd.DataFrame.from_dict(
            {gene: row[suffix] for gene, row in curve_rows.items()}, orient="index", columns=bin_cols
        ).reindex(binned.genes)
        for suffix in _CURVE_SUFFIXES
    }
    coef_dfs = {
        name: pd.DataFrame.from_dict(
            {gene: row[name] for gene, row in curve_rows.items()},
            orient="index",
            columns=_coef_columns(binned.n_bsplines),
        ).reindex(binned.genes)
        for name in _COEF_NAMES
    }

    if not inplace:
        stats_df.attrs.update(test_params)
        return stats_df, preds_df, {**curve_dfs, **coef_dfs}

    adata.varm[stats_key] = stats_df.reindex(adata.var_names)
    adata.varm[preds_key] = preds_df.reindex(adata.var_names)
    for suffix, df in curve_dfs.items():
        adata.varm[f"{curves_key}_{suffix}"] = df.reindex(adata.var_names)
    for name, df in coef_dfs.items():
        adata.varm[f"{coefs_key}_{name}"] = df.reindex(adata.var_names)
    adata.uns[f"{curves_key}_condition_categories"] = binned.condition_categories
    adata.uns[f"{stats_key}_params"] = test_params
    return None


# ============================================================
# 3b) Condition-free profiles + the one-vs-rest test
# ============================================================


_PROFILE_SUFFIXES = ("mean", "ci_low", "ci_high")


def fit_profiles(
    adata,
    model_setup,
    n_workers=1,
    progressbar=True,
    inplace=True,
    use_binned=True,
    profiles_key="dsvg_profiles",
    sample_kwargs=None,
    predictive_seed=None,
    inference="auto",
    **pymc_kwargs,
):
    """Fit the condition-free model for every gene, giving one expression profile per gene.

    The counterpart to `fit_model` for data binned without a `condition`
    (`diffsvg.pp.bin_obs(..., condition=None)`): it drops the condition
    covariate and its spline interaction, so each gene gets a single curve
    rather than one per condition. No differential statistic is computed
    here - `one_vs_rest` produces that by comparing genes against each
    other.

    Also usable on data that *does* have conditions, in which case both
    conditions' bins feed one pooled curve per gene. That is an alternative
    to what `one_vs_rest` does by default there (averaging the two fitted
    condition curves), and the two need not agree exactly.

    Parameters
    ----------
    adata
        The `AnnData` `model_setup.binned` was built from; the write target
        when `inplace` is `True`.
    model_setup
        Output of `setup_model`.
    n_workers, progressbar, use_binned, sample_kwargs, predictive_seed, **pymc_kwargs
        As in `fit_model`.
    inplace
        If `True` (default), write the results to `adata.varm` and return
        `None`; otherwise return ``(profile_dfs, diagnostics_df)``.
    profiles_key
        Prefix for three `adata.varm` tables (`n_genes x (n_bins + 1)`), one
        per ``{"mean", "ci_low", "ci_high"}``, plus a
        ``f"{profiles_key}_diagnostics"`` table of per-gene sampler
        diagnostics and ``f"{profiles_key}_beta"`` (`n_genes x n_bsplines`,
        the posterior means of the spline coefficients, for
        `diffsvg.tl.cluster_curves` with ``by="profile"``). With
        ``inplace=False`` that table is ``profile_dfs["beta"]``.

    Returns
    -------
    None or tuple[dict[str, pandas.DataFrame], pandas.DataFrame]
    """
    if inference not in ("auto", "exact", "nuts"):
        raise ValueError(f"Unknown inference {inference!r} for fit_profiles. Available: ['auto', 'exact', 'nuts']")
    priors = _resolve_priors(model_setup.priors)
    exact_ok = (
        model_setup.likelihood == "normal"
        and model_setup.binned.sample is None
        and all(priors[n][0] == "normal" for n in ("intercept", "beta"))
        and priors["sigma"][0] in _exact._SIGMA_LOGPDF
    )
    if inference == "exact" and not exact_ok:
        raise ValueError("inference='exact' needs the normal likelihood and normal priors on the coefficients.")
    inference = ("exact" if exact_ok else "nuts") if inference == "auto" else inference
    if inference != "exact":
        _require_pymc(inference)
    sample_defaults = {
        "nuts_sampler": "nutpie",
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "random_seed": 42,
        "compute_convergence_checks": False,
        "init": "adapt_diag",
        "progressbar": False,
    }
    if inference == "exact":
        sample_defaults = _exact_settings({**(sample_kwargs or {}), **pymc_kwargs})
    else:
        sample_defaults.update(sample_kwargs or {})
        sample_defaults.update(pymc_kwargs)

    binned = model_setup.binned
    if binned.sample is not None:
        raise ValueError("`fit_profiles` does not model slides; bin without `sample`.")
    B, _, _, Y, intercept_initvals = _fit_inputs(adata, binned, use_binned)
    weights, ss_within, n_obs_grid = _weighting(binned, use_binned)
    _warn_predictive_seed(predictive_seed)
    fit_genes = _fittable_genes(binned.genes, Y)
    shared = _ProfileShared(
        B=B,
        n_bsplines=binned.n_bsplines,
        likelihood=model_setup.likelihood,
        priors=model_setup.priors,
        B_pred=binned.B_pred,
        weights=weights,
        # One pooled curve: with conditions binned separately, a grid point holds both levels' cells.
    )

    n_jobs = 1 if n_workers is None else n_workers
    if inference == "exact":
        results = _exact_profile_results(shared, Y, fit_genes, ss_within, sample_defaults)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_fit_and_predict_profile)(
                gene,
                Y[gene],
                intercept_initvals[gene],
                shared,
                sample_defaults,
                ss_within=None if ss_within is None else ss_within[gene],
            )
            for gene in fit_genes
        )
    if progressbar:
        results = tqdm(results, total=len(binned.genes), desc="Fitting DiffSVG profiles")

    diag_rows = []
    curve_rows = {}
    for gene, diag_row, curve_row in results:
        diag_rows.append(diag_row)
        curve_rows[gene] = curve_row

    diagnostics_df = pd.DataFrame(diag_rows).set_index("gene").reindex(binned.genes)
    bin_cols = [str(round(float(c), 4)) for c in binned.bin_centers]
    profile_dfs = {
        suffix: pd.DataFrame.from_dict(
            {gene: row[suffix] for gene, row in curve_rows.items()}, orient="index", columns=bin_cols
        ).reindex(binned.genes)
        for suffix in _PROFILE_SUFFIXES
    }
    profile_dfs["beta"] = pd.DataFrame.from_dict(
        {gene: row["beta"] for gene, row in curve_rows.items()},
        orient="index",
        columns=_coef_columns(binned.n_bsplines),
    ).reindex(binned.genes)

    if not inplace:
        return profile_dfs, diagnostics_df

    for suffix, df in profile_dfs.items():
        adata.varm[f"{profiles_key}_{suffix}"] = df.reindex(adata.var_names)
    adata.varm[f"{profiles_key}_diagnostics"] = diagnostics_df.reindex(adata.var_names)
    fitted = [g for g in binned.genes if g in curve_rows]
    adata.uns[f"{profiles_key}_beta_cov"] = {
        "genes": np.asarray(fitted, dtype=str),
        "cov": np.stack([curve_rows[g]["beta_cov"] for g in fitted]),
        "degree": int(binned.degree),
    }
    return None


def _gene_profiles(adata, curves_key, profiles_key):
    """One curve per gene, as a (genes x bins) frame, plus a note on where it came from.

    Prefers an explicit `fit_profiles` result; otherwise averages the two
    per-condition curves from `fit_model`, which is what collapses a
    two-condition fit into a single profile per gene.
    """
    if f"{profiles_key}_mean" in adata.varm:
        return adata.varm[f"{profiles_key}_mean"], "profiles"
    if f"{curves_key}_cond0_mean" in adata.varm:
        cond0 = adata.varm[f"{curves_key}_cond0_mean"]
        cond1 = adata.varm[f"{curves_key}_cond1_mean"]
        return (cond0 + cond1) / 2.0, "condition-average"
    raise KeyError(
        f"neither adata.varm[{profiles_key + '_mean'!r}] nor adata.varm[{curves_key + '_cond0_mean'!r}] found - "
        "run `diffsvg.tl.fit_profiles` (no conditions) or `diffsvg.tl.fit_model` (with conditions) first."
    )


def one_vs_rest(
    adata,
    gene,
    curves_key="dsvg_curves",
    profiles_key="dsvg_profiles",
    null_quantile=0.9,
    inplace=False,
    key_added=None,
    tau=0.1,
):
    """Compare one gene's expression profile against every other gene's.

    Each gene is reduced to a single curve along `anchor` (see
    `_gene_profiles`), and the probe gene is then tested against each of the
    remaining genes individually.

    With a `fit_profiles` result (which stores each gene's posterior
    covariance of its spline coefficients) every pair gets the parallel-curves
    test of `parallel_curves_test`: do the two genes' curves differ in shape,
    beyond `tau`, rather than only in level? The covariance of the difference
    is the sum of the two genes' posterior covariances (their fits are
    independent). Without it (curves averaged from `fit_model`), the pairs
    are ranked by ``T`` against a Gamma null fitted to the "rest" genes.

    Parameters
    ----------
    adata
        Annotated data matrix that `fit_profiles` or `fit_model` wrote into.
    gene
        The probe gene, i.e. the "one" in one-vs-rest.
    curves_key, profiles_key
        `adata.varm` prefixes written by `fit_model` / `fit_profiles`. An
        explicit `fit_profiles` result wins when both are present.
    null_quantile
        Only without a `fit_profiles` result: fraction of the "rest" genes
        assumed to follow the null; the Gamma null is fitted to the lower
        `null_quantile` of their ``T`` values, as a truncated sample (see
        `_gamma_null_pvalues`).
    inplace
        If `True`, also write the table to ``adata.varm[key_added]``
        (reindexed to ``adata.var_names``, so the probe gene's own row is
        `NaN`). The table is returned either way.
    key_added
        `adata.varm` key used when `inplace`; defaults to
        ``f"dsvg_ovr_{gene}"``.
    tau
        Smallest shape difference between two genes' curves that counts, as in
        `parallel_curves_test`.

    Returns
    -------
    pandas.DataFrame
        Indexed by the "rest" genes, with ``delta`` (the signed area between
        the probe curve and that gene's, negative when the probe sits below
        it) and ``T``; with a `fit_profiles` result ``shape_diff``,
        ``parallel_stat``, ``parallel_pval`` and ``parallel_pval_adj``,
        otherwise the Gamma-null ``gamma_pval`` and ``gamma_pval_adj``.
    """
    profiles, source = _gene_profiles(adata, curves_key, profiles_key)
    if gene not in profiles.index:
        raise KeyError(f"{gene!r} not found in adata.var_names")

    profiles = profiles.dropna(how="all")
    if gene not in profiles.index:
        raise KeyError(f"{gene!r} has no fitted curve - it was not part of the fitted gene set.")
    rest = profiles.drop(index=gene)
    if rest.empty:
        raise ValueError("need at least two genes with fitted curves for a one-vs-rest comparison.")

    bin_centers = profiles.columns.to_numpy(dtype=float)
    # probe minus rest, so a downregulated probe gives a negative delta.
    delta = profiles.loc[gene].to_numpy()[None, :] - rest.to_numpy()

    result = pd.DataFrame(
        {
            "delta": np.trapezoid(delta, x=bin_centers, axis=1),
            "T": (delta**2).mean(axis=1),
        },
        index=rest.index,
    )
    cov_key = f"{profiles_key}_beta_cov"
    if source == "profiles" and cov_key in adata.uns:
        stored = adata.uns[cov_key]
        position = {str(g): i for i, g in enumerate(stored["genes"])}
        covs = np.asarray(stored["cov"])
        betas = adata.varm[f"{profiles_key}_beta"]
        metric = shape_metric(covs.shape[1], int(stored["degree"]))
        rows = []
        for other in rest.index:
            d_beta = betas.loc[gene].to_numpy(dtype=float) - betas.loc[other].to_numpy(dtype=float)
            cov = covs[position[gene]] + covs[position[other]]
            rows.append(parallel_curves_test(d_beta, metric, tau=tau, cov=cov))
        shape_diff, stat, _, pval = (np.asarray(v) for v in zip(*rows, strict=True))
        result["shape_diff"], result["parallel_stat"], result["parallel_pval"] = shape_diff, stat, pval
        result["parallel_pval_adj"] = _benjamini_hochberg(pval)
    else:
        result["gamma_pval"] = _gamma_null_pvalues(result["T"].to_numpy(), null_quantile=null_quantile)
        result["gamma_pval_adj"] = _benjamini_hochberg(result["gamma_pval"].to_numpy())
    result.attrs["probe_gene"] = gene
    result.attrs["profile_source"] = source

    if inplace:
        adata.varm[key_added or f"dsvg_ovr_{gene}"] = result.reindex(adata.var_names)
    return result


# ============================================================
# 4) Statistics helpers
# ============================================================


def pattern_wald_test(pred0, pred1, ridge=0.0):
    """tradeSeq-like functional Wald test that two curves are equal, from posterior draws of the curves.

    Pass draws of the *mean* curves (as `fit_model` does): posterior-predictive draws add observation
    noise to the covariance, which makes the test depend on how the data were binned.

    Parameters
    ----------
    pred0, pred1 : array, shape (n_draws, M)
        Posterior draws of the two condition curves.
    ridge : float
        Diagonal regularization of the covariance (relative to its largest eigenvalue).

    Returns
    -------
    stat : float
        Wald statistic.
    pval : float
        Chi-square p-value.
    df : int
        Effective degrees of freedom (rank of the covariance).
    delta_mean : array
        Mean difference curve ``pred1 - pred0``.
    """
    delta = pred1 - pred0  # (draws, M)
    mu = delta.mean(axis=0)

    W, pval, df = _wald_from_moments(mu, np.cov(delta, rowvar=False), ridge=ridge)
    return W, pval, df, mu


def _wald_from_moments(mu, Sigma, ridge=0.0, df_denominator=None):
    """Wald statistic of ``mu`` against covariance `Sigma`, pseudoinverse over its non-null directions.

    The cutoff is relative, so the degrees of freedom are the rank of the difference curve, not the
    number of grid points. With `df_denominator` (slides) the p-value is from ``F(df, df_denominator)``
    on ``stat / df``, as in `parallel_curves_test`. Returns ``(stat, pval, df)``.
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if ridge:
        eigvals = eigvals + ridge * eigvals.max()
    keep = eigvals > eigvals.max() * 1e-10
    z = eigvecs[:, keep].T @ mu
    W = float((z**2 / eigvals[keep]).sum())
    df = int(keep.sum())
    pval = float(chi2.sf(W, df)) if df_denominator is None else float(f.sf(W / df, df, df_denominator))
    return W, pval, df


def _shape_grid(n_bsplines, degree, n_grid, support=(0.0, 1.0)):
    """Fine grid on `support`, its trapezoid weights (summing to 1) and the uncentred basis on it."""
    x = np.linspace(support[0], support[1], n_grid)
    w = np.full(n_grid, 1.0 / (n_grid - 1))
    w[[0, -1]] /= 2
    return x, w, _bspline_basis_01(x, n_bsplines, degree)


def shape_metric(n_bsplines=10, degree=3, n_grid=2001, support=(0.0, 1.0)):
    """Quadratic form `Q` that measures how far two condition curves are from parallel.

    For interaction coefficients `b` (``beta_1`` of the per-gene model),
    ``sqrt(b @ Q @ b)`` is the shape difference of the difference curve
    ``d(x) = f1(x) - f0(x)`` on ``[0, 1]``, in slope-equivalent units:
    ``sqrt(12) * RMS(d - mean(d))``. When the gap between the curves changes
    linearly, ``d(x) = a + s * x``, this is exactly ``|s|``, the slope
    difference; for any other shape it is the linear change with the same RMS
    departure from parallel. Level differences contribute nothing
    (``Q @ ones = 0``). Centring the basis does not change `Q`, so it depends
    only on the knots, i.e. on `n_bsplines` and `degree` of
    `diffsvg.pp.bin_obs`.

    Parameters
    ----------
    n_bsplines, degree
        As in `diffsvg.pp.bin_obs`.
    n_grid
        Grid points for the numerical integral.
    support
        The part of ``[0, 1]`` the shape is measured over (see `fit_model`).
        Slopes stay in units of the full anchor range.

    Returns
    -------
    numpy.ndarray
        `Q`, shape ``(n_bsplines, n_bsplines)``, positive semi-definite with
        rank ``n_bsplines - 1``.
    """
    _, w, B = _shape_grid(n_bsplines, degree, n_grid, support)
    centred = B - w @ B
    return 12.0 / (support[1] - support[0]) ** 2 * (centred * w[:, None]).T @ centred


def _slope_weights(n_bsplines, degree, n_grid=2001, support=(0.0, 1.0)):
    """`g` with ``g @ b`` = the least-squares slope over `support` of the difference curve for coefficients `b`."""
    x, w, B = _shape_grid(n_bsplines, degree, n_grid, support)
    xc = x - w @ x
    return ((xc * w) @ B) / (w @ xc**2)


def parallel_curves_test(beta_1, metric, tau=0.1, cov=None, df_denominator=None):
    """Test whether two condition curves are parallel, up to a shape difference `tau`.

    The curves are parallel when they differ only in level, i.e. when their
    difference ``d(x) = f1(x) - f0(x)`` is constant. How far they are from
    that is ``shape_diff = sqrt(beta_1 @ metric @ beta_1)`` (see
    `shape_metric`; for a gap that changes linearly it is the slope
    difference), and the test is ``H0: shape_diff <= tau`` against
    ``H1: shape_diff > tau``.

    The statistic is the Wald statistic of the ``K - 1`` shape directions of
    `beta_1` (the constant direction is the level and is left out). With
    ``tau = 0`` it is the classic test of parallel curves, chi-square with
    ``K - 1`` degrees of freedom. With ``tau > 0`` it is compared against the
    largest distribution it can have when the true shape difference is
    `tau`: noncentral chi-square with noncentrality ``tau**2`` over the
    variance of the best-measured shape direction. Shape differences below
    `tau` therefore do not become significant however precisely they are
    measured, while the p-value stays valid for every difference up to `tau`.

    With `df_denominator` (replicates: the posterior covariance then holds
    the variation between slides, which was estimated from only a few) the
    statistic over its ``K - 1`` degrees of freedom is compared against an F
    distribution instead (noncentral for ``tau > 0``, with the same
    noncentrality), which is wider than the chi-square and so keeps the test
    honest when there are few slides.

    Parameters
    ----------
    beta_1 : array, shape (n_draws, K) or (K,)
        Posterior draws of the interaction coefficients, or their posterior
        mean (then `cov` is required).
    metric : array, shape (K, K)
        `shape_metric` for the basis `beta_1` belongs to.
    tau : float
        Smallest shape difference that counts, in units of ``shape_diff``.
    cov : array, shape (K, K), optional
        Posterior covariance of `beta_1`, when `beta_1` is a mean.
    df_denominator : int, optional
        Denominator degrees of freedom of the F reference, e.g. the number
        of slides minus 2 when comparing two groups of slides.

    Returns
    -------
    shape_diff : float
        Shape difference of the posterior-mean curves. Noise in the fitted
        curves adds to it, so poorly measured genes show a sizeable
        ``shape_diff`` even when the true curves are parallel; the p-value
        accounts for that, the raw number does not.
    stat : float
        Wald statistic.
    df : int
        Degrees of freedom, ``K - 1``.
    pval : float
    """
    beta_1 = np.asarray(beta_1, dtype=float)
    if beta_1.ndim == 2:
        mean, cov = beta_1.mean(axis=0), np.cov(beta_1, rowvar=False)
    elif cov is None:
        raise ValueError("pass `cov` along with a posterior mean, or pass the posterior draws.")
    else:
        mean, cov = beta_1, np.asarray(cov, dtype=float)
    if tau < 0:
        raise ValueError(f"`tau` must be >= 0, got {tau}.")

    # Q = L L' over its K - 1 non-zero directions, so z = L' beta_1 has |z| = shape_diff.
    evals, evecs = np.linalg.eigh(np.asarray(metric, dtype=float))
    keep = evals > evals.max() * 1e-10
    L = evecs[:, keep] * np.sqrt(evals[keep])
    z = L.T @ mean
    omega = L.T @ cov @ L
    omega_min = np.linalg.eigvalsh(omega)[0]
    if omega_min <= 0:
        raise ValueError("the covariance of `beta_1` is singular in the shape directions.")

    stat = float(z @ np.linalg.solve(omega, z))
    df = int(keep.sum())
    noncentrality = tau**2 / omega_min
    if df_denominator is None:
        pval = float(ncx2.sf(stat, df, noncentrality)) if tau > 0 else float(chi2.sf(stat, df))
    elif df_denominator < 1:
        raise ValueError(f"`df_denominator` must be >= 1, got {df_denominator}.")
    elif tau > 0:
        pval = float(ncf.sf(stat / df, df, df_denominator, noncentrality))
    else:
        pval = float(f.sf(stat / df, df, df_denominator))
    return float(np.sqrt(z @ z)), stat, df, pval


def compute_T(curve0, curve1):
    """Functional L2 distance between two curves."""
    return np.mean((curve1 - curve0) ** 2)
