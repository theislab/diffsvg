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
from scipy.special import logsumexp
from scipy.stats import chi2, f, gamma, ncf, ncx2
from tqdm.auto import tqdm

from diffsvg.pp.utils import BinnedData, _bspline_basis_01, _expression, spline_basis_at
from diffsvg.tl import _exact

try:  # the `pymc` extra: only needed for inference="nuts"/"advi"
    import arviz as az
    import pymc as pm
    import pytensor
    from pymc.blocking import DictToArrayBijection, RaveledVars
except ImportError:
    az = pm = pytensor = DictToArrayBijection = RaveledVars = None


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
    - `record_mu` keeps the row-wise mean as the ``"mu"`` deterministic (one value per row and draw, so
      ADVI leaves it out).
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
# Posterior inference: NUTS, or full-rank ADVI
# ------------------------------------------------------------
# `fit_model(..., inference=...)` picks one of `_INFERENCE`. Each runs inside ``with model`` and returns the
# posterior as `InferenceData` plus a dict of diagnostic columns.
#
# ADVI is full-rank and starts at the Laplace approximation (mode and inverse Hessian on the unconstrained scale):
# the spline coefficients have a long, thin, tilted posterior that plain `pm.fit` from a unit Gaussian does not
# reach on single cells, and mean-field cannot represent it. The shape test is built on that covariance.
# See .claude/advi_plan.md.

#: Settings of ``inference="advi"``, overridable through `fit_model`'s `sample_kwargs`.
#:
#: - ``n``: Adam steps after the Laplace start (0 returns the Laplace approximation itself).
#: - ``learning_rate``, ``final_learning_rate``: decayed exponentially over ``n``; small, since the start is close.
#: - ``obj_n_mc``: Monte Carlo samples per gradient.
#: - ``draws``: draws from the fitted approximation.
#: - ``init``: ``"laplace"``, or ``None`` for PyMC's unit-Gaussian start (not recommended).
#: - ``diagnostic_draws``: draws for the importance-ratio diagnostics (0 skips them).
#: - ``random_seed``.
#: - ``slide_scales``: only with `sample`: ``"empirical_bayes"`` (see `_empirical_bayes_scales`) or
#:   ``"data_start"`` (scales stay random, started at `_guess_slide_scales`).
_VI_DEFAULTS = {
    "n": 2000,
    "learning_rate": 1e-3,
    "final_learning_rate": 1e-5,
    "obj_n_mc": 1,
    "draws": 4000,
    "init": "laplace",
    "diagnostic_draws": 1000,
    "random_seed": 42,
    "slide_scales": "empirical_bayes",
}

#: Overrides of `_VI_DEFAULTS` for data binned with `sample`. With ~100-200 parameters the default polish adds more
#: noise than it removes (simulated slides: the importance ESS share falls to 0.05 at 1e-3 and stays at 0.45 at 1e-4,
#: with the same Wald statistic as NUTS either way), so the steps are ten times smaller.
_VI_SLIDE_DEFAULTS = {"learning_rate": 1e-4, "final_learning_rate": 1e-6}
#: Smallest acceptable ``vi_ess`` without / with slides. The share falls with the number of parameters even for a good
#: approximation (simulated slides, ~130 parameters: 0.45 while matching NUTS), so with slides only a collapse counts.
_VI_MIN_ESS = {False: 0.5, True: 0.1}


def _check_vi_settings(settings):
    """Fail on a mistyped setting (or one that belongs to NUTS) before any gene is fitted."""
    unknown = sorted(set(settings) - set(_VI_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown setting(s) for inference='advi': {unknown}. Available: {sorted(_VI_DEFAULTS)}")
    if settings["init"] not in ("laplace", None):
        raise ValueError(f"`init` must be 'laplace' or None, got {settings['init']!r}.")
    if settings["n"] < 0 or settings["draws"] < 2:
        raise ValueError("`n` must be >= 0 and `draws` >= 2.")
    if settings["learning_rate"] <= 0 or settings["final_learning_rate"] <= 0:
        raise ValueError("the learning rates must be positive.")
    if settings["slide_scales"] not in _SLIDE_SCALE_METHODS:
        raise ValueError(f"`slide_scales` must be one of {_SLIDE_SCALE_METHODS}, got {settings['slide_scales']!r}.")


class _FlatDensity:
    """A model's log density and its gradient on the unconstrained scale, as functions of one flat vector.

    The vector has the order of ADVI's parameters (the value variables of the
    model, ravelled one after the other), so a point found here can serve as
    the mean of the approximation as it is. The density is the one ADVI is
    fitted to: it includes the Jacobians of the transforms (``sigma`` is
    sampled as ``log(sigma)``).
    """

    def __init__(self, model):
        self.start = DictToArrayBijection.map(model.initial_point())
        self._logp = model.compile_logp(jacobian=True)
        self._dlogp = model.compile_dlogp(jacobian=True)

    def _point(self, x):
        return DictToArrayBijection.rmap(RaveledVars(x, self.start.point_map_info))

    @property
    def names(self):
        return [name for name, *_ in self.start.point_map_info]

    def logp(self, x):
        return float(self._logp(self._point(x)))

    def dlogp(self, x):
        return np.asarray(self._dlogp(self._point(x)), dtype=float)


def _inverse_psd(precision, floor=1e-10):
    """Inverse of a symmetric matrix, with eigenvalues below `floor` times the largest lifted to it."""
    values, vectors = np.linalg.eigh(precision)
    if values.max() <= 0:
        raise ValueError("the Hessian of the log density has no positive direction at its mode.")
    values = np.maximum(values, floor * values.max())
    return (vectors / values) @ vectors.T


def _mode(density, start=None, free=None, newton_steps=3, step=1e-6):
    """Mode of a density over the `free` coordinates (the others held at `start`), and the precision there.

    L-BFGS finds the mode roughly, a few Newton steps refine it - the
    posterior is thin, so L-BFGS alone stops early along the narrow
    directions - and the Hessian comes from central differences of the
    analytic gradient, which needs no second-order graph to compile.
    Returns ``(x, precision)``: the full flat vector and minus the Hessian
    of the log density over the free coordinates.
    """
    x = np.array(density.start.data if start is None else start, dtype=float)
    free = np.arange(len(x)) if free is None else np.asarray(free)

    def at(values):
        point = x.copy()
        point[free] = values
        return point

    def precision(values):
        # minus the Hessian of the log density, one column per free parameter
        out = np.empty((len(free), len(free)))
        for a, j in enumerate(free):
            h = np.zeros(len(x))
            h[j] = step * max(1.0, abs(values[a]))
            out[:, a] = (density.dlogp(at(values) - h)[free] - density.dlogp(at(values) + h)[free]) / (2 * h[j])
        return (out + out.T) / 2

    result = optimize.minimize(
        lambda v: (-density.logp(at(v)), -density.dlogp(at(v))[free]),
        x[free],
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8},
    )
    values = result.x
    for _ in range(newton_steps):
        proposal = values + np.linalg.solve(precision(values), density.dlogp(at(values))[free])
        if not density.logp(at(proposal)) >= density.logp(at(values)):
            break
        values = proposal
    return at(values), precision(values)


def _laplace(density, start=None, free=None):
    """Mode and covariance of a density (its Laplace approximation) on the unconstrained scale; see `_mode`."""
    x, precision = _mode(density, start, free)
    return x, _inverse_psd(precision)


def _tril_params(cov):
    """`FullRank`'s ``L_tril`` for a covariance: its Cholesky factor, lower triangle ravelled, diagonal before the softplus."""
    chol = np.linalg.cholesky(cov)
    diag = np.diag(chol)
    chol[np.diag_indices_from(chol)] = np.where(diag > 30.0, diag, np.log(np.expm1(np.minimum(diag, 30.0))))
    return chol[np.tril_indices(len(chol))]


def _gpd_shape(x):
    """Shape of a generalized Pareto distribution fitted to sorted, positive exceedances `x`.

    The empirical-Bayes fit of Zhang and Stephens (2009), with the weakly
    informative prior on the shape (towards 0.5) of Vehtari et al. (2024).
    """
    n = len(x)
    quartile = x[int(n / 4 + 0.5) - 1]
    if quartile <= x[0]:  # far from any Pareto tail: a few draws carry all the weight
        return np.inf
    m = 30 + int(np.sqrt(n))
    b = (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3 * quartile) + 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    profile = n * (np.log(-b / k) - k - 1)
    weight = np.exp(profile - logsumexp(profile))
    keep = weight >= 10 * np.finfo(float).eps
    b_mean = (b[keep] * weight[keep]).sum() / weight[keep].sum()
    shape = np.log1p(-b_mean * x).mean()
    return (n * shape + 10 * 0.5) / (n + 10)


def _pareto_k(weights):
    """Pareto k-hat of importance weights: the tail index of their largest values (Vehtari et al., 2024).

    Importance sampling has a finite variance when it is below 0.5 and is
    still usable up to about 0.7.
    """
    w = np.sort(weights)
    n_tail = int(min(0.2 * len(w), 3 * np.sqrt(len(w))))
    if n_tail < 5:
        return np.nan
    tail = w[-n_tail:] - w[-n_tail - 1]
    if tail[-1] <= tail[0]:  # all tail weights equal, i.e. the approximation is exact
        return np.nan
    return float(_gpd_shape(tail))


def _importance_diagnostics(density, mean, cov, n_draws, seed):
    """How far a Gaussian approximation is from the density it was fitted to, from importance ratios (Yao et al., 2018).

    Draws from the approximation and weighs each draw by density over
    approximation. Returns ``(k, ess)``: the Pareto k-hat of the weights - an
    approximation that is too narrow somewhere has heavy-tailed ratios, and
    above about 0.7 it should not be trusted - and their effective sample size
    as a share of the draws, which a covariance that is too wide or an
    off-centre mean lose as well (below about 0.5 the covariance is more than
    ~25% off, and with it the Wald statistic built on it).
    """
    chol = np.linalg.cholesky(cov)
    z = np.random.default_rng(seed).standard_normal((n_draws, len(mean)))
    # The Gaussian's normalizing constant is left out: it shifts all log-weights alike, and neither diagnostic sees it.
    log_q = -0.5 * (z**2).sum(axis=1) - np.log(np.diag(chol)).sum()
    log_w = np.array([density.logp(mean + chol @ zi) for zi in z]) - log_q
    w = np.exp(log_w - log_w.max())
    return _pareto_k(w), float(w.sum() ** 2 / (w**2).sum() / n_draws)


# ------------------------------------------------------------
# Slides with ADVI
# ------------------------------------------------------------
# With `bin_obs(sample=...)` the slide effects are non-centred (``scale * z``), and the joint mode of
# (z, log scale), where a plain Laplace start would sit, is far from the posterior (simulated slides: scales
# 1.3 / 2.3 vs 0.40 / 0.55, W 0.15 vs NUTS's 4.6). The ``slide_scales`` setting picks the way around it:
# - "empirical_bayes": fix the scales at the mode of their marginal posterior, slide effects integrated out by a
#   Laplace approximation (as lme4/TMB do), then the usual Laplace start + ADVI.
# - "data_start": keep the scales random and start ADVI at a least-squares guess of them.

#: The slide scales (value variables) and the standard-normal slide effects each one multiplies.
_SLIDE_SCALES = {"sample_level_sd_log__": "sample_level_z", "sample_curve_sd_log__": "sample_curve_z"}
_SLIDE_SCALE_METHODS = ("empirical_bayes", "data_start")
#: Standard deviation of the log-scales in the starting approximation of ``slide_scales="data_start"``.
_DATA_START_LOG_SCALE_SD = 0.3


def _blocks(density):
    """Index range of every named variable in the flat vector of a `_FlatDensity`."""
    out, start = {}, 0
    for name, _, size, _ in density.start.point_map_info:
        out[name] = np.arange(start, start + size)
        start += size
    return out


def _guess_slide_scales(y, B, condition, sample_index, n_samples, weights=None, ridge=1e-3):
    """Rough slide scales by least squares, as a starting point (assumes the Normal likelihood).

    Fits the pooled model, then each slide's residuals with a level and a spline curve, and returns the
    RMS of those levels and of those curve coefficients (biased upwards by noise, fine for a start).
    """
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)

    def solve(X, target, w_rows):
        return np.linalg.solve(
            (X * w_rows[:, None]).T @ X + ridge * np.eye(X.shape[1]), (X * w_rows[:, None]).T @ target
        )

    X = np.column_stack([np.ones(len(y)), B, condition, B * condition[:, None]])
    residual = y - X @ solve(X, y, w)
    levels, curves = [], []
    for j in range(n_samples):
        rows = np.asarray(sample_index) == j
        coef = solve(np.column_stack([np.ones(rows.sum()), B[rows]]), residual[rows], w[rows])
        levels.append(coef[0])
        curves.append(coef[1:])
    return (
        max(float(np.sqrt(np.mean(np.square(levels)))), 1e-2),
        max(float(np.sqrt(np.mean(np.square(curves)))), 1e-2),
    )


def _expansion_optimum(log_scales, expansion):
    """The slide effects maximizing the expanded log density at `log_scales`, and their prior precision there."""
    e = expansion
    prior_precision = np.zeros(len(e["u_hat"]))
    for idx, h in zip(e["effects"], log_scales, strict=True):
        prior_precision[idx] = np.exp(-2 * h)
    u = e["u_hat"] + np.linalg.solve(e["A"] + np.diag(prior_precision), e["b"] - prior_precision * e["u_hat"])
    return u, prior_precision


def _neg_log_marginal(log_scales, expansion, density):
    """Minus the Laplace-approximated log marginal posterior of the log-scales, from one expansion (up to a constant)."""
    e = expansion
    u, prior_precision = _expansion_optimum(log_scales, e)
    try:
        chol = np.linalg.cholesky(e["A"] + np.diag(prior_precision))
    except np.linalg.LinAlgError:
        return np.inf
    delta = u - e["u_hat"]
    value = e["b"] @ delta - 0.5 * delta @ e["A"] @ delta - 0.5 * (prior_precision * u * u).sum()
    value -= np.dot([len(idx) for idx in e["effects"]], log_scales)  # normalizers of N(u; 0, scale^2)
    # The scales' own prior and Jacobian, read off the density with every effect at 0, where the likelihood ignores them.
    point = e["no_effects"].copy()
    point[e["hyper"]] = log_scales
    return -(value - np.log(np.diag(chol)).sum() + density.logp(point))


def _empirical_bayes_scales(density, start_scales, rounds=5, tol=0.01, bounds=(1e-3, 10.0)):
    """Slide scales at the mode of their marginal posterior, the slide effects integrated out by a Laplace approximation.

    Each round finds the mode of everything else at the current scales and
    the precision there, rewrites that expansion in terms of the slide
    effects themselves (``u = scale * z``), where the scales enter only
    through the Gaussian prior of ``u``, and maximizes the resulting marginal
    over the log-scales (priors and Jacobians included); every evaluation is
    one Cholesky factorization. The next round re-expands at the new scales.
    For a Normal likelihood the expansion is exact up to sigma, so it settles
    in two or three rounds. Returns ``(level_sd, curve_sd)``.
    """
    blocks = _blocks(density)
    hyper = np.concatenate([blocks[name] for name in _SLIDE_SCALES])
    free = np.setdiff1d(np.arange(len(density.start.data)), hyper)
    position = np.full(len(density.start.data), -1)
    position[free] = np.arange(len(free))
    effects = [position[blocks[z]] for z in _SLIDE_SCALES.values()]  # per scale, its effects among the free coordinates
    is_effect = np.zeros(len(free), dtype=bool)
    for idx in effects:
        is_effect[idx] = True

    def to_z(log_scales):
        """Per free coordinate the factor taking u to z (1/scale on the slide effects, 1 elsewhere)."""
        out = np.ones(len(free))
        for idx, h in zip(effects, log_scales, strict=True):
            out[idx] = np.exp(-h)
        return out

    x = np.array(density.start.data, dtype=float)
    x[hyper] = np.log(start_scales)
    for _ in range(rounds):
        x, precision = _mode(density, x, free)
        log_scales = x[hyper].copy()
        # The expansion in u = scale * z without z's N(0, 1) prior: b'(u - u_hat) - (u - u_hat)'A(u - u_hat)/2.
        t = to_z(log_scales)
        no_effects = x.copy()
        no_effects[free[is_effect]] = 0.0
        expansion = {
            "b": t * (density.dlogp(x)[free] + np.where(is_effect, x[free], 0.0)),
            "A": t[:, None] * (precision - np.diag(is_effect.astype(float))) * t[None, :],
            "u_hat": x[free] / t,
            "effects": effects,
            "hyper": hyper,
            "no_effects": no_effects,
        }
        result = optimize.minimize(
            _neg_log_marginal,
            log_scales,
            args=(expansion, density),
            method="L-BFGS-B",
            bounds=[tuple(np.log(bounds))] * len(hyper),
        )
        u, _ = _expansion_optimum(result.x, expansion)
        x[free] = u * to_z(result.x)
        x[hyper] = result.x
        if np.max(np.abs(result.x - log_scales)) < tol:
            break
    return tuple(float(v) for v in np.exp(x[hyper]))


def _fit_advi_slides(model, settings, rebuild, guess):
    """Full-rank ADVI for a model with slides, the two slide scales handled as ``settings["slide_scales"]`` says.

    `rebuild(scales)` builds the same model with the scales fixed, `guess` is
    `_guess_slide_scales`. Called outside any model context, as it builds a
    second model for empirical Bayes. Returns ``(model the draws belong to, idata,
    columns)``; the columns include the scales (estimates or posterior means).
    """
    density = _FlatDensity(model)
    if settings["slide_scales"] == "empirical_bayes":
        scales = _empirical_bayes_scales(density, guess)
        fixed = rebuild(scales)
        with fixed:
            idata, columns = _fit_advi(fixed, settings, None)
        return fixed, idata, {**columns, "sample_level_sd": scales[0], "sample_curve_sd": scales[1]}

    blocks = _blocks(density)
    hyper = np.concatenate([blocks[name] for name in _SLIDE_SCALES])
    free = np.setdiff1d(np.arange(len(density.start.data)), hyper)
    start = np.array(density.start.data, dtype=float)
    start[hyper] = np.log(guess)
    x, precision = _mode(density, start, free)
    cov = np.zeros((len(x), len(x)))
    cov[np.ix_(free, free)] = _inverse_psd(precision)
    cov[hyper, hyper] = _DATA_START_LOG_SCALE_SD**2
    with model:
        idata, columns = _fit_advi(model, settings, None, start=(x, cov))
    posterior = idata.posterior
    return (
        model,
        idata,
        {
            **columns,
            "sample_level_sd": float(posterior["sample_level_sd"].mean()),
            "sample_curve_sd": float(posterior["sample_curve_sd"].mean()),
        },
    )


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


def _fit_advi(model, settings, diag_vars, start=None):
    """Fit the posterior with full-rank ADVI started at its Laplace approximation (see `_VI_DEFAULTS`).

    `start` ``(mean, covariance)`` on the flat unconstrained scale replaces the Laplace start.
    """
    density = _FlatDensity(model)
    seed, n = settings["random_seed"], settings["n"]
    inference = pm.FullRankADVI(model=model, random_seed=seed)
    group = inference.approx.groups[0]
    if density.names != list(group.ordering):
        raise RuntimeError(f"ADVI orders its parameters {list(group.ordering)}, not {density.names}.")
    if start is None and settings["init"] == "laplace":
        start = _laplace(density)
    if start is not None:
        mode, cov = start
        group.shared_params["mu"].set_value(mode)
        group.shared_params["L_tril"].set_value(_tril_params(cov))

    first, last = settings["learning_rate"], settings["final_learning_rate"]
    learning_rate = pytensor.shared(np.float64(first), "learning_rate")

    def anneal(approx, loss, step):
        learning_rate.set_value(np.float64(first * (last / first) ** (step / n)))

    approx = inference.fit(
        n=n,
        progressbar=False,
        obj_n_mc=settings["obj_n_mc"],
        obj_optimizer=pm.adam(learning_rate=learning_rate),
        callbacks=[anneal],
    )
    idata = approx.sample(settings["draws"], random_seed=seed)

    k, ess = np.nan, np.nan
    if settings["diagnostic_draws"] > 0:
        k, ess = _importance_diagnostics(
            density, approx.mean.eval(), approx.cov.eval(), settings["diagnostic_draws"], seed
        )
    return idata, {"vi_khat": k, "vi_ess": ess}


#: ``{name: function}`` - how `fit_model` can obtain the posterior of a gene; see `fit_model`'s `inference`.
_INFERENCE = {
    "nuts": _sample_nuts,
    "advi": _fit_advi,
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
            record_mu=shared.inference == "nuts",
            sample_index=shared.sample_index,
            n_samples=shared.n_samples,
            sample_scales=sample_scales,
        )

    model = build()
    slides = shared.sample_index is not None

    with _quiet_pymc():
        if slides and shared.inference == "advi":  # builds its own models, so outside any model context
            guess = _guess_slide_scales(
                Y, shared.B, shared.condition_fit, shared.sample_index, shared.n_samples, shared.weights
            )
            model, idata, fit_columns = _fit_advi_slides(model, sample_kwargs, build, guess)
        else:
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
        draws = {"intercept": theta[:, 0], "beta": theta[:, 1 : K + 1], "categorical_beta": theta[:, K + 1], "beta_1": theta[:, K + 2 :]}
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
    post = _exact_fit(design, Y, genes, shared.weights, ss_within, _resolve_priors(shared.priors), names, shared.n_bsplines, settings)
    K = shared.n_bsplines
    seeds = np.random.SeedSequence(settings["random_seed"]).spawn(len(genes))
    results = []
    for i, gene in enumerate(genes):
        theta = _exact.draws(post, i, settings["draws"], np.random.default_rng(seeds[i]))
        draws = {"intercept": theta[:, 0], "beta": theta[:, 1 : K + 1], "categorical_beta": theta[:, K + 1], "beta_1": theta[:, K + 2 :]}
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
        diag_row = {"gene": gene, "sigma_mean": float(np.sqrt(post.sigma2_mean[i])), "exact_edge_mass": float(post.edge_mass[i])}
        curve_row = {"mean": A @ post.mean[i], "ci_low": ci_low, "ci_high": ci_high, "beta": post.mean[i][1:]}
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
    curve_row = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "beta": beta_mean}
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
    if moments is None:
        wald_stat, wald_pval, wald_df, delta_mean = pattern_wald_test(draws0, draws1)
    else:
        A0, A1 = _condition_maps(shared)
        A = A1 - A0
        delta_mean = A @ moments[0]
        wald_stat, wald_pval, wald_df = _wald_from_moments(delta_mean, A @ moments[1] @ A.T)
    T = float(np.mean(delta_mean**2))
    return {"wald_stat": wald_stat, "wald_pval": wald_pval, "wald_df": wald_df, "T": T}, delta_mean


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
}


_CURVE_SUFFIXES = ("cond0_mean", "cond0_ci_low", "cond0_ci_high", "cond1_mean", "cond1_ci_low", "cond1_ci_high")
_COEF_NAMES = ("beta", "beta_1")


def _coef_columns(n_bsplines):
    """Column labels of a coefficient table, one per spline basis function (strings, so they round-trip through h5ad)."""
    return [f"B{k}" for k in range(n_bsplines)]


def _resolve_inference(inference, model_setup):
    """``"auto"`` picks exact inference where the model allows it (see `_exact.supported`), NUTS otherwise."""
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
            "inference='exact' needs the normal likelihood and normal priors on the coefficients; use 'nuts' or 'advi'."
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
    """The genes worth fitting. Genes whose fitted rows are constant (e.g. all zero) are left out with a
    warning: the model reproduces them exactly, so the noise scale has no lower bound and the posterior
    is improper (the sampler diverges and the test would be meaningless). Their stats stay missing."""
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
    tau=0.1,
    predictive_seed=None,
    inference="auto",
    coefs_key="dsvg_coefs",
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
        With ``inference="advi"`` they are the settings of the variational
        fit instead (see `inference`).
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

        With data binned with `sample` (several slides per condition) only
        ``"parallel"`` is available. The model then has a random level and a
        random curve per slide, so the covariance of the interaction
        coefficients holds the variation between slides, and the p-value
        uses an F reference with ``n_slides - 2`` denominator degrees of
        freedom (column ``parallel_df_denominator``). The fitted curves are
        the population-level ones. Fit on bins (NUTS is fine) or on cells
        (``use_binned=False``, e.g. with ``inference="advi"``).

        Both tests also write ``integral`` (the area between the curves),
        the ``categorical_beta`` summary and the sampler diagnostics.
        Binned fits with overlapping windows (`diffsvg.pp.bin_obs`'s default
        `window_radius`) treat neighbouring bins as independent although they
        share cells, which makes the model over-confident and the
        ``"parallel"`` p-values too small (a warning says so); contiguous
        bins (``bin_obs(..., window_radius=None)``) and fits on the cells
        (``use_binned=False``) do not have that problem.
    tau
        Only for ``test="parallel"``: the smallest shape difference that
        counts, in the units of ``shape_diff`` (see `parallel_curves_test`).
        The p-value tests ``shape_diff <= tau``, so smaller differences do
        not become significant however many cells there are. ``0`` tests
        whether the curves are exactly parallel.
    predictive_seed
        Deprecated, has no effect. The curves, their bands, ``integral`` and
        the Wald test are now computed from the posterior draws of the mean
        function (no observation noise), so they need no extra random draws.
    inference
        How each gene's posterior is obtained:

        - ``"nuts"`` (default): MCMC with `pymc.sample`, configured by
          `sample_kwargs`. Reports ``n_divergent``, ``rhat_max`` and
          ``ess_bulk_min``.
        - ``"advi"``: full-rank variational inference, started at the
          Laplace approximation and polished with Adam (settings: ``n``,
          ``learning_rate``, ``final_learning_rate``, ``draws``, ... see
          ``_VI_DEFAULTS``; they go in `sample_kwargs`). Much cheaper than
          NUTS where there are many rows, i.e. with ``use_binned=False``.
          Reports ``vi_khat`` and ``vi_ess`` instead of the MCMC columns:
          importance-sampling diagnostics of the approximation against the
          true posterior (``vi_khat`` below ~0.7 and ``vi_ess`` above ~0.5
          mean the covariance, which the ``"parallel"`` test is built on,
          can be trusted; check them before reading the p-values). Mean-field
          ADVI is deliberately not offered: it ignores the correlations of
          the spline coefficients and so gets that covariance badly wrong.
          ``n=0`` returns the Laplace approximation itself.
          With slides (`diffsvg.pp.bin_obs` with `sample`) the setting
          ``slide_scales`` decides how the two slide scales are handled:
          ``"empirical_bayes"`` (default; fixed at the mode of their marginal
          posterior, matches NUTS on simulated slides) or ``"data_start"``
          (random, started at a least-squares guess; drifts to too small
          scales and anti-conservative p-values on simulated slides, kept
          for comparison). The polish then uses smaller steps
          (``_VI_SLIDE_DEFAULTS``), the stats gain ``sample_level_sd`` and
          ``sample_curve_sd``, and ``vi_ess`` above ~0.1 suffices: it falls
          with the number of parameters even for a good approximation.
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
    if tau < 0:
        raise ValueError(f"`tau` must be >= 0, got {tau}.")
    inference = _resolve_inference(inference, model_setup)
    if inference == "exact":
        sample_defaults = _exact_settings(sample_kwargs)
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
    else:
        sample_defaults = dict(_VI_DEFAULTS)
        if model_setup is not None and model_setup.binned.sample is not None:
            sample_defaults.update(_VI_SLIDE_DEFAULTS)
    if inference != "exact":
        sample_defaults.update(sample_kwargs or {})
        sample_defaults.update(pymc_kwargs)
    if inference == "advi":
        _check_vi_settings(sample_defaults)

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
    if n_samples is not None:
        if test != "parallel":
            raise ValueError("Data binned with `sample` support only `test='parallel'`.")
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
        shape_metric=shape_metric(binned.n_bsplines, binned.degree) if test == "parallel" else None,
        slope_weights=_slope_weights(binned.n_bsplines, binned.degree) if test == "parallel" else None,
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
    _TESTS[test][1](stats_df, null_quantile)
    test_params = {"test": test, "tau": float(tau)} if test == "parallel" else {"test": test}
    if inference == "advi" and sample_defaults["diagnostic_draws"] > 0:
        min_ess = _VI_MIN_ESS[binned.sample is not None]
        failing = ~((stats_df["vi_khat"] < 0.7) & (stats_df["vi_ess"] > min_ess))
        if failing.any():
            warnings.warn(
                f"{int(failing.sum())} of {len(failing)} genes fail the variational diagnostics (expected: vi_khat < 0.7 "
                f"and vi_ess > {min_ess}), so the approximation does not match the posterior for them: do not read their "
                "p-values."
                + (" With slides, try the other `slide_scales` setting or NUTS on bins." if binned.sample else ""),
                UserWarning,
                stacklevel=2,
            )

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
        sample_defaults = _exact_settings(sample_kwargs)
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
):
    """Compare one gene's expression profile against every other gene's.

    Each gene is reduced to a single curve along `anchor` (see
    `_gene_profiles`), and the probe gene is then tested against each of the
    remaining genes individually. The per-comparison statistics mirror
    `fit_model`'s: `T` is the functional L2 distance between the two curves
    and drives the same empirical/Gamma null p-values, computed here over
    the population of "rest" genes rather than over conditions.

    No Wald test is reported: it tests the interaction coefficient, which
    the condition-free model does not have, and which is not what is being
    compared here anyway.

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
        Fraction of the "rest" genes assumed to follow the null: the Gamma
        null is fitted to the lower `null_quantile` of their ``T`` values, as a
        truncated sample (see `_gamma_null_pvalues`).
    inplace
        If `True`, also write the table to ``adata.varm[key_added]``
        (reindexed to ``adata.var_names``, so the probe gene's own row is
        `NaN`). The table is returned either way.
    key_added
        `adata.varm` key used when `inplace`; defaults to
        ``f"dsvg_ovr_{gene}"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by the "rest" genes, with ``delta`` (the signed area between
        the probe curve and that gene's, negative when the probe sits below
        it), ``T``, and the Gamma-null p-value ``gamma_pval`` with its
        BH-adjusted ``gamma_pval_adj``. (The empirical p-value that used to be
        here was circular: it always gave the top `1 - null_quantile` of the
        genes the smallest possible p-value.)
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


def _wald_from_moments(mu, Sigma, ridge=0.0):
    """Wald statistic of ``mu`` against covariance `Sigma`, pseudoinverse over its non-null directions.

    The cutoff is relative, so the degrees of freedom are the rank of the difference curve, not the
    number of grid points. Returns ``(stat, pval, df)``.
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if ridge:
        eigvals = eigvals + ridge * eigvals.max()
    keep = eigvals > eigvals.max() * 1e-10
    z = eigvecs[:, keep].T @ mu
    W = float((z**2 / eigvals[keep]).sum())
    df = int(keep.sum())
    return W, float(chi2.sf(W, df)), df


def _shape_grid(n_bsplines, degree, n_grid):
    """Fine grid on ``[0, 1]``, its trapezoid weights (summing to 1) and the uncentred basis on it."""
    x = np.linspace(0.0, 1.0, n_grid)
    w = np.full(n_grid, 1.0 / (n_grid - 1))
    w[[0, -1]] /= 2
    return x, w, _bspline_basis_01(x, n_bsplines, degree)


def shape_metric(n_bsplines=10, degree=3, n_grid=2001):
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

    Returns
    -------
    numpy.ndarray
        `Q`, shape ``(n_bsplines, n_bsplines)``, positive semi-definite with
        rank ``n_bsplines - 1``.
    """
    _, w, B = _shape_grid(n_bsplines, degree, n_grid)
    centred = B - w @ B
    return 12.0 * (centred * w[:, None]).T @ centred


def _slope_weights(n_bsplines, degree, n_grid=2001):
    """`g` with ``g @ b`` = the least-squares slope over ``[0, 1]`` of the difference curve for coefficients `b`."""
    x, w, B = _shape_grid(n_bsplines, degree, n_grid)
    return 12.0 * ((x - 0.5) * w) @ B


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
