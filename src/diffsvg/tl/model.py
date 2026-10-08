"""Per-gene GAM (P-spline) models for differential spatially/pseudotemporally variable genes.

Preprocessing (binning `adata` along `anchor`/`condition`, assigning each
cell to a bin, and stashing the result in ``adata.uns["binned"]``) lives in
`diffsvg.pp.bin_obs`; this module only defines and fits models given that
output. The public entry points here are `setup_model` and `fit_model`:

- `setup_model` reads ``adata.uns["binned"]`` (written by
  `diffsvg.pp.bin_obs`), validates it against the requested `likelihood`,
  and returns a `DiffSVGModelSetup` - a lightweight config, not a
  collection of built models: with potentially thousands of genes, holding
  a `pm.Model` per gene in memory at once is wasteful when every one of
  them gets rebuilt from scratch per gene anyway (see below). It does no
  binning itself and does not modify `adata`.
- `fit_model` samples the posterior of every per-gene model - in parallel,
  across worker processes - samples the posterior predictive on a fixed
  grid for both levels of `condition`, computes a functional Wald test
  (tradeSeq-style) between the two predicted curves, and writes the result
  to `adata.varm`.

Parallelization note
---------------------
Every gene needs its *own* model, because it has its own target `Y`. Two
tempting-but-broken ways to parallelize that:

1. Build one model, fit it, then loop over genes calling
   ``pm.set_data({"Y_data": next_gene}); pm.sample(...)`` on the *same*
   compiled model - this is what a naive sequential version (and the
   original draft of this module) does. Doing this **concurrently** from
   multiple threads/processes against the same model is not safe: the
   compiled sampler holds mutable/cached state, so one worker's
   `pm.set_data` call can clobber the data another worker is mid-sample on.
2. Build all `n_genes` models up front in the main process (so each gene
   gets its own model, no set_data races) and hand each `pm.Model` to a
   worker via `joblib`. This turns out to be broken for a different reason:
   a `pm.Model` holds PyTensor graph nodes whose IPython ``_repr_latex_`` is
   a `functools.partial`, which `cloudpickle` cannot serialize - so the
   model object can never actually cross a process boundary once built
   (verified: `joblib`/`loky` raises a `PicklingError` on it).

The pattern used here avoids both: everything a per-gene model needs is
either (a) shared, plain-numpy-array data that does not depend on the gene
(the spline design matrices `B`/`B_masked`, the encoded `condition` column,
...) or (b) a single per-gene vector (`Y`) plus a scalar
(`intercept_initval`). `diffsvg.pp.bin_obs` computes both once, up front, in
the main process. Neither `setup_model` nor `fit_model` ever build/hold a
whole gene's worth of `pm.Model` objects at once - `fit_model` ships only
that lightweight, pure-data payload to each worker; the worker then calls
`_build_gene_model` to construct its **own private** model from scratch
(cheap - graph construction, not sampling) and samples it. Every gene's
model lives in its own process for its entire lifetime, so nothing is ever
shared,
swapped-in-place, or raced across workers.
"""

import contextlib
import logging
import os
import warnings
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import chi2, gamma
from tqdm.auto import tqdm

from diffsvg.pp.utils import BinnedData, build_spline_basis, compute_interior_knots

# ============================================================
# 1) Likelihoods
# ============================================================
# Registry so additional observation models can be added later without
# touching `_build_gene_model`. Only "normal" is implemented for now.


def _normal_likelihood(mu, observed, total_size):
    sigma = pm.HalfNormal("sigma", 1, initval=0.5)
    return pm.Normal("D", mu=mu, sigma=sigma, observed=observed, total_size=total_size)


_LIKELIHOODS = {
    "normal": _normal_likelihood,
}


# ============================================================
# 2) Model setup
# ============================================================


@dataclass
class DiffSVGModelSetup:
    """Config bundle returned by `setup_model` and consumed by `fit_model`.

    Deliberately holds no `pm.Model` objects - just enough configuration to
    build one for any given gene. `fit_model` builds each gene's model on
    the fly, per worker, right before sampling it (see the module
    docstring), so materializing a whole gene panel's worth of models here
    up front would just be extra memory held for no benefit; to inspect a
    single gene's model structure (e.g. via `pm.model_to_graphviz`), call
    `diffsvg.tl.model._build_gene_model` for that one gene directly instead.

    Attributes
    ----------
    binned
        The `diffsvg.pp.utils.BinnedData` this setup was built from - holds
        `Y`, the shared design matrices, and everything else that does not
        depend on PyMC.
    likelihood
        Observation model every gene's model is built with.
    """

    binned: BinnedData
    likelihood: str


def _build_gene_model(Y, B, B_masked, condition_arr, n_bsplines, intercept_initval, likelihood="normal"):
    """Build the P-spline GAM for a single gene.

    `B`/`B_masked`/`condition_arr` (shared across genes, see `setup_model`)
    only depend on `anchor`/`condition`; only `Y` (the binned expression)
    and `intercept_initval` change per gene. Building this is cheap (graph
    construction only, no compilation/sampling), so it is safe - and
    intentional - to call it again inside each worker in `fit_model` instead
    of trying to reuse a model built elsewhere.
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    likelihood_fn = _LIKELIHOODS[likelihood]

    with pm.Model() as gene_model:
        Y_data = pm.Data("Y_data", np.asarray(Y, dtype=float))
        B_data = pm.Data("B_data", B)
        condition_data = pm.Data("condition_data", np.asarray(condition_arr, dtype=float))
        B_masked_data = pm.Data("B_masked_data", B_masked)

        intercept = pm.Normal("intercept", mu=0, sigma=1, initval=intercept_initval)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_bsplines, initval=np.zeros(n_bsplines))
        beta_1 = pm.Normal("beta_1", mu=0, sigma=1, shape=n_bsplines, initval=np.zeros(n_bsplines))
        categorical_beta = pm.Normal("categorical_beta", mu=0, sigma=1, initval=0.0)

        mu = pm.Deterministic(
            "mu",
            intercept
            + pm.math.dot(B_data, beta)
            + categorical_beta * condition_data
            + pm.math.dot(B_masked_data, beta_1),
        )

        likelihood_fn(mu, Y_data, len(Y))

    return gene_model


def _build_profile_model(Y, B, n_bsplines, intercept_initval, likelihood="normal"):
    """Build the condition-free P-spline GAM for a single gene.

    Same priors, spline basis and likelihood as `_build_gene_model`, minus
    the `condition` covariate and its interaction with the spline: with no
    conditions there is nothing for them to encode, and the Wald test on the
    interaction coefficient does not apply either (see `one_vs_rest`, which
    ranks on the curve distance instead). Kept as its own builder rather
    than branching inside `_build_gene_model`, so each model reads as one
    straight-line definition.
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    likelihood_fn = _LIKELIHOODS[likelihood]

    with pm.Model() as gene_model:
        Y_data = pm.Data("Y_data", np.asarray(Y, dtype=float))
        B_data = pm.Data("B_data", B)

        intercept = pm.Normal("intercept", mu=0, sigma=1, initval=intercept_initval)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_bsplines, initval=np.zeros(n_bsplines))

        mu = pm.Deterministic("mu", intercept + pm.math.dot(B_data, beta))

        likelihood_fn(mu, Y_data, len(Y))

    return gene_model


def setup_model(adata, likelihood="normal"):
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

    Returns
    -------
    DiffSVGModelSetup
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {likelihood!r}. Available: {sorted(_LIKELIHOODS)}")
    if "binned" not in adata.uns:
        raise KeyError("adata.uns['binned'] not found - run `diffsvg.pp.bin_obs(adata, ...)` first.")

    binned = BinnedData(**adata.uns["binned"])
    return DiffSVGModelSetup(binned=binned, likelihood=likelihood)


# ============================================================
# 3) Fitting (parallel across genes) + posterior predictive
# ============================================================


@dataclass
class _FitShared:
    """The plain-array payload shared by every gene, shipped once per task.

    Deliberately holds no PyMC/PyTensor objects (see the module docstring) -
    everything here pickles as ordinary numpy arrays, so there is nothing
    for concurrent workers to race on or overwrite.
    """

    B: np.ndarray
    B_masked: np.ndarray
    condition_fit: np.ndarray
    n_bsplines: int
    likelihood: str
    bin_centers: np.ndarray
    B_pred: np.ndarray
    B_pred_masked_0: np.ndarray
    B_pred_masked_1: np.ndarray
    condition_pred_0: np.ndarray
    condition_pred_1: np.ndarray


@dataclass
class _ProfileShared:
    """`_FitShared`'s counterpart for the condition-free model - same contract, fewer arrays."""

    B: np.ndarray
    n_bsplines: int
    likelihood: str
    B_pred: np.ndarray


def _fit_inputs(adata, binned, use_binned):
    """The targets and design matrices to fit on: the binned aggregates, or one row per cell.

    ``use_binned=True`` hands back exactly what `diffsvg.pp.bin_obs`
    precomputed. ``use_binned=False`` rebuilds the same quantities at cell
    resolution - the *same* knots, degree and condition encoding, so the two
    stay comparable - and reads `Y` straight from `adata` rather than
    duplicating the expression matrix into ``adata.uns``.

    Returns ``(B, B_masked, condition_fit, Y, intercept_initvals)``.
    """
    if use_binned:
        return binned.B, binned.B_masked, binned.condition_fit, binned.Y, binned.intercept_initvals

    design = adata.obsm[binned.design_matrix]
    anchor_vals = design[binned.anchor].to_numpy(dtype=float)
    interior_knots = compute_interior_knots(binned.n_bsplines, degree=binned.degree)[1:-1]
    B, _ = build_spline_basis(anchor_vals, interior_knots, degree=binned.degree)

    condition_fit, B_masked = None, None
    if binned.has_condition:
        code_map = {label: float(code) for code, label in binned.condition_categories.items()}
        condition_fit = design[binned.condition].map(code_map).to_numpy(dtype=float)
        if np.isnan(condition_fit).any():
            raise ValueError(
                f"`condition` column {binned.condition!r} holds values that were not seen when binning "
                f"(expected {sorted(code_map)}) - re-run `diffsvg.pp.bin_obs`."
            )
        B_masked = B * condition_fit[:, None]

    sub = adata[:, binned.genes]
    X = sub.layers[binned.layer] if binned.layer is not None else sub.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    X = np.asarray(X, dtype=float)

    Y = {gene: X[:, i] for i, gene in enumerate(binned.genes)}
    # Same intent as the binned initval (mean expression at the anchor
    # origin), but no cell sits exactly at 0, so the first bin stands in.
    first_bin = np.asarray(binned.bin_assignment) == 0
    intercept_initvals = {
        gene: float(y[first_bin].mean()) if first_bin.any() else float(y.mean()) for gene, y in Y.items()
    }
    return B, B_masked, condition_fit, Y, intercept_initvals


@contextlib.contextmanager
def _quiet_fds():
    """Redirect the OS-level stdout/stderr file descriptors to `os.devnull`.

    `contextlib.redirect_stdout`/`redirect_stderr` only swap the Python
    `sys.stdout`/`sys.stderr` objects, which misses writes from compiled
    extensions that write straight to the process's file descriptors -
    `nutpie`'s sampler is a Rust extension, so it could. This redirects the
    actual fds instead, so it catches those too.

    Safe to wrap around code that might raise: a raised exception is a
    Python object that propagates through the normal call stack (and, here,
    through `joblib`) regardless of what stdout/stderr are pointed at: it
    is not lost, and by the time anything prints it (e.g. `joblib` re-
    raising in the main process), the `finally` below has already restored
    the real fds.
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
    """Silence PyMC/PyTensor/nutpie's sampler announcements, warnings, and any raw fd output.

    Scoped tightly around the actual `pm.sample`/`pm.sample_posterior_predictive`
    calls in `_fit_and_predict_gene`, i.e. this runs inside whichever
    process actually samples (the main process for `n_workers=1`, a worker
    process otherwise - see the module docstring on why it has to be here
    rather than configured once up front in `fit_model`). The outer `tqdm`
    progress bar over completed genes lives in the parent process and
    writes to stderr there, so it is unaffected either way.
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


def _fit_and_predict_gene(gene, Y, intercept_initval, shared, sample_kwargs):
    """Build + sample one gene's own private model, then its posterior predictive at both condition levels.

    Runs inside a worker process. The model is built fresh, right here, from
    plain arrays (`Y`, `shared`) - it is never received pre-built from the
    main process, so there is no compiled model/cache shared with any other
    worker to race on. Returns ``(gene, stats_row, curve_row)`` where
    `stats_row` is a flat dict and `curve_row` is a dict of 1D arrays (each
    of length ``len(shared.B_pred)``): the ``condition==1`` minus
    ``condition==0`` curve (``"delta"``), and, per condition, the posterior
    mean curve and a 94% equal-tailed credible band (``"cond{0,1}_mean"``,
    ``"cond{0,1}_ci_low"``, ``"cond{0,1}_ci_high"``).
    """
    model = _build_gene_model(
        Y, shared.B, shared.B_masked, shared.condition_fit, shared.n_bsplines, intercept_initval, shared.likelihood
    )
    n_grid = shared.B_pred.shape[0]

    # Everything below is wrapped in `_quiet_pymc()` (not just `pm.sample`
    # itself): `pm.sample_posterior_predictive` and `az.rhat`/`az.ess` can
    # each emit their own logging/warnings too, and the goal here is a
    # clean run with only the outer progress bar and real errors visible.
    with model, _quiet_pymc():
        idata = pm.sample(**sample_kwargs)

        pm.set_data(
            {
                "B_data": shared.B_pred,
                "condition_data": shared.condition_pred_0,
                "B_masked_data": shared.B_pred_masked_0,
                "Y_data": np.zeros(n_grid),
            }
        )
        pred0 = pm.sample_posterior_predictive(idata, var_names=["D"], extend_inferencedata=False, progressbar=False)

        pm.set_data(
            {
                "B_data": shared.B_pred,
                "condition_data": shared.condition_pred_1,
                "B_masked_data": shared.B_pred_masked_1,
                "Y_data": np.zeros(n_grid),
            }
        )
        pred1 = pm.sample_posterior_predictive(idata, var_names=["D"], extend_inferencedata=False, progressbar=False)

        draws0 = pred0.posterior_predictive["D"].stack(sample=("chain", "draw")).to_numpy().T  # (n_draws, n_grid)
        draws1 = pred1.posterior_predictive["D"].stack(sample=("chain", "draw")).to_numpy().T

        wald_stat, wald_pval, wald_df, delta_mean = pattern_wald_test(draws0, draws1)
        cond0_mean, cond1_mean = draws0.mean(axis=0), draws1.mean(axis=0)
        cond0_ci_low, cond0_ci_high = np.quantile(draws0, [0.03, 0.97], axis=0)
        cond1_ci_low, cond1_ci_high = np.quantile(draws1, [0.03, 0.97], axis=0)
        T = compute_T(cond0_mean, cond1_mean)
        integral = float(np.trapezoid(delta_mean, x=shared.bin_centers))

        diag_vars = ["intercept", "beta", "beta_1", "categorical_beta", "sigma"]
        n_divergent = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0

        # Computed directly from the posterior (rather than via `az.summary`'s
        # column names, e.g. "hdi_3%"), since those have changed across arviz
        # versions; `az.rhat`/`az.ess` return a variable -> array mapping
        # (`.data_vars`) which has stayed stable, so those are used for the
        # convergence diagnostics.
        post_cb = idata.posterior["categorical_beta"].to_numpy().ravel()
        ci_low, ci_high = (float(v) for v in np.quantile(post_cb, [0.03, 0.97]))

        rhat_ds = az.rhat(idata, var_names=diag_vars)
        ess_ds = az.ess(idata, var_names=diag_vars, method="bulk")
        rhat_max = float(max(np.asarray(rhat_ds[v]).max() for v in rhat_ds.data_vars))
        ess_bulk_min = float(min(np.asarray(ess_ds[v]).min() for v in ess_ds.data_vars))

    stats_row = {
        "gene": gene,
        "wald_stat": wald_stat,
        "wald_pval": wald_pval,
        "wald_df": wald_df,
        "T": T,
        "integral": integral,
        "categorical_beta_mean": float(post_cb.mean()),
        "categorical_beta_ci_low": ci_low,
        "categorical_beta_ci_high": ci_high,
        "n_divergent": n_divergent,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
    }
    curve_row = {
        "delta": delta_mean,
        "cond0_mean": cond0_mean,
        "cond0_ci_low": cond0_ci_low,
        "cond0_ci_high": cond0_ci_high,
        "cond1_mean": cond1_mean,
        "cond1_ci_low": cond1_ci_low,
        "cond1_ci_high": cond1_ci_high,
    }
    return gene, stats_row, curve_row


def _fit_and_predict_profile(gene, Y, intercept_initval, shared, sample_kwargs):
    """Build + sample one gene's condition-free model, then its single posterior predictive curve.

    The `_fit_and_predict_gene` counterpart for data without conditions: one
    curve per gene instead of one per condition, so there is no delta and no
    differential statistic here - `one_vs_rest` derives those by comparing
    genes against each other. Returns ``(gene, diag_row, curve_row)``.
    """
    model = _build_profile_model(Y, shared.B, shared.n_bsplines, intercept_initval, shared.likelihood)
    n_grid = shared.B_pred.shape[0]

    with model, _quiet_pymc():
        idata = pm.sample(**sample_kwargs)

        pm.set_data({"B_data": shared.B_pred, "Y_data": np.zeros(n_grid)})
        pred = pm.sample_posterior_predictive(idata, var_names=["D"], extend_inferencedata=False, progressbar=False)

        draws = pred.posterior_predictive["D"].stack(sample=("chain", "draw")).to_numpy().T  # (n_draws, n_grid)
        mean = draws.mean(axis=0)
        ci_low, ci_high = np.quantile(draws, [0.03, 0.97], axis=0)

        diag_vars = ["intercept", "beta", "sigma"]
        n_divergent = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
        rhat_ds = az.rhat(idata, var_names=diag_vars)
        ess_ds = az.ess(idata, var_names=diag_vars, method="bulk")
        rhat_max = float(max(np.asarray(rhat_ds[v]).max() for v in rhat_ds.data_vars))
        ess_bulk_min = float(min(np.asarray(ess_ds[v]).min() for v in ess_ds.data_vars))

    diag_row = {"gene": gene, "n_divergent": n_divergent, "rhat_max": rhat_max, "ess_bulk_min": ess_bulk_min}
    curve_row = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    return gene, diag_row, curve_row


def _compute_null_pvalues(T_values, null_quantile=0.9):
    """Empirical + Gamma-null p-values for an array of functional Wald/`T` statistics.

    tradeSeq-style: the lower `null_quantile` fraction of `T_values` is
    treated as an empirical null population.
    """
    T_values = np.asarray(T_values, dtype=float)
    cutoff = np.nanquantile(T_values, null_quantile)
    null_pool = T_values[T_values <= cutoff]

    empirical_pvals = np.array([(1 + np.sum(null_pool >= t)) / (1 + len(null_pool)) for t in T_values])

    shape, loc, scale = gamma.fit(null_pool, floc=0)
    gamma_pvals = 1 - gamma.cdf(T_values, a=shape, loc=loc, scale=scale)

    return empirical_pvals, gamma_pvals


def _benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (kept dependency-free rather than pulling in `statsmodels`)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty(n)
    out[order] = adjusted
    return out


_CURVE_SUFFIXES = ("cond0_mean", "cond0_ci_low", "cond0_ci_high", "cond1_mean", "cond1_ci_low", "cond1_ci_high")


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
    **pymc_kwargs,
):
    """Fit every per-gene model from `setup_model`, in parallel, and summarize the result.

    For each gene: (re-)builds its own private model and samples the
    posterior, then samples the posterior predictive on the fixed `anchor`
    grid under both levels of `condition`, and runs a functional
    (tradeSeq-style) Wald test between the two predicted curves.

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
        directly comparable.
    progressbar
        Show a `tqdm` progress bar over completed genes. Per-worker MCMC
        progress bars are always disabled (they would interleave garbage
        across processes); pass e.g. ``sample_kwargs={"progressbar": True}``
        to override that only when `n_workers=1`.
    inplace
        If `True` (default), write results to `adata.varm`/`adata.uns`
        (aligned to ``adata.var_names``, with genes not in `model_setup`
        left as `NaN`) and return `None`. If `False`, leave `adata` untouched
        and return ``(stats_df, preds_df, curve_dfs)``.
    stats_key
        `adata.varm` key for the per-gene summary statistics (`n_genes x n`;
        see `diffsvg.pl.plot_volcano`/`diffsvg.pl.plot_pvalue_histogram`).
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
        Passed to `_compute_null_pvalues` for the empirical/Gamma null
        p-values.
    sample_kwargs
        Extra keyword arguments merged into the `pymc.sample` call (applied
        to every gene). Any extra keyword arguments to `fit_model` itself are
        forwarded the same way, so ``fit_model(..., chains=2)`` also works.

    Returns
    -------
    None or tuple[pandas.DataFrame, pandas.DataFrame, dict[str, pandas.DataFrame]]
    """
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
    shared = _FitShared(
        B=B,
        B_masked=B_masked,
        condition_fit=condition_fit,
        n_bsplines=binned.n_bsplines,
        likelihood=model_setup.likelihood,
        bin_centers=binned.bin_centers,
        B_pred=binned.B_pred,
        B_pred_masked_0=binned.B_pred_masked_0,
        B_pred_masked_1=binned.B_pred_masked_1,
        condition_pred_0=binned.condition_pred_0,
        condition_pred_1=binned.condition_pred_1,
    )

    n_jobs = 1 if n_workers is None else n_workers
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(_fit_and_predict_gene)(gene, Y[gene], intercept_initvals[gene], shared, sample_defaults)
        for gene in binned.genes
    )
    if progressbar:
        results = tqdm(results, total=len(binned.genes), desc="Fitting DiffSVG models")

    stats_rows = []
    curve_rows = {}
    for gene, stats_row, curve_row in results:
        stats_rows.append(stats_row)
        curve_rows[gene] = curve_row

    stats_df = pd.DataFrame(stats_rows).set_index("gene").reindex(binned.genes)
    stats_df["empirical_pval"], stats_df["gamma_pval"] = _compute_null_pvalues(
        stats_df["T"].to_numpy(), null_quantile=null_quantile
    )
    stats_df["empirical_pval_adj"] = _benjamini_hochberg(stats_df["empirical_pval"].to_numpy())
    stats_df["gamma_pval_adj"] = _benjamini_hochberg(stats_df["gamma_pval"].to_numpy())

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

    if not inplace:
        return stats_df, preds_df, curve_dfs

    adata.varm[stats_key] = stats_df.reindex(adata.var_names)
    adata.varm[preds_key] = preds_df.reindex(adata.var_names)
    for suffix, df in curve_dfs.items():
        adata.varm[f"{curves_key}_{suffix}"] = df.reindex(adata.var_names)
    adata.uns[f"{curves_key}_condition_categories"] = binned.condition_categories
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
    n_workers, progressbar, use_binned, sample_kwargs, **pymc_kwargs
        As in `fit_model`.
    inplace
        If `True` (default), write the results to `adata.varm` and return
        `None`; otherwise return ``(profile_dfs, diagnostics_df)``.
    profiles_key
        Prefix for three `adata.varm` tables (`n_genes x (n_bins + 1)`), one
        per ``{"mean", "ci_low", "ci_high"}``, plus a
        ``f"{profiles_key}_diagnostics"`` table of per-gene sampler
        diagnostics.

    Returns
    -------
    None or tuple[dict[str, pandas.DataFrame], pandas.DataFrame]
    """
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
    sample_defaults.update(sample_kwargs or {})
    sample_defaults.update(pymc_kwargs)

    binned = model_setup.binned
    B, _, _, Y, intercept_initvals = _fit_inputs(adata, binned, use_binned)
    shared = _ProfileShared(
        B=B,
        n_bsplines=binned.n_bsplines,
        likelihood=model_setup.likelihood,
        B_pred=binned.B_pred,
    )

    n_jobs = 1 if n_workers is None else n_workers
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(_fit_and_predict_profile)(gene, Y[gene], intercept_initvals[gene], shared, sample_defaults)
        for gene in binned.genes
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
        Passed to `_compute_null_pvalues`.
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
        it), ``T``, and the empirical/Gamma p-values with their BH-adjusted
        counterparts.
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
    result["empirical_pval"], result["gamma_pval"] = _compute_null_pvalues(
        result["T"].to_numpy(), null_quantile=null_quantile
    )
    result["empirical_pval_adj"] = _benjamini_hochberg(result["empirical_pval"].to_numpy())
    result["gamma_pval_adj"] = _benjamini_hochberg(result["gamma_pval"].to_numpy())
    result.attrs["probe_gene"] = gene
    result.attrs["profile_source"] = source

    if inplace:
        adata.varm[key_added or f"dsvg_ovr_{gene}"] = result.reindex(adata.var_names)
    return result


# ============================================================
# 4) Standalone statistics helpers
# ============================================================
# (Used internally by `_fit_and_predict_gene`/`fit_model`, and kept public
# for anyone who wants to run the tradeSeq-style comparison on
# posterior-predictive curves obtained some other way.)


def wald_test(beta, cov, contrast=None):
    """
    Multivariate Wald test.

    Parameters
    ----------
    beta : array-like, shape (p,)
        Coefficient estimates.

    cov : array-like, shape (p, p)
        Covariance matrix of coefficient estimates.

    contrast : array-like, optional
        Contrast matrix C for testing:
            H0: C beta = 0

        If None:
            C = identity matrix

    Returns
    -------
    stat : float
        Wald chi-square statistic.

    df : int
        Degrees of freedom.

    pval : float
        Chi-square p-value.
    """
    beta = np.asarray(beta)
    cov = np.asarray(cov)

    if contrast is None:
        Cbeta = beta
        Ccov = cov
    else:
        C = np.asarray(contrast)
        Cbeta = C @ beta
        Ccov = C @ cov @ C.T

    # numerically stable inverse
    stat = float(Cbeta.T @ np.linalg.solve(Ccov, Cbeta))

    df = len(Cbeta)

    pval = chi2.sf(stat, df)

    return stat, df, pval


def pattern_wald_test(pred0, pred1, ridge=1e-6):
    """
    tradeSeq-like functional Wald test on posterior predictive curves.

    Parameters
    ----------
    pred0 : array, shape (n_draws, M)
        Posterior predictive samples for condition 0

    pred1 : array, shape (n_draws, M)
        Posterior predictive samples for condition 1

    ridge : float
        Small diagonal regularization for covariance stability

    Returns
    -------
    stat : float
        Wald statistic

    pval : float
        Chi-square p-value

    df : int
        Effective degrees of freedom (= rank of covariance)

    delta_mean : array
        Mean curve difference
    """
    # posterior samples of curve differences
    delta = pred1 - pred0  # (draws, M)

    # posterior mean difference
    mu = delta.mean(axis=0)  # (M,)

    # covariance of contrasts
    Sigma = np.cov(delta, rowvar=False)
    # stabilize
    Sigma += ridge * np.eye(Sigma.shape[0])

    # eigendecomposition (same idea as tradeSeq)
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # remove near-singular directions
    keep = eigvals > 1e-8

    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    # pseudoinverse
    Sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

    # Wald statistic
    W = mu.T @ Sigma_inv @ mu

    # effective df
    df = keep.sum()

    # asymptotic p-value
    pval = 1.0 - chi2.cdf(W, df)

    return W, pval, df, mu


def zscore_curve(x):
    """Z-score a 1D prediction curve.

    Removes mean and scale -> shape-only comparison.
    """
    x = np.asarray(x)

    mu = np.mean(x)
    sd = np.std(x)

    return (x - mu) / (sd + 1e-8)


def compute_T(curve0, curve1):
    """Functional L2 distance between two curves."""
    return np.mean((curve1 - curve0) ** 2)


def compute_shape_statistics(pred_dict, null_quantile=0.9):
    """Compute the tradeSeq-style shape statistic and null p-values for a set of gene curve pairs.

    Parameters
    ----------
    pred_dict : dict

        Format:
        {
            gene_name: (pred_cond0, pred_cond1)
        }

        where each prediction array is 1D:
        shape = (n_bins,)

    null_quantile : float
        Lower-tail fraction used for empirical null estimation.

    Returns
    -------
    results_df : pd.DataFrame
    """
    genes = list(pred_dict.keys())
    T_values = np.array([compute_T(pred0, pred1) for pred0, pred1 in pred_dict.values()])

    empirical_pvals, gamma_pvals = _compute_null_pvalues(T_values, null_quantile=null_quantile)

    results_df = pd.DataFrame(
        {
            "gene": genes,
            "T": T_values,
            "empirical_pval": empirical_pvals,
            "gamma_pval": gamma_pvals,
        }
    )

    return results_df.sort_values("T", ascending=False).reset_index(drop=True)
