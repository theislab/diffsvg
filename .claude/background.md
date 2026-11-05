# Introduction

This document summarizes the purpose behind this package.

# Scientific Background

In spatial transcriptomics it is possible to analyze variables e.g. gene counts w.r.t. morphological markers, i.e. you can track how expression changes with increasing distance to an "anchor point" (which can be coordinate based or a cluster of cells or some annotated region). Comparing this expression trend between samples or conditions is where the differential aspect comes in. We want to detect genes (named spatial differentially expressed genes or differential spatially variable genes) which have different expression patterns by distance between groups.

# Method

To detect these genes we run a generalized additive model defined using PyMC and rank the genes using a wald statistic on interaction term coefficients or by calculating an empirical p-value if the panel size is large enough.

# Implementation

`documentation.md` explains the curve test (formula, every parameter and its source) and why the ADVI model
and its slide handling look as they do.

The package follows the scverse `pp`/`tl`/`pl` convention. The intended order of calls is `pp.bin_obs`
-> `tl.setup_model` -> `tl.fit_model` (or `tl.fit_profiles`) -> the `pl` plots; `tl.one_vs_rest` and
`tl.check_stability` are run afterwards, on their output.

`pp`:
- `bin_obs(adata, anchor, condition, n_bsplines=10, n_bins=50, ...)` - the main preprocessing entry
  point. Averages each gene's expression in sliding windows along `anchor`, separately per
  `condition` level (or pooled if `condition=None`, for the condition-free models below), and
  assigns every cell to its nearest bin. Mutates `adata` in place by default: writes
  `adata.obs[f"{anchor}_bin"]` and everything downstream needs to `adata.uns["binned"]`. The default
  windows (`window_radius=0.02`) overlap, so neighbouring bins share cells; `window_radius=None` bins
  each cell once and records cell counts and within-bin spread, and the fitters then weight bins by
  cell count so a binned fit matches a fit on the cells (needed for calibrated `test="parallel"`). `sample=` names the slide column
  for data with several slides per condition: bins are then per slide and the fitters add a random level and curve per
  slide, with an F reference (`test="parallel"` only; see `.claude/shape_test_plan.md`, Stage 3).
- `bin_counts`, `compute_interior_knots`, `build_spline_basis` - the lower-level pieces `bin_obs`
  is built from (sliding-window averaging, B-spline knot placement, design matrix construction).
  Not normally called directly.

`tl`:
- `setup_model(adata, likelihood="normal", priors=None)` - validates `adata.uns["binned"]` and
  bundles it with the model configuration into a `DiffSVGModelSetup`. Builds no PyMC model itself;
  each gene's model is (re)built on the fly, per worker, once fitting actually starts.
- `fit_model(adata, model_setup, n_workers=1, use_binned=True, ...)` - the main model: one P-spline
  GAM per gene (intercept + spline(anchor) + condition + spline(anchor):condition), fit in parallel
  across genes. Samples the posterior predictive for both condition levels and runs a tradeSeq-style
  functional Wald test between them, or with `test="parallel"` the parallel-curves test
  (`parallel_curves_test`: are the curves parallel, i.e. different only in level, up to a shape
  difference `tau`; per gene, no panel needed; see `.claude/shape_test_plan.md`). Writes
  `adata.varm["dsvg_stats"]` (per-gene statistics: the chosen test's columns, `integral` and sampler
  diagnostics; the test and `tau` go to `adata.uns["dsvg_stats_params"]`), `adata.varm["dsvg_preds"]` (the
  per-bin difference curve) and six `adata.varm["dsvg_curves_..."]` tables (per-condition predicted
  curves with a credible band). The posterior-predictive draws behind the curves are seeded by
  `predictive_seed`, which defaults to the sampler's `random_seed`, so fits are reproducible.
  `inference="advi"` swaps NUTS for full-rank variational inference (Laplace start, Adam polish; settings go in
  `sample_kwargs`, defaults in `_VI_DEFAULTS`) and reports `vi_khat`/`vi_ess` in place of the MCMC diagnostics;
  meant for fits on single cells (`use_binned=False`), see `.claude/advi_plan.md`. With slides
  (`bin_obs(sample=...)`) the slide scales are set by empirical Bayes (`slide_scales`, default) before ADVI.
- `fit_profiles(adata, model_setup, use_binned=True, ...)` - the condition-free counterpart: one
  curve per gene (no condition covariate, no interaction, no Wald test), for data binned without a
  `condition`. Writes `adata.varm["dsvg_profiles_..."]`.
- `one_vs_rest(adata, gene, ...)` - compares one gene's expression profile against every other
  gene's (reduced to a single curve each - an explicit `fit_profiles` result, or the average of
  `fit_model`'s two condition curves if that is all that is available), using the same functional
  distance and null p-values as `fit_model`'s Wald test, but no Wald test itself (there is no
  interaction coefficient to test here).
- `check_stability(adata, anchor, condition, grid, genes=50, rank_by="wald_stat", n_seeds=2, ...)` -
  re-fits the whole pipeline once per combination of a parameter grid (knots, bins, priors,
  likelihood, `use_binned`) and compares the resulting gene rankings: a run-by-run similarity matrix,
  a one-factor-at-a-time sensitivity breakdown, and a noise floor from re-running the baseline under
  a different seed, so a ranking change can be judged against how much the ranking moves from MCMC
  noise alone.
- `cluster_curves(adata, by="average", n_clusters=5, genes=None, scale=False, ...)` - k-means (sklearn) on the posterior
  means of the spline coefficients, which `fit_model` now writes to `adata.varm["dsvg_coefs_beta"]` and
  `["dsvg_coefs_beta_1"]` (`fit_profiles`: `["dsvg_profiles_beta"]`). `by="average"` is `beta + beta_1 / 2`, the combined
  curve (mean of the two condition curves), `"interaction"` is `beta_1`, `"profile"` the condition-free fit. `beta` alone
  is the curve of condition 0 under the 0/1 coding, so it is not offered. Mean coefficient removed always (that direction does not change
  the curve); `scale=True` adds unit length. `n_init=100` by default: 10 gave up to 25 partitions over 30 seeds. Writes
  `adata.var[key]` and `adata.uns[key]`.
- `DEFAULT_PRIORS` / the `priors=` argument of `setup_model` - every prior is overridable, e.g.
  `priors={"beta": ("normal", 0.0, 2.0)}`; unset ones keep the hardcoded defaults.

`pl` (all take the `adata`/tables the functions above wrote, not an `AnnData` column name):
- `plot_volcano(stats_df, ...)` - the usual integral-vs-significance plot off `fit_model`'s stats
  table, with gene labels and collision avoidance.
- `plot_prediction_curve(adata, gene, ...)` - the predicted curve(s) with uncertainty band for one
  gene, or a grid of panels (4 per row by default) for several.
- `plot_pvalue_histogram(stats_df, ...)` - QC histogram of a p-value column.
- `plot_cluster_curves(adata, cluster_key, curves=None, ...)` - a `cluster_curves` result as one holoviews panel per
  cluster with each gene's curve overlaid: `curves` is `"combined"`, `"conditions"` (two panels per cluster),
  `"difference"` or a condition label (that condition alone), by default the curves the clustering compared.
- `plot_stability(result, kind="heatmap"|"sensitivity", ...)` - visualizes a `check_stability`
  result: a clustered agreement heatmap across runs, or a bar chart of which single parameter change
  moved the ranking the most.

# Data

Currently we mainly use a Xenium Mouse brain dataset with the dentate gyrus used as anchor.

# Open problems

A precise hypothesis test is not trivial, as most genes will have slight differences between conditions, causing the test statistic to quickly get huge, resulting in p-value inflation. The temporary fix is to use the wald statistic for ranking alone and separately calculating empirical p-values. The p-value distribution is fine now, but depends on the gene panel. So you can't get a meaningful p-value when only probing one gene, because you need a null distribution to compare to. Additionally we use binned data because most spatial data consists of zero counts which add little information but slow down the model fitting process drastically.
