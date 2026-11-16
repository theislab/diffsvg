# Method

diffsvg compares how a gene's expression changes with the distance to an anchor (an annotated region, a cell type or a set of coordinates) between two conditions.

## Data and binning

Each cell has a distance to the nearest anchor cell, scaled to [0, 1] per section (e.g. with `squidpy.tl.var_by_distance`). `pp.bin_obs` assigns every cell to the nearest of `n_bins + 1` grid points and averages each gene per condition (or per replicate) and grid point. Each bin is weighted by its cell count, and the within-bin sum of squares is kept, so a fit on the bins has the same likelihood as a fit on the cells, up to a constant. Fits on single cells (`use_binned=False`) are also possible.

## Model

For gene expression `y` of cell (or bin) `i` at distance `x` in condition `c` (0 or 1):

```
y_i ~ Normal(mu_i, sigma^2 / w_i)
mu_i = intercept + B(x_i) beta + c_i * categorical_beta + c_i * B(x_i) beta_1
```

`B` is a centred cubic B-spline basis with `n_bsplines` columns and equally spaced knots on [0, 1], and `w_i` is the cell count of a bin (1 for a cell). All coefficients have independent normal priors and `sigma` a half-normal prior (see `tl.model.DEFAULT_PRIORS`). `beta_1` describes how the curve of condition 1 differs from that of condition 0.

With several biological replicates per condition (`bin_obs(..., sample=...)`, e.g. donors, animals or organoids), every replicate `s` adds its own level and curve, `a_s + B(x) b_s`, with `a_s ~ Normal(0, sample_level_sd^2)` and `b_s` with separate scales for its slope direction (`sample_slope_sd`) and the rest of the curve (`sample_curve_sd`). Several sections of one donor or organoid are one replicate, so pass the donor or organoid as `sample`, not the section; otherwise the degrees of freedom below count sections and the tests are too confident. The basis of `b_s` is rotated so that one column carries the slope over the anchor support and the others carry none. `sample_kwargs={"slope_scale": False}` gives the single-scale model. The conditions are then compared against the variation between replicates rather than between cells. Without this, variation between replicates counts as a condition effect.

## Inference

The model is linear and Gaussian once the scale parameters are fixed, so its posterior is computed exactly (`inference="auto"`, the default):

- Given `sigma`, the coefficient posterior is Gaussian, and `sigma` is integrated out numerically on a grid of `log sigma` for each gene.
- With replicates, the three replicate scales are integrated on a log grid (28 points per scale over [1e-5, 5]; 72 x 72 for the single-scale model), and `sigma` with Gauss-Hermite quadrature at each grid point.
- All genes share the design matrix, so the expensive linear algebra is done once for the whole panel.

`exact_edge_mass` reports the posterior mass at the edge of the integration grid. NUTS (`inference="nuts"`, needs `pip install 'diffsvg[pymc]'`) is only needed for non-normal coefficient priors.

## Tests

The difference curve is `d(x) = f1(x) - f0(x) = categorical_beta + B(x) beta_1`.

- `test="parallel"` (default) asks whether the curves differ in shape, beyond a shift in level. The shape difference is `shape_diff = sqrt(12) * RMS(d - mean(d))`; for a gap that changes linearly over [0, 1] it equals the change in slope. The null hypothesis is `shape_diff <= tau`, and the statistic is the Wald statistic of `beta_1` in the shape directions, compared against a noncentral chi-square (noncentral F with replicates, with `n_replicates - 2` denominator degrees of freedom, or for `tau = 0` with `max(n_replicates - 2, shape_eff_df)`). `n_replicates` counts the replicates of both conditions, and `shape_eff_df` is the Satterthwaite degrees of freedom of the shape variance over the posterior of the replicate scales. The shape variance is pooled over the spline directions, so it rests on more than `n_replicates - 2` degrees of freedom; this rule was chosen on the random splits described under Validation. Shape differences below `tau` do not become significant however many cells there are.
- `test="level"` asks whether one curve is higher on average, measured as the mean of `d` over the anchor support, with one degree of freedom (F with `n_replicates - 2` denominator degrees of freedom with replicates). The null hypothesis is `|level_diff| <= tau`.
- `test="slope"` asks whether the gap between the curves grows or shrinks with distance: the least-squares slope of `d` over the anchor support, in units of the full [0, 1] range, tested like the level.
- `test="wald"` asks whether the curves differ at all, in level or shape: a Wald test of `d` against 0 with the posterior covariance. Its degrees of freedom are the rank of that covariance on the anchor support, with a chi-square reference, or F with `n_replicates - 2` denominator degrees of freedom with replicates. It has no `tau`, so with many cells it is best used for ranking.

Fits on contiguous bins (`bin_obs(..., window_radius=None)`, as in the tutorials) test only the anchor support, the central 96% of distances where both conditions have cells. Outside it the curves are extrapolated from the spline and the prior, and including that part made the tests overconfident for cell types that occupy part of the range. Fits on cells (`use_binned=False`) and on overlapping bins use the whole range. With replicates, `fit_model` warns when the support covers less than 75% of the range.

`fit_model(..., two_stage=True)` computes the tests from one curve per replicate instead: a weighted least-squares spline per replicate on the support, then a t-test of the replicates' levels or slopes, or Hotelling's T² on their shape coefficients, with `n_replicates - 2` denominator degrees of freedom. Every replicate counts once and nothing is assumed about how the replicates vary, so it is designed to hold its error rate where the joint model's random curve does not fit, at some cost in power with few replicates. The curves and bands still come from the joint model.

`tl.one_vs_rest` applies the parallel test to pairs of genes fitted with `tl.fit_profiles` (one curve per gene, no condition), and `tl.cluster_curves` groups genes by their spline coefficients.

## Gene sets

With several replicates per condition, `tl.gene_set_test` asks whether the genes of a set change together. Every replicate contributes one level and one slope per gene, from a weighted least-squares line through its bin means on the anchor support (`tl.replicate_summaries`), so the replicate is the unit of replication. The default rotation test ({cite:p}`Wu_2010`, as limma's `roast`) compares the mean per-gene t of a set with random rotations of the replicates' residuals; rotating all genes together keeps their correlation.

The rotation test is self-contained, which means that it asks whether the genes of a set change at all, so a set that contains a few strongly changed genes is significant even if the others don't change. `method="ulm"` asks the competitive question, whether the set changes more than the other genes. The output has `mean_t`, the mean per-gene t (positive: higher or steeper in condition 1); `pval_up` and `pval_down` for a common change up or down; `pval`, two-sided, `2 * min(pval_up, pval_down)`; `pval_mixed`, from the mean absolute t, for genes changing in either direction; and `pval_adj`, Benjamini-Hochberg over the `pval` of the sets of each summary. The smallest p-value is `1 / (n_rotations + 1)`, so a p-value at that floor means "at most" that value. Gene sets can be a dict of gene lists or a decoupler network (e.g. `decoupler.op.hallmark()`); `method="ulm"` runs decoupler's univariate linear model on the per-gene t values instead. On panels of a few hundred genes, pathway collections such as Hallmark or PROGENy have few genes per set; curated sets of a few marker genes work better.

## Validation

The tests were checked on random splits of the replicates of real data, where no gene truly differs, and on data with known changes added:

- The joint model was fitted on the 18 splits of the 8 Legnini organoids {cite:p}`Legnini_2023`, and on the 18 splits of 7 human liver donors, that balance the two conditions between the two groups. At most 1 split per dataset and test had any gene at FDR < 0.05, for each of the shape, level and slope tests. The p-values were conservative on these splits, partly because each group mixes both conditions, which raises the variance between replicates of the responsive genes.
- The rotation gene-set test called 1 to 4% of sets at p < 0.05 on random replicate splits of the liver, organoid and SEA-AD data.
- On the SEA-AD MERFISH data (27 donors split at random into two groups), sets of 10 genes with a level shift of 2% in one group were detected in 80% of cases, and single genes with a 5% shift in 45%.

The scripts are in [`benchmarks/`](https://github.com/theislab/diffsvg/tree/d013e68/benchmarks) of the development branch.

```{bibliography}
:filter: docname in docnames
```
