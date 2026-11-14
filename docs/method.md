# Method

diffsvg compares how a gene's expression changes with the distance to an anchor (an annotated region, a cell type or a set of coordinates) between two conditions.

## Data and binning

Each cell has a distance to the nearest anchor cell, scaled to [0, 1] per section (e.g. with `squidpy.tl.var_by_distance`). `pp.bin_obs` assigns every cell to the nearest of `n_bins + 1` grid points and averages each gene per condition (or per slide) and grid point. Each bin is weighted by its cell count, and the within-bin sum of squares is kept, so a fit on the bins has the same likelihood as a fit on the cells, up to a constant. Fits on single cells (`use_binned=False`) are also possible.

## Model

For gene expression `y` of cell (or bin) `i` at distance `x` in condition `c` (0 or 1):

```
y_i ~ Normal(mu_i, sigma^2 / w_i)
mu_i = intercept + B(x_i) beta + c_i * categorical_beta + c_i * B(x_i) beta_1
```

`B` is a centred cubic B-spline basis with `n_bsplines` columns and equally spaced knots on [0, 1], and `w_i` is the cell count of a bin (1 for a cell). All coefficients have independent normal priors and `sigma` a half-normal prior (see `tl.model.DEFAULT_PRIORS`). `beta_1` describes how the curve of condition 1 differs from that of condition 0.

With several slides per condition (`bin_obs(..., sample=...)`), every slide `s` adds its own level and curve, `a_s + B(x) b_s`, with `a_s ~ Normal(0, sample_level_sd^2)` and every coefficient of `b_s ~ Normal(0, sample_curve_sd^2)`. The conditions are then compared against the variation between slides rather than between cells. Without this, slide-to-slide variation counts as a condition effect.

## Inference

The model is linear and Gaussian once the scale parameters are fixed, so its posterior is computed exactly (`inference="auto"`, the default):

- Given `sigma`, the coefficient posterior is Gaussian, and `sigma` is integrated out numerically on a grid of `log sigma` for each gene.
- With slides, the two slide scales are integrated on a shared log grid (72 x 72 points over [1e-5, 5]), and `sigma` with Gauss-Hermite quadrature at each grid point.
- All genes share the design matrix, so the expensive linear algebra is done once for the whole panel.

`exact_edge_mass` reports the posterior mass at the edge of the integration grid. NUTS (`inference="nuts"`, needs `pip install 'diffsvg[pymc]'`) is only needed for non-normal coefficient priors.

## Tests

The difference curve is `d(x) = f1(x) - f0(x) = categorical_beta + B(x) beta_1`.

- `test="parallel"` (default) asks whether the curves differ in shape, beyond a shift in level. The shape difference is `shape_diff = sqrt(12) * RMS(d - mean(d))`; for a gap that changes linearly over [0, 1] it equals the change in slope. The null hypothesis is `shape_diff <= tau`, and the statistic is the Wald statistic of `beta_1` in the shape directions, compared against a noncentral chi-square (noncentral F with `n_slides - 2` denominator degrees of freedom when slides are modelled). Shape differences below `tau` do not become significant however many cells there are.
- `test="wald"` asks whether the curves differ at all, in level or shape: a Wald test of `d` against 0 with the posterior covariance, chi-square with `n_bsplines` degrees of freedom. It has no `tau`, so with many cells it is best used for ranking.

`tl.one_vs_rest` applies the parallel test to pairs of genes fitted with `tl.fit_profiles` (one curve per gene, no condition), and `tl.cluster_curves` groups genes by their spline coefficients.

## Gene sets

With several slides per condition, `tl.gene_set_test` asks whether the genes of a set change together. Every slide contributes one level and one slope per gene, from a weighted least-squares line through its bin means on the anchor support (`tl.replicate_summaries`), so the slide is the unit of replication. The default rotation test (Wu et al. 2010, as limma's `roast`) compares the mean per-gene t of a set with random rotations of the slides' residuals; rotating all genes together keeps their correlation. Gene sets can be a dict of gene lists or a decoupler network (e.g. `decoupler.op.hallmark()`); `method="ulm"` runs decoupler's univariate linear model on the per-gene t values instead.

On random splits of the slides (liver, 7 donors; organoids, 8; 34 to 68 splits each) the rotation test called 1 to 4% of sets at p < 0.05. Sets of 10 genes with a level shift of 2% in one group of a random SEA-AD donor split (27 donors) were detected in 80% of cases, while single genes with a 5% shift were detected in 45%. On panels of a few hundred genes, pathway collections such as Hallmark or PROGENy have few genes per set; curated sets of a few marker genes work better.
