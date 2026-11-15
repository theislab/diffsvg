# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `tl.gene_set_test` and `tl.replicate_summaries`: gene-set tests with slides as replicates, as a rotation test (default) or with decoupler's univariate linear model; gene sets can be lists or decoupler networks ([#18](https://github.com/theislab/diffsvg/pull/18))
- A separate slide scale for the slope direction of each slide's curve (`sample_slope_sd`), and the columns `level_eff_df`, `slope_eff_df` and `shape_eff_df` ([#19](https://github.com/theislab/diffsvg/pull/19))
- Tutorial "Comparing conditions with replicates"

### Changed

- With slides, the shape test at `tau = 0` uses F with `max(n_slides - 2, shape_eff_df)` denominator degrees of freedom. On balanced splits of organoids and liver donors, at most 1 of 18 splits had any gene at FDR < 0.05, and the shape test finds the SHH targets in the Legnini organoids ([#19](https://github.com/theislab/diffsvg/pull/19))
- Exact fits with slides take about 4 times longer; `sample_kwargs={"slope_scale": False}` gives the previous model

## [0.2.1] - 2026-11-13

### Fixed

- The shape test with `tau > 0` returned p = 1 for every gene when the cells covered only part of the anchor range ([#17](https://github.com/theislab/diffsvg/pull/17))

## [0.2.0] - 2026-11-13

### Added

- `fit_model(..., two_stage=True)`: tests from one curve per slide (t-test for level and slope, Hotelling's T² for shape) ([#16](https://github.com/theislab/diffsvg/pull/16))

### Changed

- The condition tests use only the part of the anchor range where both conditions have cells; `fit_model` warns when that part covers less than 75% of the range ([#16](https://github.com/theislab/diffsvg/pull/16))

### Removed

- The `inference="advi"` alias

## [0.1.0] - 2026-11-12

### Added

- Basic tool, preprocessing and plotting functions
- `tl.cluster_curves` and `pl.plot_cluster_curves`: k-means on the coefficients of the combined curve or the interaction, which `fit_model` and `fit_profiles` now store
- Exact posterior inference (`inference="auto"`, the default); PyMC, nutpie and arviz moved to the optional `pymc` extra ([#8](https://github.com/theislab/diffsvg/pull/8))
- `test="level"` and `test="slope"`, and the Wald test with slides ([#10](https://github.com/theislab/diffsvg/pull/10))

### Changed

- The default test is the shape (parallel curves) test on contiguous bins weighted by their cell counts ([#7](https://github.com/theislab/diffsvg/pull/7))
- `tau` defaults to 0 when slides are modelled ([#10](https://github.com/theislab/diffsvg/pull/10))
- `tl.one_vs_rest` tests every pair of genes with the stored posterior covariance ([#9](https://github.com/theislab/diffsvg/pull/9))
- `pl.plot_prediction_curve` and `pl.plot_cluster_curves` draw with matplotlib and return axes; `holoviews` is no longer a dependency

### Removed

- ADVI inference; the exact posterior covers the same models ([#9](https://github.com/theislab/diffsvg/pull/9))
