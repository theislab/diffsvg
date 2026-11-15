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
- `datasets.legnini23`: the Legnini et al. organoid data, downloaded from the scverse example-data CDN and pinned by sha256
- Tutorial "Comparing conditions with replicates"

### Changed

- With replicates, the shape test at `tau = 0` uses F with `max(n_replicates - 2, shape_eff_df)` denominator degrees of freedom instead of `n_replicates - 2` ([#19](https://github.com/theislab/diffsvg/pull/19))
- Exact fits with slides take about 4 times longer; `sample_kwargs={"slope_scale": False}` gives the previous model

## [0.2.1] - 2026-11-14

### Fixed

- The shape test with `tau > 0` returned p = 1 for every gene when the cells covered only part of the anchor range ([#17](https://github.com/theislab/diffsvg/pull/17))

## [0.2.0] - 2026-11-13

### Added

- `fit_model(..., two_stage=True)`: tests from one curve per slide (t-test for level and slope, Hotelling's T² for shape) ([#16](https://github.com/theislab/diffsvg/pull/16))

### Changed

- The condition tests on contiguous bins use only the part of the anchor range where both conditions have cells; with replicates and `two_stage=False`, `fit_model` warns when that part covers less than 75% of the range ([#16](https://github.com/theislab/diffsvg/pull/16))

### Removed

- The `inference="advi"` alias ([#16](https://github.com/theislab/diffsvg/pull/16))

## [0.1.0] - 2026-11-12

### Added

- Binning (`pp.bin_obs`), model setup and fitting (`tl.setup_model`, `tl.fit_model`), profiles and one-vs-rest tests (`tl.fit_profiles`, `tl.one_vs_rest`), stability checks (`tl.check_stability`) and plots (`pl.plot_volcano`, `pl.plot_prediction_curve`, `pl.plot_pvalue_histogram`)
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

[unreleased]: https://github.com/theislab/diffsvg/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/theislab/diffsvg/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/theislab/diffsvg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/theislab/diffsvg/releases/tag/v0.1.0
