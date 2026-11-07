# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- `tl.cluster_curves` and `pl.plot_cluster_curves`: k-means on the coefficients of the combined curve or the interaction, which `fit_model` and `fit_profiles` now store

### Changed

- `pl.plot_prediction_curve` and `pl.plot_cluster_curves` draw with matplotlib and return axes; `holoviews` is no longer a dependency
