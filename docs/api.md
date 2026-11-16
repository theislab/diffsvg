# API

```{eval-rst}
.. module:: diffsvg
.. module:: diffsvg.pp
.. module:: diffsvg.tl
.. module:: diffsvg.pl
.. module:: diffsvg.datasets
.. currentmodule:: diffsvg
```

## Main workflow

Bin the cells along the distance to the anchor, fit and test every gene, and look at the results.

```{eval-rst}
.. currentmodule:: diffsvg

.. autosummary::
    :toctree: generated

    pp.bin_obs
    tl.setup_model
    tl.fit_model
    tl.gene_set_test
    pl.plot_volcano
    pl.plot_prediction_curve
    datasets.legnini23
```

## Checking the settings

```{eval-rst}
.. currentmodule:: diffsvg

.. autosummary::
    :toctree: generated

    tl.check_stability
    pl.plot_stability
    pl.plot_pvalue_histogram
```

## Single samples and clustering

```{eval-rst}
.. currentmodule:: diffsvg

.. autosummary::
    :toctree: generated

    tl.fit_profiles
    tl.one_vs_rest
    tl.cluster_curves
    pl.plot_cluster_curves
```

## Low-level functions

The building blocks of the functions above, for custom models and tests.

```{eval-rst}
.. currentmodule:: diffsvg

.. autosummary::
    :toctree: generated

    pp.bin_counts
    pp.BinnedData
    pp.build_spline_basis
    pp.compute_interior_knots
    pp.spline_basis_at
    tl.replicate_summaries
    tl.parallel_curves_test
    tl.shape_metric
    tl.DiffSVGModelSetup
    tl.StabilityResult
```
