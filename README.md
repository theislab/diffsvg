# diffsvg

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/diffsvg/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/diffsvg/badge/

diffsvg finds genes whose expression changes differently with the distance to an anchor (a region, a cell type or a set of points) between two conditions. It fits one smooth curve per gene and condition, models biological replicates, and tests whether the curves differ in shape, level or slope.

## Getting started

```python
import diffsvg as dvg
import squidpy as sq

# Distance of every cell to the nearest cell of the anchor group "tumor", per section, scaled to 0 to 1.
# It is stored in adata.obsm["design_matrix"], in a column named after the group, next to the covariates.
sq.tl.var_by_distance(
    adata, cluster_key="region", groups="tumor", library_key="section", covariates=["condition", "donor"]
)
# Bin along the distance, with each donor as one replicate, and test every gene.
dvg.pp.bin_obs(adata, anchor="tumor", condition="condition", sample="donor", layer="log1p")
dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), test="parallel")
dvg.pl.plot_volcano(adata.varm["dsvg_stats"])
```

The results are in `adata.varm["dsvg_stats"]`, one row per gene, e.g. `shape_diff`, `slope_diff`, `parallel_pval` and `parallel_pval_adj` (Benjamini-Hochberg).

Which page to read:

- [Comparing conditions with replicates](https://diffsvg.readthedocs.io/page/notebooks/replicates_tutorial.html): start here, for data with 2 or more donors, animals or organoids per condition.
- [Full tutorial](https://diffsvg.readthedocs.io/page/notebooks/full_tutorial.html): choosing the settings, and clustering genes by their curves.
- [Comparing the tests](https://diffsvg.readthedocs.io/page/notebooks/test_comparison_legnini.html): how the Wald and shape tests and the replicate model differ.
- [Single-slide test](https://diffsvg.readthedocs.io/page/notebooks/single_slide_test.html): one sample, one gene against the rest.
- [Method](https://diffsvg.readthedocs.io/page/method.html) and the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install diffsvg:

1. Install the latest release from [PyPI][]:

   ```bash
   pip install diffsvg  # or `uv add diffsvg`
   ```

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/theislab/diffsvg.git  # (or `uv add`)
   ```

The default inference is exact and needs no sampler. To use NUTS (`inference="nuts"`, only needed for non-normal priors), install the optional extra: `pip install "diffsvg[pymc]"`. Gene-set tests with decoupler networks need `pip install decoupler`.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/diffsvg/issues
[tests]: https://github.com/theislab/diffsvg/actions/workflows/test.yaml
[documentation]: https://diffsvg.readthedocs.io
[changelog]: https://diffsvg.readthedocs.io/page/changelog.html
[api documentation]: https://diffsvg.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/diffsvg
[venv]: https://docs.python.org/3/tutorial/venv.html
