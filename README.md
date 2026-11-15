# diffsvg

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/diffsvg/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/diffsvg/badge/

diffsvg finds genes whose expression changes differently with the distance to an anchor (a region, a cell type or a set of points) between two conditions. It fits one smooth curve per gene and condition, models biological replicates, and tests whether the curves differ in shape, level or slope.

## Getting started

```python
import diffsvg as dvg

# adata.obsm["design_matrix"] holds the distance to the anchor (e.g. from squidpy.tl.var_by_distance),
# the condition and the replicate of every cell
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", sample="donor")
setup = dvg.tl.setup_model(adata)
dvg.tl.fit_model(adata, setup, test="parallel")
dvg.pl.plot_volcano(adata.varm["dsvg_stats"])
```

See the [documentation][], in particular the tutorials and the [API documentation][].

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
