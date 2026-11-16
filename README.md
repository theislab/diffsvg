# diffsvg

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/diffsvg/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/diffsvg/badge/

diffsvg compares how genes change with the distance to an anchor (a region, a cell type or a set of points) between two conditions. It fits one curve per gene and condition, uses biological replicates (donors, animals or organoids) as the unit of comparison, and tests whether the curves differ in shape, level or slope. Gene-set tests summarise changes that are too weak for single genes.

## Getting started

[diffsvg in 10 minutes](https://diffsvg.readthedocs.io/page/notebooks/quickstart.html) runs the whole workflow on example data. Then:

- [Comparing conditions with replicates](https://diffsvg.readthedocs.io/page/notebooks/replicates_tutorial.html): which test to use and how to check the hits.
- [Choosing settings and clustering curves](https://diffsvg.readthedocs.io/page/notebooks/settings_and_clustering.html): whether the ranking depends on the settings, and groups of genes with similar curves.
- [Data without replicates](https://diffsvg.readthedocs.io/page/notebooks/without_replicates.html): one sample, one gene compared with the rest.

The [method page](https://diffsvg.readthedocs.io/page/method.html) describes the model and how it was validated.

## Installation

diffsvg needs Python 3.12 or newer.

```bash
pip install diffsvg
```

The development version installs with `pip install git+https://github.com/theislab/diffsvg.git`. Gene-set tests with decoupler networks need `pip install decoupler`, and NUTS sampling, only needed for non-normal priors, needs `pip install "diffsvg[pymc]"`.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/diffsvg/issues
[tests]: https://github.com/theislab/diffsvg/actions/workflows/test.yaml
[documentation]: https://diffsvg.readthedocs.io
[changelog]: https://diffsvg.readthedocs.io/page/changelog.html
