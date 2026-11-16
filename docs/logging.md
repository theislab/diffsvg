# Logging

diffsvg logs through the shared scverse logger, as `scverse.diffsvg`. One setting controls diffsvg and every other scverse package that uses it:

```python
from scverse_misc.logging import config

config.verbosity = "info"  # default "warning"
```

At `"info"`, `pp.bin_obs` reports how many cells it left out (anchor cells and sections without anchor cells), and `tl.fit_model` reports one line per call with the number of genes, the test, `tau`, the inference, the replicates, the part of the anchor range it tested and the run time. Dropping half or more of the cells, genes or gene sets is reported as a warning at any verbosity. Problems that need a decision, e.g. replicates that look nested in the condition but were not passed as `sample`, are Python warnings.
