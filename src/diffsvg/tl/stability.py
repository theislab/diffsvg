"""Grid search over preprocessing/model options, to see how stable the gene ranking is.

`check_stability` re-runs the pipeline per parameter combination (re-binning once per binning signature,
on a gene subsample), ranks on a statistic rather than the panel-dependent p-values, and re-runs the
baseline under other seeds to get a noise floor for the agreement.
"""

import inspect
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from diffsvg.pp.utils import bin_obs

from . import _exact
from .model import DiffSVGModelSetup, _resolve_priors, fit_model, setup_model

#: Parameters that need `bin_obs` re-run; the others only need a refit.
_BIN_PARAMS = ("n_bsplines", "degree", "n_bins", "window_radius", "layer")
_MODEL_PARAMS = ("likelihood", "priors", "use_binned")


def _stability_defaults():
    """Default for every tunable, read off the real signatures so they cannot drift."""
    bin_params = inspect.signature(bin_obs).parameters
    setup_params = inspect.signature(setup_model).parameters
    fit_params = inspect.signature(fit_model).parameters
    return {
        **{name: bin_params[name].default for name in _BIN_PARAMS},
        "likelihood": setup_params["likelihood"].default,
        "priors": setup_params["priors"].default,
        "use_binned": fit_params["use_binned"].default,
    }


def _format_value(value):
    """Compact, readable rendering of a parameter value for run labels."""
    if isinstance(value, dict):  # a priors override
        # The full spec, not just the distribution name: two overrides of the same prior with different
        # parameters must get different run labels.
        return "+".join(f"{name}:{spec[0]}({','.join(str(v) for v in spec[1:])})" for name, spec in sorted(value.items()))
    return str(value)


@dataclass
class StabilityResult:
    """What `check_stability` found; consumed by `diffsvg.pl.plot_stability`.

    Attributes
    ----------
    stats, ranks
        `genes x runs` tables of the ranking statistic and of the rank
        derived from it (``1`` = strongest).
    similarity
        `runs x runs` Spearman correlation between rank columns.
    top_overlap
        `runs x runs` fraction of each run's top `top_k` genes that it
        shares with the other run.
    sensitivity
        One row per configuration that differs from the baseline in exactly
        one parameter, with its agreement against the baseline - i.e. the
        one-factor-at-a-time view of which knob actually moves the ranking.
    configs
        `runs x parameters` table of what each run was.
    noise_floor
        Baseline-vs-reseeded-baseline agreement, as
        ``{"spearman": ..., "top_overlap": ...}``. Agreement at or below
        this is within MCMC noise.
    baseline, rank_by, top_k, genes
        The baseline run's label, and the settings the comparison used.
    """

    stats: pd.DataFrame
    ranks: pd.DataFrame
    similarity: pd.DataFrame
    top_overlap: pd.DataFrame
    sensitivity: pd.DataFrame
    configs: pd.DataFrame
    noise_floor: dict
    baseline: str
    rank_by: str
    top_k: int
    genes: list


def _resolve_genes(adata, genes, random_state):
    if genes is None:
        selected = list(adata.var_names)
    elif isinstance(genes, int):
        rng = np.random.default_rng(random_state)
        size = min(genes, adata.n_vars)
        selected = sorted(rng.choice(np.asarray(adata.var_names), size=size, replace=False).tolist())
    else:
        selected = list(genes)
        missing = set(selected) - set(adata.var_names)
        if missing:
            raise KeyError(f"Genes not found in adata.var_names: {sorted(missing)}")
    if len(selected) < 3:
        raise ValueError(f"need at least 3 genes to compare rankings, got {len(selected)}.")
    return selected


def check_stability(
    adata,
    anchor,
    condition,
    grid,
    baseline=None,
    genes=50,
    rank_by=None,
    higher_is_stronger=True,
    top_k=10,
    n_seeds=2,
    n_workers=1,
    progressbar=True,
    random_state=0,
    design_matrix="design_matrix",
    sample_kwargs=None,
    test="parallel",
    tau=None,
    sample=None,
    inference="auto",
):
    """Re-rank genes under every combination in `grid` and measure how much the ranking moves.

    Parameters
    ----------
    adata
        Annotated data matrix, with `anchor`/`condition` in
        ``adata.obsm[design_matrix]``. Never modified: each run bins and
        fits out-of-place.
    anchor, condition, design_matrix
        Passed to `diffsvg.pp.bin_obs`. `condition` is required - the
        ranking being checked is the differential one, so the condition-free
        models are out of scope here.
    grid
        ``{parameter: [values]}`` over any of
        ``("n_bsplines", "degree", "n_bins", "window_radius", "layer")``
        (these re-bin) and ``("likelihood", "priors", "use_binned")`` (these
        only refit). Priors are varied by passing a list of override dicts,
        e.g. ``{"priors": [{"beta": ("normal", 0.0, 0.5)},
        {"beta": ("normal", 0.0, 2.0)}]}``; see
        `diffsvg.tl.model.DEFAULT_PRIORS`.
    baseline
        The configuration everything is compared against, as a dict of
        parameter values. Defaults to the first value of each `grid` axis.
    genes
        Number of genes to subsample (default 50), an explicit list, or
        `None` for all of them. The same genes are used for every run - a
        different subsample per run would confound the comparison.
    rank_by, higher_is_stronger
        Column of `fit_model`'s statistics table to rank on, and whether
        larger means stronger. Rank on a statistic rather than a p-value:
        the Wald test's p-values are computed against the panel, so
        subsampling shifts them and they do not compare across runs. Defaults
        to ``"parallel_stat"`` for ``test="parallel"`` and ``"wald_stat"`` for
        ``test="wald"``.
    top_k
        Size of the "top genes" set used for the overlap metric.
    n_seeds
        How many times to run the baseline, each with a different
        `random_seed`, to establish the noise floor. `1` skips it. Ignored
        when every run uses exact inference: the statistics are then
        deterministic and the noise floor is 1.
    n_workers, progressbar, sample_kwargs
        As in `fit_model`; `sample_kwargs` is where to make the sweep
        cheaper (e.g. ``{"draws": 300, "tune": 300}``), since ranking needs
        far less precision than a final fit. The per-run progress bars are
        suppressed in favour of one bar over runs.
    random_state
        Seed for the gene subsample.
    test, tau
        Which condition test every run uses, as in `fit_model`.
    inference
        Passed to `fit_model`.
    sample
        Passed to `diffsvg.pp.bin_obs`: the slide column for data with several
        slides per condition, so every run uses the model with slide effects
        (needs ``window_radius=None`` and ``test="parallel"``). Without it,
        multi-slide data is pooled and the rankings are over-confident.

    Returns
    -------
    StabilityResult
    """
    unknown = set(grid) - set(_BIN_PARAMS) - set(_MODEL_PARAMS)
    if unknown:
        raise ValueError(
            f"Unknown grid parameter(s) {sorted(unknown)}. Available: {sorted(_BIN_PARAMS + _MODEL_PARAMS)}"
        )
    if not grid:
        raise ValueError("`grid` is empty - pass at least one parameter to vary.")

    if rank_by is None:
        rank_by = "parallel_stat" if test == "parallel" else "wald_stat"
    defaults = _stability_defaults()
    gene_list = _resolve_genes(adata, genes, random_state)
    sample_kwargs = dict(sample_kwargs or {})
    base_seed = sample_kwargs.pop("random_seed", 42)

    axes = list(grid)
    configs = [
        dict(defaults, **dict(zip(axes, combo, strict=True))) for combo in itertools.product(*(grid[a] for a in axes))
    ]

    baseline_config = dict(defaults, **{a: grid[a][0] for a in axes})
    if baseline is not None:
        unknown_baseline = set(baseline) - set(defaults)
        if unknown_baseline:
            raise ValueError(f"Unknown baseline parameter(s) {sorted(unknown_baseline)}.")
        baseline_config = dict(defaults, **baseline)
        if baseline_config not in configs:
            configs.insert(0, baseline_config)

    # Label runs by every parameter that differs between them, including a `baseline` value that is not on
    # the grid, so that no two runs share a label (they would overwrite each other's column).
    varying = [a for a in defaults if len({_format_value(c[a]) for c in configs}) > 1]

    def label_for(config):
        if not varying:
            return "default"
        return ", ".join(f"{a}={_format_value(config[a])}" for a in varying)

    baseline_label = label_for(baseline_config)
    runs = [(label_for(config), config, base_seed) for config in configs]
    # Exact inference has no Monte Carlo error, so reseeding would only repeat the same run.
    all_exact = inference in ("auto", "exact") and all(
        _exact.supported(_resolve_priors(c["priors"]), c["likelihood"], sample is not None) for c in configs
    )
    if not all_exact:
        runs += [
            (f"{baseline_label} (reseed {i})", baseline_config, base_seed + i) for i in range(1, max(1, int(n_seeds)))
        ]
    labels = [label for label, _, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Two runs got the same label, which would merge their results: {labels}")

    # Group by binning signature so each distinct binning is computed once.
    grouped = {}
    for run in runs:
        grouped.setdefault(tuple(_format_value(run[1][p]) for p in _BIN_PARAMS), []).append(run)

    iterator = tqdm(total=len(runs), desc="Checking DiffSVG stability") if progressbar else None
    columns = {}
    for group in grouped.values():
        bin_config = group[0][1]
        binned = bin_obs(
            adata,
            anchor,
            condition,
            genes=gene_list,
            design_matrix=design_matrix,
            inplace=False,
            sample=sample,
            **{p: bin_config[p] for p in _BIN_PARAMS},
        )
        for label, config, seed in group:
            model_setup = DiffSVGModelSetup(binned=binned, likelihood=config["likelihood"], priors=config["priors"])
            stats_df, _, _ = fit_model(
                adata,
                model_setup,
                n_workers=n_workers,
                progressbar=False,
                inplace=False,
                use_binned=config["use_binned"],
                sample_kwargs={**sample_kwargs, "random_seed": seed},
                test=test,
                tau=tau,
                inference=inference,
            )
            columns[label] = stats_df[rank_by]
            if iterator is not None:
                iterator.update(1)
    if iterator is not None:
        iterator.close()

    stats = pd.DataFrame(columns).reindex(gene_list)
    ranks = stats.rank(ascending=not higher_is_stronger, method="average")

    similarity = ranks.corr(method="spearman")
    tops = {label: set(ranks[label].nsmallest(top_k).index) for label in ranks}
    top_overlap = pd.DataFrame(
        [[len(tops[a] & tops[b]) / top_k for b in ranks] for a in ranks], index=list(ranks), columns=list(ranks)
    )

    reseed_labels = [label for label, _, _ in runs if label.endswith(")") and " (reseed " in label]
    noise_floor = {
        "spearman": 1.0 if all_exact else (float(min(similarity.loc[baseline_label, reseed_labels])) if reseed_labels else float("nan")),
        "top_overlap": 1.0 if all_exact else (float(min(top_overlap.loc[baseline_label, reseed_labels])) if reseed_labels else float("nan")),
    }

    sensitivity_rows = []
    for label, config, _ in runs:
        if label == baseline_label or label in reseed_labels:
            continue
        changed = [a for a in varying if config[a] != baseline_config[a]]
        if len(changed) == 1:
            sensitivity_rows.append(
                {
                    "parameter": changed[0],
                    "value": _format_value(config[changed[0]]),
                    "run": label,
                    "spearman": float(similarity.loc[baseline_label, label]),
                    "top_overlap": float(top_overlap.loc[baseline_label, label]),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows, columns=["parameter", "value", "run", "spearman", "top_overlap"])

    config_table = pd.DataFrame(
        [{a: _format_value(config[a]) for a in axes} for _, config, _ in runs], index=[label for label, _, _ in runs]
    )
    config_table = config_table[~config_table.index.duplicated()]

    return StabilityResult(
        stats=stats,
        ranks=ranks,
        similarity=similarity,
        top_overlap=top_overlap,
        sensitivity=sensitivity,
        configs=config_table,
        noise_floor=noise_floor,
        baseline=baseline_label,
        rank_by=rank_by,
        top_k=top_k,
        genes=gene_list,
    )
