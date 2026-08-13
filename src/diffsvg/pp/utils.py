"""Helpers for turning per-cell expression into per-bin summaries used by the GAMs in ``diffsvg.tl.model``."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy.sparse import issparse


def bin_counts(
    adata,
    genes,
    anchor,
    category,
    design_matrix="design_matrix",
    n_bins=50,
    window_radius=0.02,
    layer=None,
    drop_empty_bins=True,
):
    """Average expression of one or more genes in sliding windows along `anchor`, per `category` level.

    The `anchor` coordinate is assumed to live on ``[0, 1]`` (e.g. a normalized
    pseudotime or spatial distance stored in ``adata.obsm["design_matrix"]``).
    For a fixed grid of `n_bins` + 1 centers, all cells within `window_radius`
    of a center are averaged, separately for every level of `category`. Because
    the grid, the windows, and the category assignment do not depend on the
    gene, this is computed once for all requested `genes` at the same time
    rather than looping per gene.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Gene name(s) (subset of ``adata.var_names``) to bin.
    anchor
        Column in `design_matrix` holding the ordering coordinate, expected to
        be in ``[0, 1]``.
    category
        Column in `design_matrix` used to stratify the binning (e.g. a
        condition indicator). Also used as the column name for the category
        value in the returned per-gene data frames.
    design_matrix
        ``adata.obsm`` key of a data frame aligned to ``adata.obs_names``
        holding at least `anchor` and `category`. Defaults to
        ``"design_matrix"``, i.e. ``adata.obsm["design_matrix"]``.
    n_bins
        Number of bins spanning ``[0, 1]``; the grid has `n_bins` + 1 centers
        (``np.linspace(0, 1, n_bins + 1)``).
    window_radius
        Half-width of the averaging window around each bin center.
    layer
        If given, bin ``adata.layers[layer]`` instead of ``adata.X``.
    drop_empty_bins
        Drop ``(category, bin)`` combinations with zero observations. Since
        emptiness only depends on `anchor`/`category`, it is shared by all
        genes, so it is safe to drop these rows once for every gene.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One data frame per gene with columns ``[category, anchor,
        "window_start", "window_end", "n_obs", "avg_var"]``.
    """
    design_matrix = adata.obsm[design_matrix]

    genes = [genes] if isinstance(genes, str) else list(genes)

    sub = adata[:, genes]
    X = sub.layers[layer] if layer is not None else sub.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    X = np.asarray(X, dtype=float)  # (n_obs, n_genes)

    value = design_matrix[anchor].values
    cat = design_matrix[category].values

    bin_centers = np.linspace(0, 1, n_bins + 1)
    categories = np.unique(cat)

    meta_rows = []
    avg_rows = []

    for c in categories:
        cat_mask = cat == c
        for center in bin_centers:
            lower = max(0.0, center - window_radius)
            upper = min(1.0, center + window_radius)
            mask = cat_mask & (value >= lower) & (value <= upper)
            n_obs = int(mask.sum())
            avg = X[mask].mean(axis=0) if n_obs else np.full(X.shape[1], np.nan)

            meta_rows.append((c, center, lower, upper, n_obs))
            avg_rows.append(avg)

    meta = pd.DataFrame(meta_rows, columns=[category, anchor, "window_start", "window_end", "n_obs"])
    avg = np.vstack(avg_rows)  # (n_meta_rows, n_genes)

    if drop_empty_bins:
        keep = meta["n_obs"].to_numpy() > 0
        meta = meta.loc[keep].reset_index(drop=True)
        avg = avg[keep]

    return {gene: meta.assign(avg_var=avg[:, i]) for i, gene in enumerate(genes)}


def compute_interior_knots(n_bsplines, degree=3):
    """Interior knot positions (plus the ``0``/``1`` endpoints) for a B-spline basis on ``[0, 1]``.

    For `patsy`'s ``bs(x, knots=interior_knots, degree=degree, include_intercept=True)``
    the number of resulting basis columns is ``len(interior_knots) + degree +
    1``, so `n_bsplines` interior knots are chosen accordingly.

    Parameters
    ----------
    n_bsplines
        Desired number of B-spline basis functions.
    degree
        Degree of the B-spline (3 = cubic).

    Returns
    -------
    numpy.ndarray
        Knot grid including the ``0`` and ``1`` endpoints, i.e. the interior
        knots to pass to `patsy` are ``result[1:-1]``.
    """
    n_interior = n_bsplines - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_bsplines must be >= degree + 1 ({degree + 1}); got {n_bsplines}")
    return np.linspace(0, 1, n_interior + 2)


def build_spline_basis(x, interior_knots, degree=3):
    """Build a (centered, intercept-free) B-spline design matrix for `x`.

    Parameters
    ----------
    x
        1D array of coordinates in ``[0, 1]``.
    interior_knots
        Interior knots, e.g. ``compute_interior_knots(n_bsplines, degree)[1:-1]``.
    degree
        Degree of the B-spline (3 = cubic).

    Returns
    -------
    B : numpy.ndarray
        Design matrix of shape ``(len(x), len(interior_knots) + degree + 1)``.
    design_info : patsy.design_info.DesignInfo
        Design info, needed to evaluate the same basis at new `x` values via
        ``patsy.build_design_matrices([design_info], {"x": new_x})``.
    """
    dm = patsy.dmatrix(
        f"center(bs(x, knots=interior_knots, degree={degree}, include_intercept=True)) - 1",
        {"x": np.asarray(x), "interior_knots": np.asarray(interior_knots)},
    )
    return np.asarray(dm), dm.design_info


@dataclass
class BinnedData:
    """Output of `bin_obs`, consumed by `diffsvg.tl.setup_model`.

    Holds everything needed to build a per-gene GAM that does *not* depend
    on the gene (the spline design matrices, the encoded `condition`
    column, the prediction grid, ...), plus the one thing that does: `Y`.

    Attributes
    ----------
    genes
        Gene order matching `Y`/`intercept_initvals`.
    Y
        Per-gene binned expression (target `Y` of the GAM).
    intercept_initvals
        Per-gene initial value for the GAM's `intercept` parameter (mean
        binned expression at ``anchor == 0``).
    anchor, condition
        Column names in ``adata.obsm["design_matrix"]`` the data was binned
        on.
    condition_categories
        Mapping ``{"0": <original label>, "1": <original label>}`` recording
        how `condition` was encoded to ``0.0``/``1.0`` (string keys so this
        survives round-tripping through ``adata.uns`` / h5ad, whose `dict`
        writer requires string keys).
    n_bsplines, degree, n_bins
        Binning/basis hyperparameters used by `bin_obs`.
    B, B_masked, condition_fit
        Spline design matrices / encoded condition at fit time (shared
        across all genes - only `Y` differs per gene).
    bin_centers
        Grid (`n_bins` + 1 points on ``[0, 1]``), also used as the posterior
        predictive prediction grid.
    bin_assignment
        Per-cell (i.e. per ``adata.obs_names``) index into `bin_centers` -
        which bin each cell was assigned to, by nearest `anchor` value. Also
        written to ``adata.obs[f"{anchor}_bin"]`` by `bin_obs`.
    B_pred, B_pred_masked_0, B_pred_masked_1, condition_pred_0, condition_pred_1
        Precomputed design matrices for the prediction grid.
    """

    genes: list
    Y: dict
    intercept_initvals: dict
    anchor: str
    condition: str
    condition_categories: dict
    n_bsplines: int
    degree: int
    n_bins: int
    B: np.ndarray
    B_masked: np.ndarray
    condition_fit: np.ndarray
    bin_centers: np.ndarray
    bin_assignment: np.ndarray
    B_pred: np.ndarray
    B_pred_masked_0: np.ndarray
    B_pred_masked_1: np.ndarray
    condition_pred_0: np.ndarray
    condition_pred_1: np.ndarray


def bin_obs(
    adata,
    anchor,
    condition,
    genes=None,
    n_bsplines=10,
    degree=3,
    n_bins=50,
    window_radius=0.02,
    layer=None,
    design_matrix="design_matrix",
    inplace=True,
):
    """Assign each cell a bin along `anchor`, bin every gene's expression, and precompute the spline design matrices `diffsvg.tl.setup_model` needs.

    This is the preprocessing step: it only bins/reshapes data (via
    `bin_counts`) and builds the (gene-independent) B-spline design
    matrices. It does not touch PyMC at all - `diffsvg.tl.setup_model` reads
    its output back off `adata` and defines the actual model.

    Parameters
    ----------
    adata
        Annotated data matrix. ``adata.obsm["design_matrix"]`` must contain
        `anchor` (assumed to lie in ``[0, 1]``) and `condition` (must have
        exactly two levels).
    anchor
        Column in ``adata.obsm["design_matrix"]`` giving the ordering
        coordinate (e.g. pseudotime, spatial distance) that the spline basis
        is built on.
    condition
        Column in ``adata.obsm["design_matrix"]`` giving the two groups to
        compare (e.g. control vs. treatment).
    genes
        Genes to model. Defaults to all of ``adata.var_names``.
    n_bsplines
        Number of B-spline basis functions used for both the `anchor` effect
        and its `condition`-interaction.
    degree
        Degree of the B-spline basis (3 = cubic).
    n_bins
        Number of bins spanning `anchor` in ``[0, 1]``; the fitting/prediction
        grid has `n_bins` + 1 points. Every cell is also assigned to its
        nearest of these `n_bins` + 1 bin centers (see `bin_assignment`
        below).
    window_radius
        Half-width of the averaging window used by `bin_counts` (independent
        of, and typically wider than, half a bin - a cell can contribute to
        several neighboring bins' averages even though it is only ever
        *assigned* to one).
    layer
        If given, use ``adata.layers[layer]`` instead of ``adata.X``.
    design_matrix
        ``adata.obsm`` key of the design matrix. Defaults to
        ``"design_matrix"``.
    inplace
        If `True` (default), write ``adata.obs[f"{anchor}_bin"]`` (each
        cell's assigned bin index) and ``adata.uns["binned"]`` (everything
        `diffsvg.tl.setup_model` needs) and return `None`. If `False`, leave
        `adata` untouched and return the `BinnedData` instead.

    Returns
    -------
    None or BinnedData
    """
    design_matrix_df = adata.obsm[design_matrix]
    for col in (anchor, condition):
        if col not in design_matrix_df.columns:
            raise KeyError(f"{col!r} not found in adata.obsm[{design_matrix!r}]")

    genes = list(adata.var_names) if genes is None else list(genes)
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise KeyError(f"Genes not found in adata.var_names: {sorted(missing)}")

    levels = sorted(pd.unique(design_matrix_df[condition]), key=str)
    if len(levels) != 2:
        raise ValueError(f"`condition` column {condition!r} must have exactly two levels, found: {levels}")

    binned = bin_counts(
        adata,
        genes,
        anchor,
        condition,
        design_matrix=design_matrix,
        n_bins=n_bins,
        window_radius=window_radius,
        layer=layer,
    )

    # Encode `condition` to 0.0/1.0 so it can be used directly as a numeric
    # covariate; keep the original labels around for interpretability.
    code_map = {levels[0]: 0.0, levels[1]: 1.0}
    for gene in genes:
        binned[gene][condition] = binned[gene][condition].map(code_map).astype(float)

    knots_full = compute_interior_knots(n_bsplines, degree=degree)
    interior_knots = knots_full[1:-1]

    # `anchor`/`condition` are identical across genes (see bin_counts), so
    # the spline basis is built once and shared by every gene's model; only
    # `Y` (binned expression) differs per gene.
    template = binned[genes[0]]
    anchor_vals = template[anchor].to_numpy()
    condition_fit = template[condition].to_numpy(dtype=float)
    B, design_info = build_spline_basis(anchor_vals, interior_knots, degree=degree)
    B_masked = B * condition_fit[:, None]
    zero_mask = anchor_vals == 0.0

    Y = {}
    intercept_initvals = {}
    for gene in genes:
        y = binned[gene]["avg_var"].to_numpy()
        Y[gene] = y
        baseline = y[zero_mask]
        intercept_initvals[gene] = float(baseline.mean()) if baseline.size else float(y.mean())

    # Fixed grid used for posterior-predictive prediction, evaluated at both
    # condition levels.
    bin_centers = np.linspace(0, 1, n_bins + 1)
    B_pred = np.asarray(patsy.build_design_matrices([design_info], {"x": bin_centers})[0])
    condition_pred_0 = np.zeros(n_bins + 1)
    condition_pred_1 = np.ones(n_bins + 1)
    B_pred_masked_0 = B_pred * condition_pred_0[:, None]
    B_pred_masked_1 = B_pred * condition_pred_1[:, None]

    # Every cell (not just those in `genes`) gets assigned to its nearest
    # bin center - `bin_centers` is an evenly spaced grid, so this is just
    # rounding to the nearest multiple of 1/n_bins.
    all_anchor_vals = design_matrix_df[anchor].to_numpy(dtype=float)
    bin_assignment = np.clip(np.rint(all_anchor_vals * n_bins).astype(int), 0, n_bins)

    binned_data = BinnedData(
        genes=genes,
        Y=Y,
        intercept_initvals=intercept_initvals,
        anchor=anchor,
        condition=condition,
        condition_categories={"0": levels[0], "1": levels[1]},
        n_bsplines=n_bsplines,
        degree=degree,
        n_bins=n_bins,
        B=B,
        B_masked=B_masked,
        condition_fit=condition_fit,
        bin_centers=bin_centers,
        bin_assignment=bin_assignment,
        B_pred=B_pred,
        B_pred_masked_0=B_pred_masked_0,
        B_pred_masked_1=B_pred_masked_1,
        condition_pred_0=condition_pred_0,
        condition_pred_1=condition_pred_1,
    )

    if not inplace:
        return binned_data

    adata.obs[f"{anchor}_bin"] = bin_assignment
    adata.uns["binned"] = vars(binned_data).copy()
    return None
