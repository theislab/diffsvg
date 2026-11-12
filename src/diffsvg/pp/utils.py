"""Helpers for turning per-cell expression into per-bin summaries used by the GAMs in ``diffsvg.tl.model``."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy.sparse import issparse


def _expression(adata, genes, layer):
    """Dense float ``(n_obs, n_genes)`` expression of `genes`, from ``adata.layers[layer]`` or ``adata.X``."""
    sub = adata[:, genes]
    X = sub.layers[layer] if layer is not None else sub.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    return np.asarray(X, dtype=float)


def _nearest_bin(values, n_bins):
    """Index of the nearest of the ``n_bins + 1`` grid centres for every value; ``-1`` where it is missing."""
    values = np.asarray(values, dtype=float)
    index = np.full(values.shape, -1, dtype=int)
    present = np.isfinite(values)
    index[present] = np.clip(np.rint(values[present] * n_bins).astype(int), 0, n_bins)
    return index


def bin_counts(
    adata,
    genes,
    anchor,
    category,
    design_matrix="design_matrix",
    n_bins=50,
    window_radius=None,
    layer=None,
    drop_empty_bins=True,
):
    """Average expression of one or more genes in windows along `anchor`, per `category` level.

    For a grid of `n_bins` + 1 centers on ``[0, 1]``, all cells within `window_radius` of a center are
    averaged, separately for every level of `category`. The grid does not depend on the gene, so all
    `genes` are binned at once.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Gene name(s) to bin.
    anchor
        Column in `design_matrix` holding the ordering coordinate, in ``[0, 1]``.
    category
        Column in `design_matrix` to stratify by (e.g. the condition), also the column name in the output.
        `None` bins along `anchor` only, as a single group.
    design_matrix
        ``adata.obsm`` key of a data frame holding `anchor` and `category`.
    n_bins
        Number of bins spanning ``[0, 1]``; the grid has `n_bins` + 1 centers.
    window_radius
        Half-width of the averaging window. Windows wider than ``0.5 / n_bins`` overlap. `None` (default) makes
        them contiguous: each cell counts once, in the bin of its nearest center.
    layer
        If given, bin ``adata.layers[layer]`` instead of ``adata.X``.
    drop_empty_bins
        Drop ``(category, bin)`` rows without cells (the same rows for every gene).

    Returns
    -------
    dict[str, pandas.DataFrame]
        One data frame per gene with columns ``[category, anchor,
        "window_start", "window_end", "n_obs", "avg_var"]``, plus ``"ss_var"``
        (the within-bin sum of squared deviations from ``avg_var``) when
        ``window_radius=None``.
    """
    design_matrix = adata.obsm[design_matrix]

    genes = [genes] if isinstance(genes, str) else list(genes)
    X = _expression(adata, genes, layer)

    value = design_matrix[anchor].values
    # `category=None` bins along `anchor` alone, as one group.
    cat = np.zeros(len(value)) if category is None else np.asarray(design_matrix[category].values)

    bin_centers = np.linspace(0, 1, n_bins + 1)
    categories = np.unique(cat[~pd.isna(cat)])  # cells without a category (e.g. a slide without anchors) are left out
    contiguous = window_radius is None
    nearest = _nearest_bin(value, n_bins) if contiguous else None

    meta_rows = []
    avg_rows = []
    ss_rows = []

    for c in categories:
        cat_mask = cat == c
        for i, center in enumerate(bin_centers):
            if contiguous:
                lower = max(0.0, center - 0.5 / n_bins)
                upper = min(1.0, center + 0.5 / n_bins)
                mask = cat_mask & (nearest == i)
            else:
                lower = max(0.0, center - window_radius)
                upper = min(1.0, center + window_radius)
                mask = cat_mask & (value >= lower) & (value <= upper)
            n_obs = int(mask.sum())
            avg = X[mask].mean(axis=0) if n_obs else np.full(X.shape[1], np.nan)

            meta_rows.append((c, center, lower, upper, n_obs))
            avg_rows.append(avg)
            if contiguous:
                ss_rows.append(((X[mask] - avg) ** 2).sum(axis=0) if n_obs else np.full(X.shape[1], np.nan))

    meta_cols = ["_group" if category is None else category, anchor, "window_start", "window_end", "n_obs"]
    meta = pd.DataFrame(meta_rows, columns=meta_cols)
    avg = np.vstack(avg_rows)  # (n_meta_rows, n_genes)
    ss = np.vstack(ss_rows) if contiguous else None

    if drop_empty_bins:
        keep = meta["n_obs"].to_numpy() > 0
        meta = meta.loc[keep].reset_index(drop=True)
        avg = avg[keep]
        ss = ss[keep] if contiguous else None

    if contiguous:
        return {gene: meta.assign(avg_var=avg[:, i], ss_var=ss[:, i]) for i, gene in enumerate(genes)}
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
        f"center(bs(x, knots=interior_knots, degree={degree}, include_intercept=True, lower_bound=0, upper_bound=1)) - 1",
        {"x": np.asarray(x), "interior_knots": np.asarray(interior_knots)},
    )
    return np.asarray(dm), dm.design_info


def _bspline_basis_01(x, n_bsplines, degree):
    """Uncentred B-spline basis with `bin_obs`'s knots, boundary knots pinned at ``0``/``1``.

    `build_spline_basis` pins the boundary knots at ``0`` and ``1`` too, so this reproduces
    `bin_obs`'s basis for any `x`.
    """
    interior_knots = compute_interior_knots(n_bsplines, degree=degree)[1:-1]
    dm = patsy.dmatrix(
        f"bs(x, knots=interior_knots, degree={degree}, include_intercept=True, lower_bound=0, upper_bound=1) - 1",
        {"x": np.asarray(x, dtype=float), "interior_knots": interior_knots},
    )
    return np.asarray(dm)


def spline_basis_at(binned, x):
    """Evaluate the spline basis of `binned` at new `anchor` values.

    Same knots, boundary knots and centring as ``binned.B``/``binned.B_pred``,
    so coefficients fitted on this basis are valid on the prediction grid and
    vice versa. Used to fit on single cells (``use_binned=False``).

    Parameters
    ----------
    binned
        A `BinnedData`, i.e. the output of `bin_obs`.
    x
        Anchor values in ``[0, 1]``, without missing values.

    Returns
    -------
    numpy.ndarray
        Design matrix of shape ``(len(x), binned.n_bsplines)``.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("`x` holds missing or infinite anchor values - drop those cells first.")
    if ((x < 0) | (x > 1)).any():
        raise ValueError(f"anchor values must lie in [0, 1], got range [{x.min()}, {x.max()}].")

    # `binned.B_pred` is the centred basis on the grid, so the centring `bin_obs`
    # applied is the (row-constant) gap between the uncentred and centred grid basis.
    offsets = _bspline_basis_01(binned.bin_centers, binned.n_bsplines, binned.degree) - np.asarray(binned.B_pred)
    centre = offsets.mean(axis=0)
    if not np.allclose(offsets, centre, atol=1e-8):
        raise ValueError("could not reproduce the spline basis of `binned` - was it built by `bin_obs`?")
    return _bspline_basis_01(x, binned.n_bsplines, binned.degree) - centre


@dataclass
class BinnedData:
    """Output of `bin_obs`, consumed by `diffsvg.tl.setup_model`.

    Everything a per-gene GAM needs that does not depend on the gene, plus the binned targets `Y`.

    Attributes
    ----------
    genes
        Gene order matching `Y`/`intercept_initvals`.
    Y
        Per-gene binned expression.
    intercept_initvals
        Per-gene initial value of `intercept` (mean binned expression at ``anchor == 0``).
    anchor, condition
        Columns of the design matrix the data was binned on. `condition` is `None` when binned without
        one; every condition-dependent field is then `None` too.
    condition_categories
        ``{"0": <label>, "1": <label>}``, how `condition` was encoded (string keys for h5ad).
    n_bsplines, degree, n_bins
        Binning/basis hyperparameters.
    B, B_masked, condition_fit
        Spline design matrices and encoded condition of the fitted rows.
    bin_centers
        Grid (`n_bins` + 1 points on ``[0, 1]``), also the prediction grid.
    bin_assignment
        Per cell, the index of its nearest bin center, ``-1`` without an anchor value.
    design_matrix, layer
        Where the unbinned data lives, for ``use_binned=False``.
    B_pred, B_pred_masked_0, B_pred_masked_1, condition_pred_0, condition_pred_1
        Design matrices for the prediction grid.
    sample, sample_labels, sample_index
        Only when binned with `sample`: the column, the samples that have rows, and per row its sample's
        index. Rows are then one per (sample, bin).
    n_obs
        Cells per row. The fitters weight rows by it.
    n_obs_grid, ss_within
        Only for contiguous bins (``window_radius=None``): cells per grid point
        (``n_categories x (n_bins + 1)``) and per gene the within-bin sum of squares, so a binned fit
        matches a fit on the cells.
    """

    genes: list
    Y: dict
    intercept_initvals: dict
    anchor: str
    condition: str | None
    condition_categories: dict | None
    n_bsplines: int
    degree: int
    n_bins: int
    B: np.ndarray
    B_masked: np.ndarray | None
    condition_fit: np.ndarray | None
    bin_centers: np.ndarray
    bin_assignment: np.ndarray
    B_pred: np.ndarray
    B_pred_masked_0: np.ndarray | None
    B_pred_masked_1: np.ndarray | None
    condition_pred_0: np.ndarray | None
    condition_pred_1: np.ndarray | None
    # Defaulted so that a `binned` dict written before these existed still reconstructs.
    design_matrix: str = "design_matrix"
    layer: str | None = None
    n_obs: np.ndarray | None = None
    n_obs_grid: np.ndarray | None = None
    ss_within: dict | None = None
    sample: str | None = None
    sample_labels: list | None = None
    sample_index: np.ndarray | None = None

    @property
    def n_samples(self):
        """Number of samples that have rows, or `None` when binned without `sample`."""
        return None if self.sample_labels is None else len(self.sample_labels)

    @property
    def has_condition(self):
        """Whether this was binned with a `condition`, i.e. whether the two-condition models can be fit."""
        return self.condition is not None

    @property
    def weighted(self):
        """Whether the bins are contiguous with the within-bin spread recorded, i.e. whether a binned fit
        carries the same information as a fit on the cells. (Overlapping bins record `n_obs` too, and fits
        weight rows by it, but their rows share cells.)"""
        return self.ss_within is not None


def _condition_levels(values):
    """The levels of a condition column, in the order that fixes the 0/1 coding.

    A categorical column keeps its category order (only categories that occur), anything else is sorted
    as strings. Missing values are left out, as are the cells that hold them.
    """
    values = pd.Series(values)
    if isinstance(values.dtype, pd.CategoricalDtype):
        observed = set(values.dropna().unique())
        return [c for c in values.cat.categories if c in observed]
    return sorted(pd.unique(values.dropna()), key=str)


def bin_obs(
    adata,
    anchor,
    condition,
    genes=None,
    n_bsplines=10,
    degree=3,
    n_bins=50,
    window_radius=None,
    layer=None,
    design_matrix="design_matrix",
    inplace=True,
    sample=None,
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
        compare (e.g. control vs. treatment). Pass `None` for data without
        conditions: binning then runs along `anchor` only, and the result
        supports the condition-free models used by
        `diffsvg.tl.fit_profiles`/`diffsvg.tl.one_vs_rest`.
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
        *assigned* to one). Overlapping windows make neighbouring bins share
        cells, which the models treat as independent, so their uncertainty
        comes out too small. `None` bins every cell once, in its assigned
        bin, and records the cell counts and within-bin spread; the fitters
        then weight each bin by its cell count, so a binned fit matches a fit
        on the cells (needed for calibrated ``test="parallel"`` p-values).
        `None` is the default. With overlapping windows the rows are still
        weighted by how many cells they average (so a sparse edge window does
        not count as much as a full one), but neighbouring rows stay
        correlated and the p-values too small.
    layer
        If given, use ``adata.layers[layer]`` instead of ``adata.X``.
    design_matrix
        ``adata.obsm`` key of the design matrix. Defaults to
        ``"design_matrix"``.
    sample
        Column in ``adata.obsm["design_matrix"]`` naming the slide (replicate) of every cell, for data with
        several slides per `condition`. Each slide must belong to one `condition`. Bins are then built per slide
        (one row per slide and bin, contiguous: needs ``window_radius=None``), and the fitters add a random
        level and a random curve per slide, so that the condition curves are compared against the variation
        between slides rather than between cells (see ``diffsvg.tl.fit_model``).
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
    for col in (anchor, condition, sample):
        if col is not None and col not in design_matrix_df.columns:
            raise KeyError(f"{col!r} not found in adata.obsm[{design_matrix!r}]")
    if sample is not None:
        if condition is None:
            raise ValueError("`sample` needs a `condition`: it exists to compare conditions across replicates.")
        if window_radius is not None:
            raise ValueError("`sample` needs contiguous bins - pass `window_radius=None`.")

    genes = list(adata.var_names) if genes is None else list(genes)
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise KeyError(f"Genes not found in adata.var_names: {sorted(missing)}")

    anchor_values = design_matrix_df[anchor].to_numpy(dtype=float)
    present = anchor_values[np.isfinite(anchor_values)]
    if present.size and (present.min() < 0 or present.max() > 1):
        raise ValueError(
            f"`anchor` values must lie in [0, 1] (missing values are allowed and left out); "
            f"got range [{present.min()}, {present.max()}]. Normalise the distances first, e.g. with "
            "`squidpy.tl.var_by_distance`."
        )

    levels = None
    if condition is not None:
        levels = _condition_levels(design_matrix_df[condition])
        if len(levels) != 2:
            raise ValueError(f"`condition` column {condition!r} must have exactly two levels, found: {levels}")

    binned = bin_counts(
        adata,
        genes,
        anchor,
        condition if sample is None else sample,
        design_matrix=design_matrix,
        n_bins=n_bins,
        window_radius=window_radius,
        layer=layer,
    )

    sample_labels = sample_index = None
    if sample is not None:
        # Rows are (sample, bin); each sample's rows get its condition.
        pairs = design_matrix_df[[sample, condition]].dropna().drop_duplicates()
        pairs[sample] = pairs[sample].astype(str)
        mixed = pairs[sample][pairs[sample].duplicated()].unique()
        if len(mixed):
            raise ValueError(f"`sample` must be nested in `condition`; these samples have several: {sorted(mixed)}")
        sample_condition = dict(zip(pairs[sample], pairs[condition], strict=True))
        sample_labels = sorted({str(v) for v in binned[genes[0]][sample]})
        position = {label: i for i, label in enumerate(sample_labels)}
        sample_index = binned[genes[0]][sample].astype(str).map(position).to_numpy(dtype=int)
        for gene in genes:
            frame = binned[gene]
            frame[condition] = frame[sample].astype(str).map(sample_condition)

    # Encode `condition` to 0.0/1.0 so it can be used directly as a numeric
    # covariate; keep the original labels around for interpretability.
    if condition is not None:
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
    condition_fit = None if condition is None else template[condition].to_numpy(dtype=float)
    B, design_info = build_spline_basis(anchor_vals, interior_knots, degree=degree)
    B_masked = None if condition is None else B * condition_fit[:, None]
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
    condition_pred_0 = None if condition is None else np.zeros(n_bins + 1)
    condition_pred_1 = None if condition is None else np.ones(n_bins + 1)
    B_pred_masked_0 = None if condition is None else B_pred * condition_pred_0[:, None]
    B_pred_masked_1 = None if condition is None else B_pred * condition_pred_1[:, None]

    # Every cell (not just those in `genes`) gets assigned to its nearest
    # bin center - `bin_centers` is an evenly spaced grid, so this is just
    # rounding to the nearest multiple of 1/n_bins. Cells without an anchor
    # value get -1.
    bin_assignment = _nearest_bin(design_matrix_df[anchor].to_numpy(dtype=float), n_bins)

    n_obs = template["n_obs"].to_numpy(dtype=float)
    n_obs_grid = ss_within = None
    if window_radius is None:
        codes = np.zeros(len(template), dtype=int) if condition is None else condition_fit.astype(int)
        n_obs_grid = np.zeros((1 if condition is None else 2, n_bins + 1))
        np.add.at(n_obs_grid, (codes, np.rint(anchor_vals * n_bins).astype(int)), n_obs)
        ss_within = {gene: float(binned[gene]["ss_var"].sum()) for gene in genes}

    binned_data = BinnedData(
        genes=genes,
        Y=Y,
        intercept_initvals=intercept_initvals,
        anchor=anchor,
        condition=condition,
        condition_categories=None if condition is None else {"0": levels[0], "1": levels[1]},
        n_bsplines=n_bsplines,
        degree=degree,
        n_bins=n_bins,
        B=B,
        B_masked=B_masked,
        condition_fit=condition_fit,
        bin_centers=bin_centers,
        bin_assignment=bin_assignment,
        design_matrix=design_matrix,
        layer=layer,
        B_pred=B_pred,
        B_pred_masked_0=B_pred_masked_0,
        B_pred_masked_1=B_pred_masked_1,
        condition_pred_0=condition_pred_0,
        condition_pred_1=condition_pred_1,
        n_obs=n_obs,
        n_obs_grid=n_obs_grid,
        ss_within=ss_within,
        sample=sample,
        sample_labels=sample_labels,
        sample_index=sample_index,
    )

    if not inplace:
        return binned_data

    adata.obs[f"{anchor}_bin"] = bin_assignment
    adata.uns["binned"] = vars(binned_data).copy()
    return None
