import anndata as ad
import numpy as np
import pandas as pd
import pytest

import diffsvg as dvg
from diffsvg.tl.model import _build_gene_model, _predictive_seeds, _weighting

N_BINS = 20


def _adata(x, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    cond = np.where(np.arange(n) % 2 == 0, "a", "b")
    X = (np.sin(3 * np.nan_to_num(x))[:, None] + rng.normal(size=(n, 2))).astype(np.float32)
    adata = ad.AnnData(X, var=pd.DataFrame(index=["g0", "g1"]))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


def _positions(n, seed=0):
    x = np.random.default_rng(seed).random(n)
    x[[0, 1, 2, 3]] = 0.0, 0.0, 1.0, 1.0  # both conditions reach both ends
    return x


def test_contiguous_bins_count_every_cell_once():
    x = _positions(4000)
    x[10:30] = np.nan  # e.g. the anchor cells themselves
    adata = _adata(x)
    b = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, inplace=False)
    present = np.isfinite(x)
    assert b.weighted
    assert b.n_obs.sum() == present.sum()
    np.testing.assert_array_equal(b.bin_assignment[~present], -1)

    # every row is the mean of exactly the cells assigned to its bin, for its condition
    X = np.asarray(adata.X, dtype=float)
    cond = adata.obsm["design_matrix"]["condition"].to_numpy()
    ss_total = 0.0
    for code, label in b.condition_categories.items():
        for i in range(N_BINS + 1):
            cells = (cond == label) & (b.bin_assignment == i)
            assert b.n_obs_grid[int(code), i] == cells.sum()
            if cells.any():
                ss_total += ((X[cells, 0] - X[cells, 0].mean()) ** 2).sum()
    assert b.ss_within["g0"] == pytest.approx(ss_total)


def test_sliding_bins_stay_unweighted():
    b = dvg.pp.bin_obs(_adata(_positions(2000)), anchor="anchor", condition="condition", inplace=False)
    assert not b.weighted
    assert b.n_obs is None and b.n_obs_grid is None and b.ss_within is None
    assert _weighting(b, use_binned=True) == (None, None, None)


def test_weighted_bins_have_the_per_cell_likelihood():
    """With every cell on its bin centre, binned and per-cell log densities differ only by a constant."""
    centers = np.linspace(0, 1, N_BINS + 1)
    x = np.random.default_rng(1).choice(centers, size=3000)
    x[:4] = 0.0, 0.0, 1.0, 1.0
    adata = _adata(x, seed=2)
    b = dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, inplace=False)
    weights, ss_within, _ = _weighting(b, use_binned=True)
    binned_model = _build_gene_model(
        b.Y["g0"], b.B, b.B_masked, b.condition_fit, b.n_bsplines, 0.0, weights=weights, ss_within=ss_within["g0"]
    )
    c = (adata.obsm["design_matrix"]["condition"] == b.condition_categories["1"]).to_numpy(dtype=float)
    B_cells = dvg.pp.spline_basis_at(b, x)
    y = np.asarray(adata.X[:, 0], dtype=float)
    cell_model = _build_gene_model(y, B_cells, B_cells * c[:, None], c, b.n_bsplines, 0.0)

    rng = np.random.default_rng(3)
    start = binned_model.initial_point()
    points = [{k: v + rng.normal(scale=0.3, size=np.shape(v)) for k, v in start.items()} for _ in range(4)]
    logp_binned, logp_cells = binned_model.compile_logp(), cell_model.compile_logp()
    gaps = [logp_binned(p) - logp_cells(p) for p in points]
    np.testing.assert_allclose(gaps, gaps[0], rtol=0, atol=1e-6 * abs(logp_cells(points[0])))


def test_predictive_seeds_are_reproducible_and_distinct():
    seeds = _predictive_seeds(42, n_genes=5, per_gene=2)
    assert seeds == _predictive_seeds(42, n_genes=5, per_gene=2)
    flat = [s for pair in seeds for s in pair]
    assert len(set(flat)) == len(flat)  # differs between genes and between the two condition levels
    assert _predictive_seeds(None, n_genes=3, per_gene=2) == [(None, None)] * 3
