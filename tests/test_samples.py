import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diffsvg as dvg
from diffsvg.tl import model as M

K = 10
N_BINS = 20


def _slides(n_per_slide=600, n_a=5, n_b=5, level_sd=0.3, curve_sd=0.3, shape_effect=0.0, seed=0):
    """Slides with their own level and smooth curve around shared condition curves; `shape_effect` tilts condition b."""
    rng = np.random.default_rng(seed)
    n_slides = n_a + n_b
    xs, slide, cond = [], [], []
    for j in range(n_slides):
        x = rng.random(n_per_slide)
        x[:2] = 0.0, 1.0
        xs.append(x)
        slide.append(np.full(n_per_slide, j))
        cond.append(np.full(n_per_slide, "a" if j < n_a else "b"))
    x, slide, cond = np.concatenate(xs), np.concatenate(slide), np.concatenate(cond)
    level = rng.normal(0, level_sd, n_slides)
    curve = rng.normal(0, curve_sd, (n_slides, 3))
    y = (
        np.sin(3 * x)
        + level[slide]
        + sum(curve[slide, k] * np.cos((k + 1) * np.pi * x) for k in range(3))
        + shape_effect * (cond == "b") * (x - 0.5)
        + rng.normal(0, 0.5, len(x))
    )
    adata = ad.AnnData(y[:, None].astype(np.float32), var=pd.DataFrame(index=["g"]))
    labels = np.array([f"s{j}" for j in slide])
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond, "slide": labels}, index=adata.obs_names)
    return adata


def test_binning_by_sample_gives_one_row_per_slide_and_bin():
    adata = _slides()
    b = dvg.pp.bin_obs(
        adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide", inplace=False
    )
    assert b.n_samples == 10 and b.sample_labels == sorted(f"s{j}" for j in range(10))
    assert len(b.Y["g"]) == len(b.sample_index) == len(b.n_obs)
    assert b.n_obs.sum() == adata.n_obs
    # a slide's rows carry its condition, and the grid counts add the slides of a condition up
    cond_of_row = np.asarray(b.condition_fit)
    assert all(len(set(cond_of_row[b.sample_index == j])) == 1 for j in range(10))
    np.testing.assert_allclose(b.n_obs_grid.sum(axis=0).sum(), adata.n_obs)
    assert b.n_obs_grid[0].sum() == 3000 and b.n_obs_grid[1].sum() == 3000


def test_sample_must_be_nested_in_condition_and_needs_contiguous_bins():
    adata = _slides()
    dm = adata.obsm["design_matrix"]
    with pytest.raises(ValueError, match="window_radius"):
        dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", sample="slide", window_radius=0.02, inplace=False)
    with pytest.raises(ValueError, match="needs a `condition`"):
        dvg.pp.bin_obs(adata, anchor="anchor", condition=None, sample="slide", window_radius=None, inplace=False)
    dm.loc[dm["slide"] == "s0", "condition"] = "b"  # move slide s0 to condition b ...
    dm.loc[dm.index[:10], "condition"] = "a"  # ... except for ten cells, so it now has two conditions
    with pytest.raises(ValueError, match="nested"):
        dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", sample="slide", window_radius=None, inplace=False)


def test_binned_and_per_cell_models_with_slides_have_the_same_density():
    """With every cell on its bin centre, rows and cells differ only by a constant, slide terms included."""
    centers = np.linspace(0, 1, N_BINS + 1)
    adata = _slides(n_per_slide=400)
    adata.obsm["design_matrix"]["anchor"] = np.random.default_rng(1).choice(centers, adata.n_obs)
    dm = adata.obsm["design_matrix"]
    dm.loc[dm.index[:4], "anchor"] = 0.0, 0.0, 1.0, 1.0
    b = dvg.pp.bin_obs(
        adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide", inplace=False
    )
    weights, ss_within, _ = M._weighting(b, use_binned=True)
    rows = M._build_gene_model(
        b.Y["g"], b.B, b.B_masked, b.condition_fit, K, 0.0, weights=weights, ss_within=ss_within["g"],
        sample_index=b.sample_index, n_samples=b.n_samples,
    )  # fmt: skip
    keep = M._fitted_cells(dm, b)
    c = dm["condition"].map({"a": 0.0, "b": 1.0}).to_numpy()[keep]
    Bc = dvg.pp.spline_basis_at(b, dm["anchor"].to_numpy()[keep])
    index = dm["slide"].map({s: i for i, s in enumerate(b.sample_labels)}).to_numpy()[keep]
    cells = M._build_gene_model(
        np.asarray(adata.X[:, 0], dtype=float)[keep], Bc, Bc * c[:, None], c, K, 0.0, sample_index=index, n_samples=b.n_samples
    )  # fmt: skip
    rng = np.random.default_rng(3)
    start = rows.initial_point()
    points = [{k: v + rng.normal(scale=0.3, size=np.shape(v)) for k, v in start.items()} for _ in range(4)]
    f_rows, f_cells = rows.compile_logp(), cells.compile_logp()
    gaps = [f_rows(p) - f_cells(p) for p in points]
    np.testing.assert_allclose(gaps, gaps[0], rtol=0, atol=1e-6 * abs(f_cells(points[0])))


def test_f_reference_is_wider_in_the_tail_and_converges_to_chi_square():
    Q = dvg.tl.shape_metric(K, 3)
    cov = np.eye(K) * 0.01
    mean = np.random.default_rng(2).normal(size=K) * 0.3  # a clearly significant difference
    chi = dvg.tl.parallel_curves_test(mean, Q, tau=0, cov=cov)
    few = dvg.tl.parallel_curves_test(mean, Q, tau=0, cov=cov, df_denominator=8)
    many = dvg.tl.parallel_curves_test(mean, Q, tau=0, cov=cov, df_denominator=10**6)
    assert few[3] == pytest.approx(stats.f.sf(chi[1] / chi[2], chi[2], 8))
    assert few[3] > chi[3]  # few slides: a given statistic is less convincing
    assert many[3] == pytest.approx(chi[3], rel=1e-3)
    tau_few = dvg.tl.parallel_curves_test(mean, Q, tau=0.1, cov=cov, df_denominator=8)[3]
    tau_many = dvg.tl.parallel_curves_test(mean, Q, tau=0.1, cov=cov, df_denominator=10**6)[3]
    tau_chi = dvg.tl.parallel_curves_test(mean, Q, tau=0.1, cov=cov)[3]
    assert tau_few > tau_many
    assert tau_many == pytest.approx(tau_chi, rel=1e-2)
    with pytest.raises(ValueError, match="df_denominator"):
        dvg.tl.parallel_curves_test(mean, Q, cov=cov, df_denominator=0)


def test_slides_stop_the_pooled_overconfidence():
    """Slides with their own level and curve but one condition curve: pooled fits call it a difference, slide-aware fits do not."""
    adata = _slides()
    kwargs = {
        "test": "parallel",
        "tau": 0.0,
        "inplace": False,
        "progressbar": False,
        "inference": "nuts",
        "chains": 2,
        "draws": 300,
        "tune": 300,
    }

    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None)
    pooled, _, _ = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), **kwargs)

    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide")
    aware, _, curves = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), **kwargs)

    assert pooled["parallel_pval"].iloc[0] < 1e-3
    assert aware["parallel_pval"].iloc[0] > 0.01
    assert aware["parallel_df_denominator"].iloc[0] == 8
    assert aware["n_divergent"].iloc[0] <= 6  # under 1% of the 600 draws; a few divergences are sampler noise
    assert np.isfinite(curves["cond0_mean"].to_numpy()).all()


def test_slide_aware_fits_run_every_test_but_refuse_profiles():
    adata = _slides(n_per_slide=100)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide")
    setup = dvg.tl.setup_model(adata)
    for test in ("wald", "level"):
        stats, _, _ = dvg.tl.fit_model(adata, setup, test=test, inplace=False, progressbar=False)
        assert stats[f"{test}_df_denominator"].iloc[0] == 8
    with pytest.raises(ValueError, match="does not model slides"):
        dvg.tl.fit_profiles(adata, setup, inplace=False)


def test_cells_without_a_sample_are_left_out_of_the_binning():
    adata = _slides(n_per_slide=200)
    dm = adata.obsm["design_matrix"]
    dm["slide"] = dm["slide"].astype(object)
    dm.loc[dm["slide"] == "s3", ["slide", "anchor"]] = [np.nan, np.nan]  # e.g. a slide without anchor cells
    b = dvg.pp.bin_obs(
        adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide", inplace=False
    )
    assert b.n_samples == 9 and "s3" not in b.sample_labels
    assert b.n_obs.sum() == adata.n_obs - 200


def test_two_stage_slope_test_on_slides():
    """Slides with random slopes and no condition effect stay null; a real slope difference is found."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(12):
        cond = "a" if s < 6 else "b"
        x = rng.random(400)
        slope = rng.normal(1.0, 0.3) + (0.8 if cond == "b" else 0.0)
        null = rng.normal(0, 0.3) + rng.normal(1.0, 0.3) * x  # random slide slope, no condition effect
        eff = slope * x
        for xi, n_i, e_i in zip(x, null, eff, strict=True):
            rows.append((xi, cond, f"s{s}", n_i + rng.normal(0, 0.2), e_i + rng.normal(0, 0.2)))
    df = pd.DataFrame(rows, columns=["anchor", "condition", "slide", "null", "effect"])
    adata = ad.AnnData(df[["null", "effect"]].to_numpy(np.float32), var=pd.DataFrame(index=["null", "effect"]))
    adata.obsm["design_matrix"] = df[["anchor", "condition", "slide"]].set_index(adata.obs_names)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide")
    stats, _, _ = dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), test="slope", inplace=False, progressbar=False)
    assert stats.loc["effect", "slope_pval"] < 0.01 and stats.loc["null", "slope_pval"] > 0.01
    assert abs(stats.loc["effect", "slope_diff"] - 0.8) < 0.3
    assert stats.loc["effect", "slope_df_denominator"] == 10


def test_tests_use_the_anchor_range_where_the_cells_are():
    """Cells only in the middle of the anchor range: the tests use that range and stay calibrated."""
    rng = np.random.default_rng(3)
    rows = []
    for s in range(10):
        cond = "a" if s < 5 else "b"
        x = rng.uniform(0.4, 0.8, 300)
        tilt = rng.normal(0, 0.3)  # slide-level slope, no condition effect
        for xi in x:
            rows.append((xi, cond, f"s{s}", tilt * (xi - 0.6) + rng.normal(0, 0.2)))
    df = pd.DataFrame(rows, columns=["anchor", "condition", "slide", "y"])
    df.loc[df.index[:2], "anchor"] = [0.0, 1.0]  # bin_obs scales by the full range; keep [0, 1] as the anchor range
    adata = ad.AnnData(df[["y"]].to_numpy(np.float32), var=pd.DataFrame(index=["y"]))
    adata.obsm["design_matrix"] = df[["anchor", "condition", "slide"]].set_index(adata.obs_names)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None, sample="slide")
    with pytest.warns(UserWarning, match="no cells"):
        stats, _, _ = dvg.tl.fit_model(
            adata, dvg.tl.setup_model(adata), test="slope", inplace=False, progressbar=False, two_stage=False
        )
    assert stats.loc["y", "slope_pval"] > 0.01


def test_anchor_support_is_the_central_cells_where_both_conditions_are():
    from types import SimpleNamespace

    from diffsvg.tl.model import _anchor_support

    x = np.linspace(0, 1, 11)
    counts = np.zeros((2, 11))
    counts[:, 3:9] = 100  # both conditions at 0.3-0.8
    counts[0, 1] = 50  # one condition only: not part of the support
    assert np.allclose(_anchor_support(SimpleNamespace(n_obs_grid=counts, bin_centers=x)), (0.3, 0.8))
    assert _anchor_support(SimpleNamespace(n_obs_grid=None, bin_centers=x)) == (0.0, 1.0)


def test_advi_is_an_alias_for_auto():
    adata = _slides(n_per_slide=100)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=N_BINS, window_radius=None)
    with pytest.warns(FutureWarning, match="advi"):
        stats, _, _ = dvg.tl.fit_model(
            adata, dvg.tl.setup_model(adata), inference="advi", inplace=False, progressbar=False
        )
    assert stats["parallel_pval"].notna().all()
