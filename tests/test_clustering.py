import subprocess
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import diffsvg as dvg
from diffsvg.tl.clustering import _features

K = 10
N_GRID = 21


def _fitted(n_per_group=8, seed=0, noise=0.05):
    """An AnnData as `fit_model` leaves it: two coefficient groups in the average, two others in `beta_1`, and curves."""
    rng = np.random.default_rng(seed)
    rising, falling = np.linspace(-1, 1, K), np.linspace(1, -1, K)
    bump = np.cos(np.linspace(-np.pi, np.pi, K))
    n_genes = 2 * n_per_group
    genes = [f"g{i}" for i in range(n_genes)]
    in_first = np.arange(n_genes) < n_per_group
    amplitude = rng.uniform(0.2, 3.0, n_genes)[:, None]  # genes share a shape, not a strength

    average = np.where(in_first[:, None], rising, falling) * amplitude + rng.normal(0, noise, (n_genes, K))
    # the interaction groups cut across the average groups, so a wrong table gives a wrong answer
    even = np.arange(n_genes) % 2 == 0
    beta_1 = np.where(even[:, None], bump, -bump) + rng.normal(0, noise, (n_genes, K))
    beta = average - beta_1 / 2

    x = np.linspace(0, 1, N_GRID)
    columns = [str(round(float(v), 4)) for v in x]
    cond0 = (
        np.array([a * (1 - 2 * x) * (1 if f else -1) for a, f in zip(amplitude.ravel(), in_first, strict=True)]) + 5.0
    )
    cond1 = cond0 + np.array([(1 if e else -1) * np.cos(np.pi * x) + 2.0 for e in even])

    adata = ad.AnnData(np.zeros((3, n_genes), dtype=np.float32), var=pd.DataFrame(index=genes))
    adata.varm["dsvg_coefs_beta"] = pd.DataFrame(beta, index=genes, columns=[f"B{k}" for k in range(K)])
    adata.varm["dsvg_coefs_beta_1"] = pd.DataFrame(beta_1, index=genes, columns=[f"B{k}" for k in range(K)])
    adata.varm["dsvg_curves_cond0_mean"] = pd.DataFrame(cond0, index=genes, columns=columns)
    adata.varm["dsvg_curves_cond1_mean"] = pd.DataFrame(cond1, index=genes, columns=columns)
    adata.uns["dsvg_curves_condition_categories"] = {"0": "Ctrl", "1": "SHH"}
    return adata, in_first, even


def _same_partition(labels, truth):
    """Whether `labels` split the genes exactly as the boolean `truth` does, in either numbering."""
    labels = np.asarray(labels).astype(str)
    return len(set(labels[truth])) == 1 and len(set(labels[~truth])) == 1 and set(labels[truth]) != set(labels[~truth])


def test_average_and_interaction_read_their_own_coefficients():
    adata, in_first, even = _fitted()
    dvg.tl.cluster_curves(adata, n_clusters=2)
    dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2)
    assert adata.uns["dsvg_cluster_average"]["by"] == "average"
    assert _same_partition(adata.var["dsvg_cluster_average"], in_first)
    assert _same_partition(adata.var["dsvg_cluster_interaction"], even)


def test_average_is_the_coefficient_vector_of_the_mean_curve():
    adata, _, even = _fitted()
    beta, beta_1 = adata.varm["dsvg_coefs_beta"], adata.varm["dsvg_coefs_beta_1"]
    # a stronger interaction must move "average" to the interaction's partition
    adata.varm["dsvg_coefs_beta_1"] = beta_1 * 100
    average = dvg.tl.cluster_curves(adata, by="average", n_clusters=2, inplace=False)
    assert _same_partition(average, even)

    table = dvg.tl.clustering._coefficient_table(adata, "average", "dsvg_coefs", "dsvg_profiles")
    np.testing.assert_allclose(table.to_numpy(), beta.to_numpy() + adata.varm["dsvg_coefs_beta_1"].to_numpy() / 2)


def test_scaling_groups_by_shape_and_not_by_strength():
    rng = np.random.default_rng(1)
    rising, falling = np.linspace(-1, 1, K), np.linspace(1, -1, K)
    strengths = np.array([0.1, 0.1, 0.1, 5.0, 5.0, 5.0] * 2)[:, None]
    shapes = np.vstack([rising] * 6 + [falling] * 6)
    genes = [f"g{i}" for i in range(12)]
    adata = ad.AnnData(np.zeros((2, 12), dtype=np.float32), var=pd.DataFrame(index=genes))
    table = shapes * strengths + rng.normal(0, 0.005, (12, K))
    adata.varm["dsvg_coefs_beta_1"] = pd.DataFrame(table, index=genes, columns=[f"B{k}" for k in range(K)])
    is_rising = np.arange(12) < 6

    scaled = dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2, inplace=False, scale=True)
    assert _same_partition(scaled, is_rising)
    # k = 3 without scaling: the weak genes of both shapes share a flat cluster, whichever shape they have
    raw = dvg.tl.cluster_curves(adata, by="interaction", n_clusters=3, inplace=False).to_numpy()
    assert len(set(raw[[0, 1, 2, 6, 7, 8]])) == 1
    assert len({raw[3], raw[9], raw[0]}) == 3


def test_the_constant_direction_of_the_coefficients_does_not_count():
    # It leaves the curve unchanged (the centred basis columns sum to 0) ...
    x = np.linspace(0, 1, 200)
    B, _ = dvg.pp.build_spline_basis(x, dvg.pp.compute_interior_knots(K, degree=3)[1:-1], degree=3)
    np.testing.assert_allclose(B.sum(axis=1), 0, atol=1e-10)
    # ... so it must not move the features or the clusters.
    adata, in_first, _ = _fitted()
    plain = _features(adata.varm["dsvg_coefs_beta"], scale=False)
    rng = np.random.default_rng(2)
    shifted = adata.varm["dsvg_coefs_beta"] + rng.normal(0, 10, (adata.n_vars, 1))
    np.testing.assert_allclose(_features(shifted, scale=False), plain)
    adata.varm["dsvg_coefs_beta"] = shifted
    assert _same_partition(dvg.tl.cluster_curves(adata, n_clusters=2, inplace=False), in_first)


def test_labels_run_from_the_largest_cluster_and_skip_genes_not_clustered():
    adata, in_first, _ = _fitted()
    table = adata.varm["dsvg_coefs_beta"].copy()
    table.iloc[0] = np.nan  # a gene that was not fitted
    adata.varm["dsvg_coefs_beta"] = table
    subset = [g for g, keep in zip(adata.var_names, ~in_first | (np.arange(adata.n_vars) < 3), strict=True) if keep]

    dvg.tl.cluster_curves(adata, n_clusters=2, genes=subset, scale=True, key_added="mine")
    labels = adata.var["mine"]
    assert labels.isna().sum() == adata.n_vars - len(subset) + 1  # the rest of the panel, plus the NaN gene
    assert list(labels.cat.categories) == ["0", "1"]
    assert (labels == "0").sum() > (labels == "1").sum()

    params = adata.uns["mine"]
    assert params["by"] == "average" and params["n_clusters"] == 2
    assert params["centers"].shape == (2, K)
    assert -1 <= params["silhouette"] <= 1
    # centers follow the relabelled clusters: cluster 0 is the falling group here
    assert params["centers"][0, 0] > 0 > params["centers"][1, 0]


def test_clustering_is_reproducible_and_h5ad_safe(tmp_path):
    adata, *_ = _fitted()
    first = dvg.tl.cluster_curves(adata, n_clusters=3, inplace=False, random_state=7)
    second = dvg.tl.cluster_curves(adata, n_clusters=3, inplace=False, random_state=7)
    pd.testing.assert_series_equal(first, second)

    dvg.tl.cluster_curves(adata, n_clusters=3)
    adata.write_h5ad(tmp_path / "x.h5ad")
    back = ad.read_h5ad(tmp_path / "x.h5ad")
    assert list(back.var["dsvg_cluster_average"]) == list(adata.var["dsvg_cluster_average"])
    np.testing.assert_allclose(
        back.uns["dsvg_cluster_average"]["centers"], adata.uns["dsvg_cluster_average"]["centers"]
    )


def test_cluster_curves_rejects_bad_input():
    adata, *_ = _fitted()
    with pytest.raises(ValueError, match="Unknown by"):
        dvg.tl.cluster_curves(adata, by="nope")
    with pytest.raises(KeyError, match="fit_profiles"):
        dvg.tl.cluster_curves(adata, by="profile")
    with pytest.raises(KeyError, match="not found"):
        dvg.tl.cluster_curves(adata, genes=["missing"])
    with pytest.raises(ValueError, match="clusters from"):
        dvg.tl.cluster_curves(adata, n_clusters=1000)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _panel_curves(panel):
    return {line.get_label(): line for line in panel.get_lines()}


def test_one_panel_per_cluster_with_every_gene_in_its_own_colour():
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="average", n_clusters=2)
    plot = dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_average")
    assert len(plot) == 2
    for panel, cluster in zip(plot, ["0", "1"], strict=True):
        members = set(adata.var_names[adata.var["dsvg_cluster_average"] == cluster])
        assert set(_panel_curves(panel)) == members
        colors = [line.get_color() for line in panel.get_lines()]
        assert len(set(colors)) == len(members)


def test_default_curves_are_the_ones_the_clustering_compared():
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2)
    dvg.tl.cluster_curves(adata, by="average", n_clusters=2)
    x_axis = adata.varm["dsvg_curves_cond0_mean"]
    diff = adata.varm["dsvg_curves_cond1_mean"] - x_axis

    def first_curve(key, **kwargs):
        panel = dvg.pl.plot_cluster_curves(adata, key, standardize="none", **kwargs)[0]
        label, curve = next(iter(_panel_curves(panel).items()))
        return label, curve.get_ydata()

    gene, y = first_curve("dsvg_cluster_interaction")
    np.testing.assert_allclose(y, diff.loc[gene])
    gene, y = first_curve("dsvg_cluster_average")
    np.testing.assert_allclose(y, x_axis.loc[gene] + diff.loc[gene] / 2)

    # also with a `fit_profiles` result around, which is another curve than the one that was clustered
    adata.varm["dsvg_profiles_mean"] = x_axis * 0.0
    gene, y = first_curve("dsvg_cluster_average")
    np.testing.assert_allclose(y, x_axis.loc[gene] + diff.loc[gene] / 2)


def test_conditions_are_two_panels_per_cluster_and_keep_the_gap():
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2)
    plot = dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves="conditions", standardize="center")
    assert len(plot) == 4
    ctrl, shh = plot[0], plot[1]
    gene = next(iter(_panel_curves(ctrl)))
    y0 = _panel_curves(ctrl)[gene].get_ydata()
    y1 = _panel_curves(shh)[gene].get_ydata()
    # centred over both conditions together, so the 2.0 level gap between them is still there
    assert (y1 - y0).mean() == pytest.approx(2.0, abs=1e-9)
    assert (np.r_[y0, y1]).mean() == pytest.approx(0.0, abs=1e-9)
    assert ctrl.get_ylim() == shh.get_ylim()


@pytest.mark.parametrize(("label", "table"), [("Ctrl", "dsvg_curves_cond0_mean"), ("SHH", "dsvg_curves_cond1_mean")])
def test_a_condition_label_draws_that_condition_alone(label, table):
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2)
    plot = dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves=label, standardize="none")
    assert len(plot) == 2  # one panel per cluster, not two
    assert plot[0].get_title().endswith(f", {label}")
    for panel in plot:
        for gene, curve in _panel_curves(panel).items():
            np.testing.assert_allclose(curve.get_ydata(), adata.varm[table].loc[gene])

    centred = dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves=label)[0]
    for curve in _panel_curves(centred).values():
        assert curve.get_ydata().mean() == pytest.approx(0.0, abs=1e-9)


def test_standardize_is_per_gene_and_follows_the_clustering():
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="average", n_clusters=2, scale=True, key_added="scaled")
    dvg.tl.cluster_curves(adata, by="average", n_clusters=2, key_added="raw")
    panel = dvg.pl.plot_cluster_curves(adata, "scaled")[0]
    for curve in panel.get_lines():
        y = curve.get_ydata()
        assert y.mean() == pytest.approx(0, abs=1e-9)
        assert y.std() == pytest.approx(1, abs=1e-9)
    assert panel.get_ylabel() == "z-scored expression"
    raw = dvg.pl.plot_cluster_curves(adata, "raw")[0]
    assert raw.get_ylabel() == "centred expression"
    assert raw.get_lines()[0].get_ydata().mean() == pytest.approx(0, abs=1e-9)


def test_plot_cluster_curves_renders_and_rejects_bad_input(tmp_path):
    adata, *_ = _fitted()
    dvg.tl.cluster_curves(adata, by="interaction", n_clusters=2)
    out = tmp_path / "clusters.png"
    dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves="conditions", dpi=60, save_path=str(out))
    assert out.stat().st_size > 0
    assert len(dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", clusters=1)) == 1

    with pytest.raises(KeyError, match="cluster_curves"):
        dvg.pl.plot_cluster_curves(adata, "nope")
    with pytest.raises(KeyError, match="clusters not found"):
        dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", clusters=[9])
    with pytest.raises(ValueError, match="Unknown curves"):
        dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", curves="nope")
    with pytest.raises(ValueError, match="Unknown standardize"):
        dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_interaction", standardize="nope")


def _with_curves():
    """`_fitted` plus the credible bands that `plot_prediction_curve` also reads."""
    adata, *_ = _fitted()
    for cat in (0, 1):
        mean = adata.varm[f"dsvg_curves_cond{cat}_mean"]
        adata.varm[f"dsvg_curves_cond{cat}_ci_low"] = mean - 0.5
        adata.varm[f"dsvg_curves_cond{cat}_ci_high"] = mean + 0.5
    return adata


def test_prediction_curve_draws_both_conditions_with_their_bands():
    adata = _with_curves()
    ax = dvg.pl.plot_prediction_curve(adata, gene="g0")
    assert ax.get_title() == "g0"
    assert len(ax.collections) == 2  # the two bands
    labels = {line.get_label(): line for line in ax.get_lines()}
    assert set(labels) == {"Ctrl", "SHH"}
    np.testing.assert_allclose(labels["SHH"].get_ydata(), adata.varm["dsvg_curves_cond1_mean"].loc["g0"])
    x = adata.varm["dsvg_curves_cond1_mean"].columns.to_numpy(dtype=float)
    np.testing.assert_allclose(labels["SHH"].get_xdata(), x)


def test_prediction_curve_grid_has_one_panel_per_gene_in_order_and_one_legend():
    adata = _with_curves()
    genes = ["g3", "g0", "g5", "g1", "g2"]
    axes = dvg.pl.plot_prediction_curve(adata, gene=genes, ncols=2)
    assert [a.get_title() for a in axes] == genes
    assert axes[0].get_gridspec().get_geometry() == (3, 2)
    assert [a.get_legend() is not None for a in axes] == [True, False, False, False, False]
    # a one-element list is one panel, like the bare name
    assert dvg.pl.plot_prediction_curve(adata, gene=["g0"]).get_title() == "g0"


def test_prediction_curve_draws_into_a_given_axes_and_saves(tmp_path):
    adata = _with_curves()
    _, given = plt.subplots()
    out = tmp_path / "curve.png"
    assert dvg.pl.plot_prediction_curve(adata, gene="g0", ax=given, dpi=60, save_path=str(out)) is given
    assert out.stat().st_size > 0


def test_prediction_curve_rejects_bad_input():
    adata = _with_curves()
    with pytest.raises(ValueError, match="empty"):
        dvg.pl.plot_prediction_curve(adata, gene=[])
    with pytest.raises(KeyError, match="not found"):
        dvg.pl.plot_prediction_curve(adata, gene=["nope"])
    _, given = plt.subplots()
    with pytest.raises(ValueError, match="single panel"):
        dvg.pl.plot_prediction_curve(adata, gene=["g0", "g1"], ax=given)


def test_curve_plots_do_not_need_holoviews():
    code = "import sys; import diffsvg; sys.exit('holoviews' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def _simulated(n_cells=1500, seed=0):
    """Two genes with opposite curves in condition a and a different interaction each; column order is not sorted."""
    rng = np.random.default_rng(seed)
    x = rng.random(n_cells)
    x[:2] = 0.0, 1.0
    cond = np.where(np.arange(n_cells) % 2 == 0, "a", "b")
    is_b = cond == "b"
    y = np.c_[
        np.sin(3 * x) + is_b * (x - 0.5) * 2 + rng.normal(0, 0.2, n_cells),
        -np.sin(3 * x) + is_b * 0.8 + rng.normal(0, 0.2, n_cells),
    ]
    adata = ad.AnnData(y.astype(np.float32), var=pd.DataFrame(index=["tilted", "parallel"]))
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


def _on_grid(adata, table):
    """The curve of a coefficient vector on the prediction grid, up to its level."""
    B_pred = adata.uns["binned"]["B_pred"]
    curve = B_pred @ table
    return curve - curve.mean()


def test_fit_model_stores_the_coefficients_of_each_gene():
    adata = _simulated()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20, window_radius=None)
    dvg.tl.fit_model(
        adata, dvg.tl.setup_model(adata), inference="advi", n=0, test="parallel", progressbar=False, n_workers=2
    )

    beta, beta_1 = adata.varm["dsvg_coefs_beta"], adata.varm["dsvg_coefs_beta_1"]
    assert beta.shape == beta_1.shape == (2, K)
    cond0, cond1 = adata.varm["dsvg_curves_cond0_mean"], adata.varm["dsvg_curves_cond1_mean"]
    for gene in adata.var_names:  # rows must be this gene's, whatever order the workers finished in
        drawn = (cond0.loc[gene] - cond0.loc[gene].mean()).to_numpy()
        np.testing.assert_allclose(_on_grid(adata, beta.loc[gene].to_numpy()), drawn, atol=0.1)
        gap = (cond1.loc[gene] - cond0.loc[gene]).to_numpy()
        np.testing.assert_allclose(_on_grid(adata, beta_1.loc[gene].to_numpy()), gap - gap.mean(), atol=0.1)
    # the interaction tells the tilted gene from the parallel one
    assert np.ptp(_on_grid(adata, beta_1.loc["tilted"].to_numpy())) > 1.5
    assert np.ptp(_on_grid(adata, beta_1.loc["parallel"].to_numpy())) < 0.5


def test_fit_model_without_inplace_returns_the_coefficients():
    adata = _simulated()
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20, window_radius=None)
    _, _, tables = dvg.tl.fit_model(
        adata, dvg.tl.setup_model(adata), inference="advi", n=0, test="parallel", inplace=False, progressbar=False
    )
    assert list(tables["beta"].index) == ["tilted", "parallel"]
    assert list(tables["beta_1"].columns) == [f"B{k}" for k in range(K)]
    assert "dsvg_coefs_beta" not in adata.varm


def test_fit_profiles_stores_coefficients_for_clustering():
    adata = _simulated()
    dvg.pp.bin_obs(adata, anchor="anchor", condition=None, n_bins=20, window_radius=None)
    dvg.tl.fit_profiles(adata, dvg.tl.setup_model(adata), progressbar=False, draws=200, tune=200, chains=2)
    assert adata.varm["dsvg_profiles_beta"].shape == (2, K)
    dvg.tl.cluster_curves(adata, by="profile", n_clusters=2)
    assert adata.uns["dsvg_cluster_profile"]["by"] == "profile"
    plot = dvg.pl.plot_cluster_curves(adata, "dsvg_cluster_profile")
    assert len(plot) == 2
