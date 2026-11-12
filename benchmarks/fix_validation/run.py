"""Old-vs-new validation of the statistical fixes on real and synthetic data.

    python run.py --code {old,new} --exp <name> [--out results/]

`--code old` imports diffsvg from DIFFSVG_OLD_SRC (a checkout of main), `new` from this branch.
Data paths come from DIFFSVG_LEGNINI and DIFFSVG_XENIUM. Every run writes `<out>/<code>_<exp>.csv`
(the stats table) and `<out>/<code>_<exp>_curves.npz` (delta curves, grid).
"""

import argparse
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--code", choices=["old", "new"], required=True)
parser.add_argument("--exp", required=True)
parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

if args.code == "old":
    sys.path.insert(0, os.environ["DIFFSVG_OLD_SRC"])
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import squidpy as sq  # noqa: E402

import diffsvg as dvg  # noqa: E402

print("diffsvg from", dvg.__file__, flush=True)
LAYER = "log1p_norm"
FAST = {"draws": 500, "tune": 500, "chains": 2}  # for the many-gene null runs


def legnini():
    adata = sc.read_h5ad(os.environ["DIFFSVG_LEGNINI"])
    shh = adata[:, "SHH"].X
    shh = (shh.toarray() if hasattr(shh, "toarray") else np.asarray(shh)).ravel()
    adata.obs["SHH+"] = (shh > np.percentile(shh, 99)).astype(str)  # Laurens' anchor, kept for comparability
    sq.tl.var_by_distance(adata, cluster_key="SHH+", groups="True", library_key="sample", covariates="condition")
    return adata


def legnini_split_null(seed):
    """SHH slides only, randomly split into two pseudo-conditions: no gene truly differs."""
    adata = legnini()
    adata = adata[adata.obs["condition"] == "SHH"].copy()
    slides = sorted(adata.obs["sample"].unique())
    rng = np.random.default_rng(seed)
    group_a = set(rng.choice(slides, size=len(slides) // 2, replace=False))
    design = adata.obsm["design_matrix"].copy()
    design["condition"] = np.where(design["sample"].astype(str).isin(group_a), "A", "B")
    adata.obsm["design_matrix"] = design
    return adata


def xenium():
    adata = sc.read_h5ad(os.environ["DIFFSVG_XENIUM"])
    sc.pp.filter_cells(adata, min_counts=10)
    adata.layers[LAYER] = adata.X.copy()
    sc.pp.normalize_total(adata, layer=LAYER)
    sc.pp.log1p(adata, layer=LAYER)
    sq.tl.var_by_distance(
        adata, cluster_key="Dentate_Gyrus", groups="anchor", library_key="condition", covariates="condition"
    )
    return adata


def synthetic_null(seed=0, n_genes=300, n_cells=20000):
    """Two conditions, one section each, identical true curves per gene; genes differ in level, shape and noise."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_cells) ** 1.5  # denser near the anchor, sparse at the far end like real tissue
    cond = rng.choice(["c0", "c1"], n_cells)
    level = rng.uniform(0.2, 2.0, n_genes)
    amp = rng.uniform(0, 1.0, n_genes)
    freq = rng.uniform(0.5, 2.5, n_genes)
    noise = rng.uniform(0.2, 1.0, n_genes)
    truth = level + amp * np.sin(np.pi * freq * x[:, None])
    X = truth + noise * rng.standard_normal((n_cells, n_genes))
    adata = ad.AnnData(X.astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers[LAYER] = adata.X.copy()
    adata.obsm["design_matrix"] = pd.DataFrame({"anchor": x, "condition": cond}, index=adata.obs_names)
    return adata


# name: (loader, anchor, bin_obs kwargs, fit_model kwargs). An empty kwargs dict = the package defaults.
EXPERIMENTS = {
    "legnini_default": (legnini, "True", {}, {}),
    "legnini_parallel_pooled": (legnini, "True", {"window_radius": None}, {"test": "parallel", "tau": 0.1}),
    "legnini_parallel_slides": (
        legnini, "True", {"window_radius": None, "sample": "sample"}, {"test": "parallel", "tau": 0.1}
    ),
    "legnini_wald_weighted": (legnini, "True", {"window_radius": None}, {"test": "wald"}),
    "xenium_default": (xenium, "anchor", {}, {}),
    "xenium_parallel": (xenium, "anchor", {"window_radius": None}, {"test": "parallel", "tau": 0.1}),
    "synth_null_default": (synthetic_null, "anchor", {}, {"sample_kwargs": FAST}),
    "synth_null_wald_weighted": (synthetic_null, "anchor", {"window_radius": None}, {"test": "wald", "sample_kwargs": FAST}),
    "synth_null_parallel": (
        synthetic_null, "anchor", {"window_radius": None}, {"test": "parallel", "tau": 0.1, "sample_kwargs": FAST}
    ),
}
for s in range(3):
    EXPERIMENTS[f"legnini_null{s}_default"] = (lambda s=s: legnini_split_null(s), "True", {}, {"sample_kwargs": FAST})
    EXPERIMENTS[f"legnini_null{s}_wald_weighted"] = (
        lambda s=s: legnini_split_null(s), "True", {"window_radius": None}, {"test": "wald", "sample_kwargs": FAST}
    )
    EXPERIMENTS[f"legnini_null{s}_parallel_pooled"] = (
        lambda s=s: legnini_split_null(s), "True", {"window_radius": None},
        {"test": "parallel", "tau": 0.1, "sample_kwargs": FAST},
    )
    EXPERIMENTS[f"legnini_null{s}_parallel_slides"] = (
        lambda s=s: legnini_split_null(s), "True", {"window_radius": None, "sample": "sample"},
        {"test": "parallel", "tau": 0.1, "sample_kwargs": FAST},
    )

loader, anchor, bin_kwargs, fit_kwargs = EXPERIMENTS[args.exp]
adata = loader()
t0 = time.perf_counter()
dvg.pp.bin_obs(adata, anchor=anchor, condition="condition", layer=LAYER, **bin_kwargs)
dvg.tl.fit_model(adata, dvg.tl.setup_model(adata), n_workers=args.workers, progressbar=False, **fit_kwargs)
elapsed = time.perf_counter() - t0

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
stats = adata.varm["dsvg_stats"].copy()
stats.attrs = {}
stats.to_csv(out / f"{args.code}_{args.exp}.csv")
curves = {k: np.asarray(adata.varm[k]) for k in adata.varm if str(k).startswith("dsvg_")}
np.savez(
    out / f"{args.code}_{args.exp}_curves.npz",
    genes=np.asarray(adata.var_names),
    grid=np.asarray(adata.uns["binned"].bin_centers if hasattr(adata.uns["binned"], "bin_centers") else []),
    **{k: v for k, v in curves.items() if v.dtype != object},
)
print(f"{args.code} {args.exp}: {elapsed:.0f} s, {len(stats)} genes, params {adata.uns.get('dsvg_stats_params')}", flush=True)
