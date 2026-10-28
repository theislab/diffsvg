"""Rejection rates (alpha = 0.05) of the current test vs the proposed parallel-curves test, by scenario."""

import sys
import warnings

import numpy as np
import pandas as pd

from proto import (
    beta1_block,
    bin_means,
    current_wald,
    eval_basis,
    fit_closed_form,
    make_basis,
    shape_metric,
    slope_metric,
    slope_test,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
G = 200  # genes per scenario
ALPHA = 0.05
TAU_SLOPE = 0.1  # RMS slope difference over the anchor range (data units)
TAU_SHAPE = TAU_SLOPE / np.sqrt(12)  # same threshold expressed for a linear difference


def positions(n):
    return np.where(rng.random(n) < 0.8, rng.beta(1.5, 2.5, n), rng.random(n))


SCENARIOS = {
    "identical": lambda x: 0 * x,
    "level 0.3": lambda x: 0.3 + 0 * x,
    "tiny slope 0.03": lambda x: 0.03 * (x - 0.5),
    "crossing slope 0.3": lambda x: 0.3 * (x - 0.5),
    "slope 0.3 + level": lambda x: 0.3 * x,
    "peak shift": lambda x: 0.5 * np.exp(-3 * np.abs(x - 0.15)) - 0.5 * np.exp(-3 * x),
}


def run(n_cells, window_radius, weighted):
    rows = []
    x = np.r_[positions(n_cells), positions(n_cells)]
    c = np.r_[np.zeros(n_cells), np.ones(n_cells)]
    for name, d in SCENARIOS.items():
        a = rng.uniform(0.2, 1.0, G)
        b = rng.uniform(1, 5, G)
        f0 = a[None] * np.exp(-b[None] * x[:, None])
        Y = f0 + c[:, None] * d(x)[:, None] + rng.normal(0, 1, (2 * n_cells, G))
        xb, cb, nb, Yb = bin_means(x, c, Y, window_radius=window_radius)
        B, info = make_basis(xb)
        mean, cov, sig, K = fit_closed_form(Yb, B, cb, weights=nb if weighted else None)
        m1, V1 = beta1_block(mean, cov, K)
        _, p_cur = current_wald(mean, cov, sig, K, eval_basis(info, np.linspace(0, 1, 51)))
        Es, _, p_point, p_slope = slope_test(m1, V1, slope_metric(info), TAU_SLOPE)
        Ea, _, _, p_shape = slope_test(m1, V1, shape_metric(info), TAU_SHAPE)
        rows.append(
            {
                "scenario": name,
                "current wald": np.mean(p_cur < ALPHA),
                "parallel, tau=0": np.mean(p_point < ALPHA),
                "slope<=tau": np.mean(p_slope < ALPHA),
                "shape<=tau": np.mean(p_shape < ALPHA),
                "E_slope (med)": np.median(Es),
                "E_shape*sqrt12 (med)": np.median(Ea) * np.sqrt(12),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for n_cells in (2000, 10000, 50000):
        for window_radius, weighted, label in ((0.02, False, "current bins (overlapping)"), (0.01, True, "contiguous bins, n-weighted")):
            print(f"\n=== {n_cells} cells/condition, {label} ===")
            print(run(n_cells, window_radius, weighted).round(3).to_string())
