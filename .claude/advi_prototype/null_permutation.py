"""Calibration and repeatability of inference='advi' against the exact posterior (small: 2 genes, 4 workers).

(1) Permutation null: permute `condition` across cells (identical curves by construction) and compare exact / Laplace (n=0) / ADVI.
(2) Repeatability: the real Ctrl-vs-SHH data with several ADVI seeds.
The ADVI arm is the package's `_fit_advi` plus the test on its beta_1 draws, as `fit_model` runs it.
"""

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lab
from diffsvg.tl import model as M

GENES = ["GLI1", "VAMP2"]
N_PERM, N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40, int(sys.argv[2]) if len(sys.argv) > 2 else 10


def vi_row(model, **settings):
    cfg = {**M._VI_DEFAULTS, **settings}
    with model:
        idata, cols = M._fit_advi(model, cfg, None)
    draws = idata.posterior["beta_1"].stack(s=("chain", "draw")).to_numpy().T
    out = {}
    for tau in (0.0, 0.1):
        E, W, _, p = M.parallel_curves_test(draws, M.shape_metric(lab.K, 3), tau=tau)
        out[f"p{tau}"] = p
    return {"E": E, "W": W, **out, **cols}


def task(gene, kind, rep, B, y, c, init):
    rows = []
    X = lab.design(B, c)
    m, V, _ = lab.exact_posterior(X, y)
    E, W, _, p01 = lab.test_from_moments(*lab.beta1_block(m, V), tau=0.1)
    p0 = lab.test_from_moments(*lab.beta1_block(m, V), tau=0.0)[3]
    rows.append({"method": "exact", "E": E, "W": W, "p0.0": p0, "p0.1": p01})
    model = lab.build_model(B, y, c, init)
    rows.append({"method": "laplace", **vi_row(model, n=0, random_seed=rep)})
    rows.append({"method": "advi", **vi_row(model, random_seed=rep)})
    return [{"gene": gene, "kind": kind, "rep": rep, **r} for r in rows]


if __name__ == "__main__":
    adata, setup = lab.load(GENES)
    B, _, c, Y, init = lab.per_cell(adata, setup)
    jobs = []
    for g in GENES:
        for r in range(N_PERM):
            jobs.append(delayed(task)(g, "perm", r, B, Y[g], np.random.default_rng(1000 + r).permutation(c), init[g]))
        for s in range(N_SEEDS):  # real data, ADVI seed varies (the exact arm repeats, which is harmless)
            jobs.append(delayed(task)(g, "real", s, B, Y[g], c, init[g]))
    out = pd.DataFrame([r for rows in Parallel(n_jobs=4, backend="loky")(jobs) for r in rows])
    out.to_csv("results/null_permutation.csv", index=False)
    print("saved results/null_permutation.csv", out.shape)
