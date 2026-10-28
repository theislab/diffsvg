"""Helpers for the ADVI checks (see .claude/advi_plan.md): legnini loading and the exact posterior of the Normal model.

`prep_legnini.py` writes the cache (`legnini_dm.h5ad`, not to be committed); set DIFFSVG_LEGNINI_DM to put it elsewhere.
"""

import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.special import logsumexp

import diffsvg as dvg
from diffsvg.tl import model as M

warnings.filterwarnings("ignore")
DATA = os.environ.get("DIFFSVG_LEGNINI_DM", str(Path(__file__).with_name("legnini_dm.h5ad")))
K = 10


def load(genes, layer="log1p_norm"):
    """The legnini AnnData with the notebook's design matrix, binned for `genes` (only basis and grid are used per cell)."""
    adata = ad.read_h5ad(DATA)[:, list(genes)].copy()
    dvg.pp.bin_obs(
        adata, anchor="True", condition="condition", genes=list(genes), design_matrix="design_matrix", n_bins=50, layer=layer
    )
    return adata, dvg.tl.setup_model(adata)


def per_cell(adata, setup):
    """(B, B_masked, condition, {gene: y}, {gene: intercept_initval}) as `fit_model(use_binned=False)` uses them."""
    return M._fit_inputs(adata, setup.binned, use_binned=False)


def design(B, c):
    """Columns [intercept, beta (K), categorical_beta, beta_1 (K)]."""
    return np.column_stack([np.ones(len(c)), B, c, B * c[:, None]])


def exact_posterior(X, y, sigma_prior=1.0, n_grid=401, span=6.0):
    """Exact posterior moments of the Normal GAM with N(0, 1) priors, sigma integrated out on a grid.

    Given sigma the coefficients are Gaussian, so the posterior is a scale mixture; p(sigma | y) is a 1-D integral.
    Returns (mean, cov, sigma_mean) with the coefficients ordered like `design`.
    """
    n, p = X.shape
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y

    def logp(s):
        L = np.linalg.cholesky(XtX / s**2 + np.eye(p))
        z = np.linalg.solve(L, Xty / s**2)
        return -n * np.log(s) - np.log(np.diag(L)).sum() - 0.5 * (yty / s**2 - z @ z) - 0.5 * (s / sigma_prior) ** 2

    s0 = np.sqrt(max((yty - Xty @ np.linalg.solve(XtX + np.eye(p), Xty)) / n, 1e-12))
    grid = np.exp(np.linspace(np.log(s0) - 0.2, np.log(s0) + 0.2, 4001))
    mode = grid[np.argmax([logp(s) for s in grid])]
    sg = np.exp(np.log(mode) + np.linspace(-span, span, n_grid) / np.sqrt(2.0 * n))
    lw = np.array([logp(s) for s in sg]) + np.log(sg)
    w = np.exp(lw - logsumexp(lw))
    covs = np.array([np.linalg.inv(XtX / s**2 + np.eye(p)) for s in sg])
    means = np.array([c @ Xty / s**2 for c, s in zip(covs, sg)])
    m = w @ means
    V = np.einsum("g,gij->ij", w, covs) + np.einsum("g,gi,gj->ij", w, means - m, means - m)
    return m, V, float(w @ sg)


def beta1_block(m, V):
    sl = slice(K + 2, 2 * K + 2)
    return m[sl], V[sl, sl]


def test_from_moments(mean_b1, cov_b1, tau=0.1):
    """(shape_diff, stat, df, p) of the parallel-curves test from the mean and covariance of beta_1."""
    return M.parallel_curves_test(mean_b1, M.shape_metric(K, 3), tau=tau, cov=cov_b1)


def build_model(B, y, c, init):
    """The package's per-cell model, without the per-row `mu` in the trace (as the variational path builds it)."""
    return M._build_gene_model(y, B, B * c[:, None], c, K, init, "normal", None, record_mu=False)
