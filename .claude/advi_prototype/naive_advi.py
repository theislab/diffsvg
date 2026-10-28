"""Why inference='advi' starts at a Laplace approximation: plain pm.fit on the per-cell model, against the exact posterior."""

import time

import numpy as np
import pymc as pm
import pytensor

import lab

GENES = ["GLI1", "VAMP2"]


def kl(m1, V1, m0, V0):
    """KL(N1 || N0)."""
    L0, L1, d = np.linalg.cholesky(V0), np.linalg.cholesky(V1), None
    d = np.linalg.solve(L0, m1 - m0)
    return 0.5 * (np.trace(np.linalg.solve(V0, V1)) + d @ d - len(m1) + 2 * (np.log(np.diag(L0)).sum() - np.log(np.diag(L1)).sum()))


def fit(model, method, n, lr0=None, lr1=None):
    """pm.fit with PyMC's default optimiser (lr0=None) or Adam with an exponential learning-rate decay."""
    kw = {}
    if lr0 is not None:
        lr = pytensor.shared(np.float64(lr0))
        kw = dict(
            obj_optimizer=pm.adam(learning_rate=lr),
            callbacks=[lambda a, loss, i: lr.set_value(np.float64(lr0 * (lr1 / lr0) ** (i / n)))],
        )
    with model:
        return pm.fit(n=n, method=method, random_seed=0, progressbar=False, **kw)


if __name__ == "__main__":
    adata, setup = lab.load(GENES)
    B, _, c, Y, init = lab.per_cell(adata, setup)
    X = lab.design(B, c)
    for g in GENES:
        mb, Vb = lab.beta1_block(*lab.exact_posterior(X, Y[g])[:2])
        _, W0, _, p0 = lab.test_from_moments(mb, Vb)[0:4]
        print(f"\n{g}: exact W = {W0:.1f}, p(tau=0.1) = {p0:.3g}")
        model = lab.build_model(B, Y[g], c, init[g])
        for label, method, n, lr0, lr1 in [
            ("mean-field, PyMC default, 10k steps", "advi", 10000, None, None),
            ("full-rank, PyMC default, 10k steps", "fullrank_advi", 10000, None, None),
            ("mean-field, Adam .01 -> 1e-4, 10k", "advi", 10000, 0.01, 1e-4),
            ("full-rank, Adam .05 -> 1e-4, 30k", "fullrank_advi", 30000, 0.05, 1e-4),
        ]:
            t = time.time()
            approx = fit(model, method, n, lr0, lr1)
            o = approx.groups[0].ordering
            mu, cov = approx.mean.eval(), approx.cov.eval()
            s = o["beta_1"][1]
            E, W, _, p = lab.test_from_moments(mu[s], cov[s, s])
            print(f"  {label:38s} {time.time() - t:5.0f}s  W/exact {W / W0:7.3f}  p {p:8.3g}  KL(beta_1) {kl(mu[s], cov[s, s], mb, Vb):9.3g}")
