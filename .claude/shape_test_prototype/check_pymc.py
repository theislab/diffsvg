import numpy as np, pymc as pm, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("pymc").setLevel(logging.ERROR)
from proto import *
from diffsvg.tl.model import _build_gene_model
rng = np.random.default_rng(3)
# simulated cells -> current binning (overlapping windows)
N = 6000
x = np.where(rng.random(2 * N) < 0.8, rng.beta(1.5, 2.5, size=2 * N), rng.random(2 * N)); c = np.r_[np.zeros(N), np.ones(N)]
f = lambda x, c: 0.5 * np.exp(-3 * x) + c * (0.2 + 0.4 * x)
Y = (f(x, c) + rng.normal(0, 1, size=2 * N))[:, None]
xb, cb, nb, Yb = bin_means(x, c, Y)
B, info = make_basis(xb)
mean, cov, sig, K = fit_closed_form(Yb, B, cb)
m1, V1 = beta1_block(mean, cov, K)
Q = slope_metric(info)
print("closed form: sigma", sig.round(4), " E_slope", np.sqrt(m1[0] @ Q @ m1[0]).round(4), "(true 0.4)")
model = _build_gene_model(Yb[:, 0], B, B * cb[:, None], cb, K, float(Yb[cb == 0][:3].mean()))
with model:
    idata = pm.sample(draws=2000, tune=1000, chains=4, nuts_sampler="nutpie", random_seed=1, progressbar=False)
post = idata.posterior["beta_1"].stack(s=("chain", "draw")).to_numpy().T
print("pymc:        sigma", float(idata.posterior["sigma"].mean()).__round__(4), " E_slope", np.sqrt(post.mean(0) @ Q @ post.mean(0)).round(4))
print("beta_1 mean  max |pymc - closed| =", np.abs(post.mean(0) - m1[0]).max().round(4), " (posterior sd ~", np.sqrt(np.diag(V1[0])).mean().round(3), ")")
Vp = np.cov(post, rowvar=False)
L = np.linalg.cholesky(Q + 1e-12 * np.eye(10))
print("var of slope functional: pymc", float(post.mean(0) @ Q @ Vp @ Q @ post.mean(0)).__round__(6), " closed", float(m1[0] @ Q @ V1[0] @ Q @ m1[0]).__round__(6))
for nm, (bm, bc) in {"closed": (m1, V1), "pymc": (post.mean(0)[None], Vp[None])}.items():
    E, W, pp, pi = slope_test(bm, bc, Q, tau=0.2)
    print(f"{nm:6s}: E={E[0]:.4f}  W={W[0]:.2f}  p(point)={pp[0]:.3e}  p(E<=0.2)={pi[0]:.3e}")
