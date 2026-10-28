import numpy as np
from proto import *
rng = np.random.default_rng(0)
xf = np.linspace(0, 1, 102)
B, info = make_basis(xf)
Q = slope_metric(info); R = shape_metric(info)
print("Q eig:", np.round(np.linalg.eigvalsh(Q), 4)[:3], "... rank", (np.linalg.eigvalsh(Q) > 1e-8).sum(), " Q@1 =", np.abs(Q @ np.ones(10)).max().round(10))
xg = np.linspace(0, 1, 2001); Bg = eval_basis(info, xg)
for name, d in {"0.3*x": 0.3 * xg, "crossing 0.3*(x-.5)": 0.3 * (xg - .5), "level 0.5": 0.5 + 0 * xg, "0.2*sin(2pi x)": 0.2 * np.sin(2 * np.pi * xg)}.items():
    b1, *_ = np.linalg.lstsq(Bg, d - d.mean(), rcond=None)
    print(f"{name:22s} E_slope = {np.sqrt(b1 @ Q @ b1):.4f}  analytic RMS slope = {np.sqrt(np.mean(np.gradient(d, xg)**2)):.4f}")
# generalized chi-square: SPA vs Imhof vs MC
for lam, nc, x in [([3, 1, .5, .2], [4, 0, 0, 0], 30), ([3, 1, .5, .2], [4, 0, 0, 0], 80), (np.linspace(2, .1, 9), np.r_[9, np.zeros(8)], 200)]:
    lam, nc = np.array(lam, float), np.array(nc, float)
    draws = (lam * stats.ncx2.rvs(1, np.maximum(nc, 1e-12), size=(2_000_000, len(lam)), random_state=1)).sum(1)
    print(f"x={x}: SPA {gchisq_sf_spa(x, lam, nc):.3e}  Imhof {gchisq_sf_imhof(x, lam, nc):.3e}  MC {np.mean(draws > x):.3e}")
# least-favourable direction: maximise tail over random directions on the sphere vs top eigenvector
A = rng.normal(size=(9, 9)); Om = A @ A.T / 9
om, U = np.linalg.eigh(Om); om, U = om[::-1], U[:, ::-1]
tau = 2.0
for x in [20, 40, 80]:
    top = gchisq_sf_spa(x, om, np.r_[tau**2 / om[0], np.zeros(8)])
    best = 0
    for _ in range(300):
        u = rng.normal(size=9); u /= np.linalg.norm(u)
        m = tau * (U.T @ u)  # mean in eigen-coordinates
        best = max(best, gchisq_sf_imhof(x, om, m**2 / om))
    print(f"x={x}: top-eigvec tail {top:.4e}  best of 300 random directions {best:.4e}")
