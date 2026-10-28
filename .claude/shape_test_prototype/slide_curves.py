"""Slide-aware model with a full random curve per slide (random intercept + slide x spline), empirical Bayes."""
import numpy as np
from scipy import optimize


def fit_slide_curves(Yb, Xf, Z_int, B, slide_idx, w, eta=False):
    """Fixed-effect posterior with random intercepts (sd_a) and random spline curves (sd_u) per slide integrated out.

    eta=True adds a per-(slide, bin) extra variance on top of sigma^2 / n (bin-level overdispersion).
    """
    S, K = Z_int.shape[1], B.shape[1]
    Zc = np.concatenate([B * (slide_idx == s)[:, None] for s in range(S)], axis=1)
    X = np.column_stack([Xf, Z_int, Zc])
    pf = Xf.shape[1]
    n = len(w)
    out_m, out_c, out_par = [], [], []
    XtWX0 = X.T @ (X * w[:, None])
    for g in range(Yb.shape[1]):
        y = Yb[:, g]

        def post(t):
            lsa, lsu, lsig = t[:3]
            prec = np.r_[np.ones(pf), np.full(S, np.exp(-2 * lsa)), np.full(S * K, np.exp(-2 * lsu))]
            if eta:
                v = np.exp(2 * lsig) / w + np.exp(2 * t[3])  # per-row variance
                ww = 1 / v
                XtWX = X.T @ (X * ww[:, None])
                logdet_noise = np.log(v).sum()
            else:
                ww = w / np.exp(2 * lsig)
                XtWX = XtWX0 / np.exp(2 * lsig)
                logdet_noise = n * 2 * lsig - np.log(w).sum()
            Lam = XtWX + np.diag(prec)
            C = np.linalg.cholesky(Lam)
            b = X.T @ (ww * y)
            m = np.linalg.solve(Lam, b)
            logdet = 2 * np.log(np.diag(C)).sum() - np.log(prec).sum() + logdet_noise
            quad = float(y @ (ww * y)) - b @ m
            return -0.5 * (logdet + quad), m, Lam

        x0 = np.log([0.2, 0.1, 1.0] + ([0.05] if eta else []))
        res = optimize.minimize(lambda t: -post(t)[0], x0=x0, method="L-BFGS-B",
                                bounds=[(-12, 3)] * len(x0))
        _, m, Lam = post(res.x)
        out_m.append(m[:pf])
        out_c.append(np.linalg.inv(Lam)[:pf, :pf])
        out_par.append(np.exp(res.x))
    return np.array(out_m), np.array(out_c), np.array(out_par)
