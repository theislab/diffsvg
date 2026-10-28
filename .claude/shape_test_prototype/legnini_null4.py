import warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
exec(open("legnini_null.py").read().split("def analyse")[0])  # data loading only
from slide_curves import fit_slide_curves
from proto import slope_test_small_sample
import os
ETA = os.environ.get("ETA") == "1"

def analyse2(mask, group_of_slide):
    xs, ss = x[mask], sample[mask]
    sl_idx = np.searchsorted(slides, ss)
    grp = np.array([group_of_slide[s] for s in slides], dtype=float)
    cb, xb, nb, Yb = contiguous_bins(xs, grp[sl_idx].astype(int), Y[mask])
    Bm, info = make_basis(xb)
    Qa = shape_metric(info) * 12
    sb, xb2, nb2, Yb2 = contiguous_bins(xs, sl_idx, Y[mask])
    used = np.unique(sb); sb = np.searchsorted(used, sb)
    cb2 = grp[used][sb]
    B2 = eval_basis(info, xb2)
    K = B2.shape[1]
    Xf = np.column_stack([np.ones(len(xb2)), B2, cb2, B2 * cb2[:, None]])
    Zi = (sb[:, None] == np.arange(len(used))[None]).astype(float)
    mm, cc, par = fit_slide_curves(Yb2, Xf, Zi, B2, sb, nb2.astype(float), eta=ETA)
    s_ = slice(K + 2, 2 * K + 2)
    E, W, pp, pi = slope_test(mm[:, s_], cc[:, s_, s_], Qa, TAU)
    from scipy import stats as st
    dfd = len(used) - 2
    _, _, pF, pIntF = slope_test_small_sample(mm[:, s_], cc[:, s_, s_], Qa, TAU, dfd)
    return pd.DataFrame({"E": E, "p_point": pp, "p_F": pF, "p_int": pi, "p_intF": pIntF, "sd_a": par[:, 0], "sd_u": par[:, 1], "sigma": par[:, 2]}
                        | ({"sd_eta": par[:, 3]} if ETA else {}), index=genes)

shh = np.array([s for s in slides if slide_cond[s] == "SHH"])
mask_shh = np.isin(sample, shh)
runs = []
for r in range(8):
    half = set(rng.choice(shh, len(shh) // 2, replace=False))   # same splits as legnini_null.py (same seed/order)
    res = analyse2(mask_shh, {s: float(s in half) for s in slides})
    runs.append(res)
    print(f"split {r}: tau=0 chi2 {np.mean(res.p_point < ALPHA):.3f}  tau=0 F {np.mean(res.p_F < ALPHA):.3f}   shape<=tau chi2 {np.mean(res.p_int < ALPHA):.3f}  shape<=tau F {np.mean(res.p_intF < ALPHA):.3f}", flush=True)
alln = pd.concat(runs)
print("all splits:", {k: round(float(np.mean(alln[k] < ALPHA)), 3) for k in ["p_point", "p_F", "p_int", "p_intF"]},
      " median E", alln.E.median().round(3))
print("estimated variance components (median over genes x splits):", alln.drop(columns=["E", "p_point", "p_F", "p_int", "p_intF"]).median().round(4).to_dict())
q = np.sort(alln.p_F.to_numpy())
print("p_F quantiles (uniform would be .1 .25 .5):", np.quantile(q, [.1, .25, .5]).round(3))
real = analyse2(np.ones(len(x), bool), {s: float(slide_cond[s] == "SHH") for s in slides})
print("\nreal Ctrl vs SHH:", {k: round(float(np.mean(real[k] < ALPHA)), 3) for k in ["p_point", "p_F", "p_int", "p_intF"]})
print(real.sort_values("p_intF").head(15).round(4).to_string())
real.to_csv("legnini_real_final.csv"); alln.to_csv("legnini_splits_final.csv")
