"""Real-data null: split the SHH slides of legnini into two random halves and test 'condition' = half.

Also runs the real Ctrl-vs-SHH comparison for reference. log1p_norm layer, anchor = distance to SHH+ cells.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd

from proto import beta1_block, eval_basis, fit_closed_form, make_basis, shape_metric, slope_test
from sim_slides import contiguous_bins, fit_slide_aware

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
ALPHA, TAU = 0.05, 0.1

adata = ad.read_h5ad("legnini_dm.h5ad")
dm = adata.obsm["design_matrix"]
keep = dm["True"].notna().to_numpy()
x = dm["True"].to_numpy(dtype=float)[keep]
sample = adata.obs["sample"].astype(str).to_numpy()[keep]
cond = adata.obs["condition"].astype(str).to_numpy()[keep]
Y = adata.layers["log1p_norm"][keep]
Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y, dtype=float)
genes = adata.var_names.to_numpy()
slides = np.unique(sample)
slide_cond = {s: cond[sample == s][0] for s in slides}
print(f"{len(slides)} slides with an anchor: {sum(v == 'Ctrl' for v in slide_cond.values())} Ctrl, "
      f"{sum(v == 'SHH' for v in slide_cond.values())} SHH; {len(genes)} genes")


def analyse(mask, group_of_slide):
    xs, ss = x[mask], sample[mask]
    sl_idx = np.searchsorted(slides, ss)
    grp = np.array([group_of_slide[s] for s in slides], dtype=float)
    c = grp[sl_idx]
    Ym = Y[mask]
    cb, xb, nb, Yb = contiguous_bins(xs, c.astype(int), Ym)
    Bm, info = make_basis(xb)
    Qa = shape_metric(info) * 12
    mean, cov, sig, K = fit_closed_form(Yb, Bm, cb.astype(float), weights=nb)
    m1, V1 = beta1_block(mean, cov, K)
    E_pool, _, pp_pool, pi_pool = slope_test(m1, V1, Qa, TAU)

    sb, xb2, nb2, Yb2 = contiguous_bins(xs, sl_idx, Ym)
    used = np.unique(sb)
    remap = {s_: i for i, s_ in enumerate(used)}
    sb = np.array([remap[s_] for s_ in sb])
    cb2 = grp[used][sb]
    B2 = eval_basis(info, xb2)
    Xf = np.column_stack([np.ones(len(xb2)), B2, cb2, B2 * cb2[:, None]])
    Zi = (sb[:, None] == np.arange(len(used))[None]).astype(float)
    Zs = Zi * (xb2 - 0.5)[:, None]
    mm, cc = fit_slide_aware(Yb2, Xf, Zi, Zs, nb2.astype(float), None)
    s_ = slice(K + 2, 2 * K + 2)
    E_sl, _, pp_sl, pi_sl = slope_test(mm[:, s_], cc[:, s_, s_], Qa, TAU)
    return pd.DataFrame({"E_pool": E_pool, "p_pool_point": pp_pool, "p_pool_int": pi_pool,
                         "E_slide": E_sl, "p_slide_point": pp_sl, "p_slide_int": pi_sl}, index=genes)


shh = np.array([s for s in slides if slide_cond[s] == "SHH"])
mask_shh = np.isin(sample, shh)
null_runs = []
for r in range(8):
    half = set(rng.choice(shh, len(shh) // 2, replace=False))
    res = analyse(mask_shh, {s: float(s in half) for s in slides})
    null_runs.append(res)
    print(f"split {r}: rejected at 0.05 -> pooled tau=0 {np.mean(res.p_pool_point < ALPHA):.3f}, pooled shape<=tau "
          f"{np.mean(res.p_pool_int < ALPHA):.3f} | slide-aware tau=0 {np.mean(res.p_slide_point < ALPHA):.3f}, "
          f"slide-aware shape<=tau {np.mean(res.p_slide_int < ALPHA):.3f}")
allnull = pd.concat(null_runs)
print("\nSHH-half splits, all runs:", {k: round(float(np.mean(allnull[k] < ALPHA)), 3) for k in
                                       ["p_pool_point", "p_pool_int", "p_slide_point", "p_slide_int"]})
print("median E (slope-equivalent) under the split null: pooled", allnull.E_pool.median().round(3),
      " slide-aware", allnull.E_slide.median().round(3))

real = analyse(np.ones(len(x), bool), {s: float(slide_cond[s] == "SHH") for s in slides})
print("\nReal Ctrl vs SHH:", {k: round(float(np.mean(real[k] < ALPHA)), 3) for k in
                              ["p_pool_point", "p_pool_int", "p_slide_point", "p_slide_int"]})
print(real.sort_values("p_slide_int").head(12).round(4).to_string())
real.to_csv("legnini_real_slope_test.csv")
