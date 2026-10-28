import pickle, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import harness  # noqa: F401  (seeds the posterior predictive)
import diffsvg as dvg
from make_data import make_adata

SK = {"draws": 300, "tune": 300, "chains": 2}
out = {}
adata = make_adata()
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
for use_binned in (True, False):
    out[use_binned] = dvg.tl.fit_model(adata, setup, inplace=False, use_binned=use_binned, progressbar=False, sample_kwargs=SK)
pickle.dump(out, open(sys.argv[1], "wb"))

# per-cell curves vs the truth (gene "same": both conditions follow 0.6 exp(-3x))
x = np.linspace(0, 1, 51)
truth = 0.6 * np.exp(-3 * x)
for ub in (True, False):
    c0 = out[ub][2]["cond0_mean"].loc["same"].to_numpy()
    c1 = out[ub][2]["cond1_mean"].loc["same"].to_numpy()
    print(f"use_binned={ub}: gene 'same', mean(fitted - truth): cond0 {np.mean(c0 - truth):+.3f}, cond1 {np.mean(c1 - truth):+.3f}")

if len(sys.argv) > 2:  # missing anchors, per cell
    a2 = make_adata(nan_frac=0.05, seed=1)
    dvg.pp.bin_obs(a2, anchor="anchor", condition="condition")
    try:
        dvg.tl.fit_model(a2, dvg.tl.setup_model(a2), inplace=False, use_binned=False, progressbar=False, sample_kwargs=SK)
        print("per-cell fit with 5% missing anchors: ran")
    except Exception as e:
        print(f"per-cell fit with 5% missing anchors: {type(e).__name__}: {str(e)[:200]}")
