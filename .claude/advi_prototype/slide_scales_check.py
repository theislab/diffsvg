"""Both slide-scale options of inference='advi' against NUTS, through fit_model, on simulated slides (no effect / a shape effect)."""

import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, "/home/lehnerl/Arbeit/github/HiWi/diffsvg/tests")
warnings.filterwarnings("ignore")
import diffsvg as dvg  # noqa: E402
from test_samples import _slides  # noqa: E402

COLS = ["shape_diff", "parallel_stat", "parallel_pval", "sample_level_sd", "sample_curve_sd"]
rows = []
for scenario, effect in [("no effect", 0.0), ("shape effect 1.0", 1.0)]:
    adata = _slides(shape_effect=effect, seed=0)
    dvg.pp.bin_obs(adata, anchor="anchor", condition="condition", n_bins=20, window_radius=None, sample="slide")
    setup = dvg.tl.setup_model(adata)
    for label, kw in [
        ("NUTS bins", {"chains": 4, "draws": 1000, "tune": 1000}),
        ("ADVI empirical_bayes bins", {"inference": "advi"}),
        ("ADVI empirical_bayes cells", {"inference": "advi", "use_binned": False}),
        ("ADVI data_start bins", {"inference": "advi", "slide_scales": "data_start"}),
        ("ADVI data_start bins lr 1e-2 n 6000", {"inference": "advi", "slide_scales": "data_start", "learning_rate": 1e-2, "n": 6000}),
        ("ADVI data_start cells lr 1e-2 n 6000", {"inference": "advi", "slide_scales": "data_start", "learning_rate": 1e-2, "n": 6000, "use_binned": False}),
    ]:
        t = time.time()
        stats, _, _ = dvg.tl.fit_model(adata, setup, test="parallel", tau=0.0, inplace=False, progressbar=False, **kw)
        r = stats.iloc[0]
        rows.append({"scenario": scenario, "fit": label, "seconds": round(time.time() - t), **{c: r[c] for c in COLS},
                     "vi_khat": r.get("vi_khat"), "vi_ess": r.get("vi_ess"), "rhat_max": r.get("rhat_max")})
        print(rows[-1], flush=True)
pd.set_option("display.width", 250)
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
out.to_csv("results/slide_scales_check.csv", index=False)
