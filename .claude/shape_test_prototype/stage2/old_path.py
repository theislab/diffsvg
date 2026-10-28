import pickle, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import harness_force  # noqa: F401
import diffsvg as dvg
from make_data import make_adata

print("diffsvg from", dvg.__file__)
SK = {"draws": 300, "tune": 300, "chains": 2}
out = {}
adata = make_adata()
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
out["binned_uns"] = {k: v for k, v in adata.uns["binned"].items()}
for test in ("wald", "parallel"):
    out[test] = dvg.tl.fit_model(adata, setup, inplace=False, use_binned=True, progressbar=False, sample_kwargs=SK, test=test)
a2 = make_adata(nan_frac=0.05, seed=1)
out["nan_binned"] = dvg.pp.bin_obs(a2, anchor="anchor", condition="condition", inplace=False)
pickle.dump(out, open(sys.argv[1], "wb"))
