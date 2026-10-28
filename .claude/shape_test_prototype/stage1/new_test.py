import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")
import harness  # noqa: F401
import diffsvg as dvg
from make_data import make_adata
from proto import fit_closed_form

pd.set_option("display.width", 220)
SK = {"draws": 1000, "tune": 500, "chains": 2}
adata = make_adata()
dvg.pp.bin_obs(adata, anchor="anchor", condition="condition")
setup = dvg.tl.setup_model(adata)
cols = ["shape_diff", "slope_diff", "slope_diff_ci_low", "slope_diff_ci_high", "parallel_stat", "parallel_df", "parallel_pval", "parallel_pval_adj"]
res = {}
for ub in (True, False):
    for tau in (0.1, 0.0):
        s, _, _ = dvg.tl.fit_model(adata, setup, inplace=False, use_binned=ub, progressbar=False, sample_kwargs=SK, test="parallel", tau=tau)
        res[(ub, tau)] = s
        print(f"\n--- use_binned={ub}, tau={tau}: attrs {s.attrs}; columns: {list(s.columns)}")
        print(s[cols].round(4).to_string())

# closed-form prototype on the same cells, same basis -> must match the PyMC per-cell numbers
b = setup.binned
x = adata.obsm["design_matrix"]["anchor"].to_numpy(float)
c = (adata.obsm["design_matrix"]["condition"] == "treat").to_numpy(float)
B = dvg.pp.spline_basis_at(b, x)
mean, cov, sig, K = fit_closed_form(np.asarray(adata.X, float), B, c)
Q = dvg.tl.shape_metric(b.n_bsplines, b.degree)
s_ = slice(K + 2, 2 * K + 2)
print("\n--- per cell, PyMC vs closed form (tau = 0.1):")
for g, gene in enumerate(adata.var_names):
    E, W, df, p = dvg.tl.parallel_curves_test(mean[g, s_], Q, tau=0.1, cov=cov[g, s_, s_])
    r = res[(False, 0.1)].loc[gene]
    print(f"  {gene:9s} shape_diff {r.shape_diff:.4f} vs {E:.4f} | stat {r.parallel_stat:7.2f} vs {W:7.2f} | p {r.parallel_pval:.2e} vs {p:.2e}")
