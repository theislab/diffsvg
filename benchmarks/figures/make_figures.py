"""Poster figures from the benchmark results (prototype/all-prs).

    python make_figures.py --seaad DIR_WITH_v2_RESULTS
Writes PNG (300 dpi) and PDF into benchmarks/figures/out/.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BENCH = HERE.parent
OUT = HERE / "out"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair (dataviz palette slots 1-2)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

p = argparse.ArgumentParser()
p.add_argument("--seaad", default=os.path.expanduser("~/Documents/data/diffsvg/cluster_results/v2"))
p.add_argument("--legnini", default=os.path.expanduser("~/Documents/GitHub/diffsvg-fix/benchmarks/fix_validation/results"))
args = p.parse_args()
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})  # fmt: skip


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def share(files, col):
    out = []
    for f in files:
        d = pd.read_csv(f, index_col=0)
        out.append(float((d[col] < 0.05).mean()))
    return out


# ---------------------------------------------------------------- 1. calibration on random replicate splits
rows = []
liver = BENCH / "liver" / "results"
rows.append(("Liver MERFISH\n7 donors, 313 genes", share(sorted(glob.glob(str(liver / "null*_pooled_slope_tau0.csv"))), "slope_pval_adj"),
             share(sorted(glob.glob(str(liver / "null*_donor_slope_tau0.csv"))), "slope_pval_adj")))
leg = Path(args.legnini)
rows.append(("Organoids (Legnini)\n11 slides, 88 genes", share(sorted(glob.glob(str(leg / "new_legnini_null*_default.csv"))), "parallel_pval_adj"),
             share(sorted(glob.glob(str(leg / "new_legnini_null*_parallel_slides.csv"))), "parallel_pval_adj")))
sea = Path(args.seaad) / "whole"
if sea.exists():
    rows.append(("SEA-AD MERFISH\n27 donors, 140 genes", share(sorted(glob.glob(str(sea / "null*_pooled*.csv"))), "parallel_pval_adj"),
                 share(sorted(glob.glob(str(sea / "null*_donor_tau0_slope.csv"))), "slope_pval_adj")))
fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(rows))
w = 0.34
for i, (label, pooled, aware) in enumerate(rows):
    for offset, vals, color in ((-w / 2, pooled, ORANGE), (w / 2, aware, BLUE)):
        ax.bar(i + offset, np.mean(vals) * 100, width=w - 0.04, color=color, zorder=2)
        ax.scatter(np.full(len(vals), i + offset), np.array(vals) * 100, s=10, color=INK, zorder=3, linewidth=0)
        ax.text(i + offset, max(np.max(vals), np.mean(vals)) * 100 + 3, f"{np.mean(vals) * 100:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x, [r[0] for r in rows])
ax.set_ylim(0, 100)
ax.set_ylabel("genes called at FDR < 0.05 (%)")
ax.set_title("Random splits of the replicates (no true difference)", fontsize=9, color=INK, loc="left")
from matplotlib.patches import Patch

ax.legend(handles=[Patch(color=ORANGE, label="cells as replicates (pooled)"), Patch(color=BLUE, label="replicate effects (diffsvg)")],
          loc="upper right", fontsize=8)  # fmt: skip
save(fig, "fig1_calibration")

# ---------------------------------------------------------------- 2. liver zonation curves
stem = liver / "disease_donor_slope_tau0"
stats = pd.read_csv(f"{stem}.csv", index_col=0)
cur = {k: pd.read_csv(f"{stem}_{k}.csv", index_col=0) for k in
       ("cond0_mean", "cond0_ci_low", "cond0_ci_high", "cond1_mean", "cond1_ci_low", "cond1_ci_high")}  # fmt: skip
um = cur["cond0_mean"].columns.astype(float).to_numpy() * 500
genes = ["CYP2E1", "CYP1A2", "CYP3A4", "SDS", "ASS1"]
fig, axes = plt.subplots(1, len(genes), figsize=(9.5, 2.3), sharex=True)
for ax, g in zip(axes, genes, strict=True):
    for c, color, label in (("cond1", BLUE, "healthy (3 donors)"), ("cond0", ORANGE, "fibrotic (4 donors)")):
        ax.fill_between(um, cur[f"{c}_ci_low"].loc[g], cur[f"{c}_ci_high"].loc[g], color=color, alpha=0.18, linewidth=0)
        ax.plot(um, cur[f"{c}_mean"].loc[g], color=color, lw=2, label=label)
    r = stats.loc[g]
    ax.set_title(f"{g}\nslope healthy − fibrotic {r.slope_diff:+.2f} [{r.slope_diff_ci_low:+.2f}, {r.slope_diff_ci_high:+.2f}]", fontsize=8, color=INK)
    ax.set_xlabel("distance to portal tract (µm)")
axes[0].set_ylabel("expression (log1p)")
axes[0].legend(loc="upper left", fontsize=7)
fig.suptitle("Hepatocyte zonation in healthy vs fibrotic liver (population curves, 94% credible bands)", fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout()
save(fig, "fig2_liver_zonation")

# ---------------------------------------------------------------- 3. liver ranking vs random splits
real = stats.sort_values("slope_pval")
null_min = [pd.read_csv(f, index_col=0)["slope_pval_adj"].min() for f in sorted(glob.glob(str(liver / "null*_donor_slope_tau0.csv")))]
fig, ax = plt.subplots(figsize=(4.2, 2.8))
from scipy.stats import t as student_t

top = real.head(12).iloc[::-1]
# 94% t interval on the test's own degrees of freedom (n_donors - 2), so the interval and the p-value agree
se = top.slope_diff.abs() / np.sqrt(top.slope_stat)
half = student_t.ppf(0.97, top.slope_df_denominator) * se
ax.errorbar(top.slope_diff, np.arange(len(top)), xerr=half, fmt="o", color=BLUE, ms=4, lw=1.2, capsize=0)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(np.arange(len(top)), [f"{g}  (FDR {q:.2f})" for g, q in zip(top.index, top.slope_pval_adj, strict=True)], fontsize=7)
ax.set_xlabel("slope difference, healthy − fibrotic (94% t interval, 5 df)")
ax.set_title(f"Top genes by slope test; best FDR in 10 random donor splits: {min(null_min):.2f}", fontsize=8, color=INK, loc="left")
save(fig, "fig3_liver_ranking")

# ---------------------------------------------------------------- 4. power (SEA-AD spike-ins, v2)
pw = Path(args.seaad) / "power"
files = sorted(glob.glob(str(pw / "*.csv")))
if files:
    d = []
    for f in files:
        kind = Path(f).stem.split("_")[0]
        t = pd.read_csv(f, index_col=0)
        col = "level_pval_adj" if kind == "level" else "parallel_pval_adj"
        sd = t["sample_level_sd"] if kind == "level" else t["sample_curve_sd"]
        d.append(pd.DataFrame({"kind": kind, "effect": t["effect"], "hit": t[col] < 0.05, "sd": sd}))
    d = pd.concat(d)
    d["stable"] = d.groupby("kind")["sd"].transform(lambda s: s <= s.median())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, kind, title in ((axes[0], "level", "level test"), (axes[1], "shape", "shape test (τ = 0)")):
        for stable, color, label in ((True, BLUE, "donor-stable genes"), (False, ORANGE, "donor-variable genes")):
            s = d[(d.kind == kind) & (d.stable == stable) & (d.effect > 0)].groupby("effect")["hit"].mean()
            ax.plot(s.index, s.values * 100, "o-", color=color, lw=2, ms=4, label=label)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("spiked effect (log1p units)")
    axes[0].set_ylabel("detected at FDR < 0.05 (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Power on SEA-AD with real donor variation (27 donors)", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "fig4_power")
print("figures in", OUT)

# ---------------------------------------------------------------- 5. liver per-donor slopes
ds = pd.read_csv(liver / "donor_slopes.csv")
genes = ["CYP2E1", "CYP1A2", "CYP3A4", "SDS", "ASS1"]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
rng = np.random.default_rng(0)
for i, g in enumerate(genes):
    for cond, color, off in (("healthy", BLUE, -0.15), ("diseased", ORANGE, 0.15)):
        v = ds.loc[ds.condition == cond, g].to_numpy()
        ax.scatter(i + off + rng.uniform(-0.04, 0.04, len(v)), v, s=28, color=color, edgecolor="white", linewidth=1, zorder=3)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(len(genes)), genes)
ax.set_ylabel("zonation slope per donor\n(log1p per 500 µm)")
ax.set_title("One dot per donor: zonation gradients are flatter in fibrotic livers", fontsize=9, color=INK, loc="left")
ax.legend(handles=[Patch(color=BLUE, label="healthy (3)"), Patch(color=ORANGE, label="fibrotic (4)")], fontsize=7, loc="lower left")
save(fig, "fig5_liver_donor_slopes")
