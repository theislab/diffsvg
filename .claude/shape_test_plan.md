# Shape (slope) difference test: staged plan

Goal: rank genes whose curves differ between two groups of cells (two slides, or two groups of
slides) in **shape/slope**, not merely in level, with
p-values that are (1) per gene, not built from a null fitted across the panel, (2) not driven to 0
by large cell counts, and (3) not tied to one pattern (linear, peak, ...).

## The test

Per gene, `d(x) = f1(x) - f0(x)` splits into a **level** part `d_bar = int_0^1 d` (today's
`integral`) and a **shape** part `d(x) - d_bar`. The curves are parallel exactly when the shape part
is 0, whatever the level difference.

- Effect size `E = sqrt(12) * RMS(d - d_bar)`, "slope-equivalent": for a gap that changes linearly
  across the anchor range, `E` is that change (the slope difference); for any other shape it is the
  linear change with the same RMS departure from parallel. In coefficients `E^2 = beta_1' Q beta_1`
  with `Q = 12 * Gram(centred basis)`; `Q 1 = 0`, so level never enters.
- Hypotheses: `H0: E <= tau` (parallel up to a negligible shape difference) vs `H1: E > tau`.
  `tau = 0` is the classic parallelism test (Wald on `D beta_1`, `K - 1` df).
- Estimate/covariance: posterior mean and covariance of `beta_1` (the coefficients), **not** the
  posterior-predictive draws of `D`. Those add `2 sigma^2 I`, so today's Wald test is conservative at
  the null, flags pure level shifts, and depends on the bin construction (with contiguous
  n-weighted bins it rejected nothing at all).
- Reference distribution (as implemented in Stage 1): the Wald statistic `W = z' Omega^-1 z`
  (`z = L' beta_1_hat`, `Omega = L' V L`, `Q = L L'`) against noncentral chi-square(`K - 1`,
  `tau^2 / omega_min`), `omega_min` = smallest eigenvalue of `Omega` (the best-measured shape
  direction). Exactly valid: under `E <= tau` the noncentrality of `W` is at most that. Continuous in
  `tau`, equal to the classic test at `tau = 0`. Chosen over the prototype's version (generalized
  chi-square on `E_hat^2`, worst case on the top eigenvector) on evidence (`variants.py`, 5k
  cells/condition, alpha 0.05): identical curves 1.0% vs 2.7%, crossing 0.3 38% vs 30%, peak shift
  26% vs 10%. With replicates (Stage 3) the same idea gives a noncentral F with `n_samples - 2` df.
- Reported per gene (`test="parallel"`): `shape_diff` (`E_hat`), `slope_diff` (least-squares slope
  of `f1 - f0` over `[0, 1]`, with a 94% credible interval), `parallel_stat`, `parallel_df`,
  `parallel_pval`, `parallel_pval_adj` (BH); `integral` stays the level difference.

How it meets the three requirements:
1. Each gene's p-value uses only that gene's estimate and covariance; a single gene can be tested.
2. As data grow, `E_hat -> E`, so p only goes to 0 when `E > tau` (simulation: a 0.03 slope
   difference is flagged 2% -> 0% from 2k to 50k cells, the `tau = 0` test 6% -> 14%).
3. It covers every non-parallel pattern the spline can represent (`K - 1` dims); `tau` is the one
   knob trading sensitivity for specificity.

**Calibration needs the right variance, not just the right statistic.** In the prototype that took
contiguous bins with cell-count weights, a random curve per sample when replicates exist, and the
small-sample reference.

## Evidence (prototype, 2026-10-15, `shape_test_prototype/`)

Rerun from that folder with the `pymc_env` python: `sim.py` and `sim_slides.py` (simulations),
`inspect_legnini.py` (writes `legnini_dm.h5ad`), then `legnini_null.py` / `legnini_null4.py`;
`raw_vs_bins.py` (per-cell fit vs bins); `stage2/` the Stage 2 checks (`old_path.py` with
`harness_force.py`, `seed_and_weights.py`, `calibration_weighted.py`); `stage1/` holds the Stage 1 checks (`reference2.py` with the
seeding `harness.py`, `new_test.py`, `plots_and_stability.py`, `calibration.py`).

Closed-form stand-in for the PyMC model (same design, same N(0, 1) priors, sigma plugged in);
matches NUTS on one gene (beta_1 means within 0.004 of a ~0.3 posterior sd, test variance within 1%).
`tau = 0.1` slope-equivalent throughout. Rejection rates at alpha = 0.05:

| Setting (200 genes/scenario) | current Wald | parallel, tau=0 | shape <= tau |
|---|---|---|---|
| identical curves, today's overlapping bins | 0-2% | 76-84% | 2-63% |
| level shift only, today's bins | 90-100% | 79-85% | 1-61% |
| identical / level only, contiguous n-weighted bins | 0% | 3-7% | 0-3% |
| tiny slope 0.03, contiguous, 2k -> 50k cells | 0% | 6% -> 14% | 2% -> 0% |
| crossing slope 0.3, contiguous, 2k -> 50k cells | 0% | 24% -> 100% | 12% -> 100% |

Derivative metric (`RMS(f1' - f0')`) was also tried: calibrated but powerless (<= 9% at 3x the
threshold), because derivatives of the unpenalized spline are noise-dominated. Dropped until a
smoothing prior exists (Stage 4).

Slides (simulated 6 vs 11 slides, level sd 0.3, slope sd 0.1, slide-specific cell positions):
pooled bins flag 59% (tau=0) / 23% (shape) of identical-curve genes; a slide-aware model (random
intercept + slope per slide) gives 5% / 0%, with the same power (72-88% for a 0.3 crossing slope).

Real data, legnini (`log1p_norm`, 16 slides with an anchor), null = two random halves of the 10
SHH slides, 8 splits x 88 genes:

| Model | tau=0 (chi2) | tau=0 (F, df = slides - 2) | shape <= tau (chi2) | shape <= tau (scale-mixture) |
|---|---|---|---|---|
| pooled, contiguous bins | 82% | - | 57% | - |
| + random intercept & linear slope per slide | 60% | - | 14% | - |
| + random spline curve per slide | 11.5% | **1.6%** | 8.1% | **2.3%** |

Worst single split: 5.7% (F), 10.2% (scale-mixture); 5 vs 5 slides make splits noisy.
Per-bin overdispersion estimated at ~0 (sd 0.0015), so it is not needed.

Per-cell fits (`raw_vs_bins.py`): the test takes `beta_1` from any fit of this model. A per-cell fit
and contiguous cell-count-weighted bins give the same `E` (e.g. 0.385 vs 0.386) and, with random
slide curves, the same variance components and p (0.0017 vs 0.0017). Overlapping bins give a
residual sd of 0.05 instead of 1 and p < 1e-4 for identical curves (per cell: 0.34). Bins estimate
`sigma` from ~90 residual df instead of ~N cells, so `W` scatters about +-20% around the per-cell
value; storing each bin's within-bin sum of squares would remove that. Real Ctrl vs SHH with the
slide-curve model: 32% of genes at F-p < 0.05 (31% for `shape <= tau`), led by HHIP, GLI1, PTCH1, FOXA1/2, MSX2, PAX7,
NEUROG2, DBX1/2 (SHH pathway and patterning genes).

## Decisions and caveats

- Found in Stage 1: the posterior-predictive draws were unseeded (fixed in Stage 2 by
  `predictive_seed`). Still open: the `cond*_ci_*` bands are quantiles of predictive draws, i.e. they
  include observation noise - bin-mean noise for binned fits, single-cell noise (very wide) for per-cell fits.

- Shape metric, not derivative (see above).
- The test compares two groups of cells. With one slide per group (Xenium example) it is a
  slide-vs-slide test, and whether that equals a condition effect is left to downstream
  interpretation (user, 2026-10-15). There the precision grows with cell numbers, so `tau` alone
  keeps negligible differences from becoming significant. With several slides per group, Stage 3
  adds the between-slide variance, which caps the precision at the number of slides.
- Scale: shape is judged on the modelled scale. On linear expression (the morphogen notebook fits
  `X`), a fold change of a gene with a gradient is a shape difference; on log1p it is a level shift.
  The N(0, 1) priors also assume O(1)-scaled data.
- `tau` default: 0.1 slope-equivalent (on log1p: the condition fold change drifts by ~10% across
  the anchor range). To be confirmed.

## Stages

| Stage | What | Verified by | Status |
|---|---|---|---|
| 0 | Prototype (closed form) | simulations + legnini split null above | done |
| 1 | `fit_model(..., test="parallel", tau=0.1)`: `parallel_curves_test` on the `beta_1` draws, binned and per cell; `shape_metric` public; plots pick the new columns; `check_stability` passes `test`/`tau` through. Per-cell inputs fixed on the way: the cell basis is now the binned basis evaluated per cell (`pp.spline_basis_at`; before, its centring shifted every per-cell curve by ~10% of its range), and cells without anchor/condition are dropped (before: crash) | default Wald path bit-identical on a seeded run (binned stats, preds, curves); per-cell PyMC matches the closed form (`shape_diff` within 0.005, stat within ~4%); 8 unit tests incl. exact calibration at the least-favourable point; calibration through PyMC (3000 cells/condition, 80 genes): per cell, null genes (30 identical + 30 level-only) flagged 0/60 at tau=0 and tau=0.1, crossing 0.4 90%/70%, peak shift 70%/60%; today's overlapping bins flag 90%/73% (identical) and 63%/37% (level), so binned p-values need Stage 2 | done |
| 2 | `bin_obs(..., window_radius=None)`: every cell in exactly its nearest bin, cell counts per row/grid point and per-gene within-bin sum of squares stored (`BinnedData.weighted`); fits then use `sigma / sqrt(n_obs)` plus a `Potential` for the within-bin spread (`ss/sigma^2`, `n_cells - n_rows` df), and predict bin means with the grid counts. Default `window_radius=0.02` unchanged; `test="parallel"` on overlapping bins warns. Also `predictive_seed` (default: the sampler's `random_seed`, per-gene and per-condition streams from one `SeedSequence`) in `fit_model`/`fit_profiles`; cells without an anchor get bin `-1` | old path bit-identical with pinned predictive seed (sliding `bin_obs`, both tests); unit test: binned and per-cell log densities differ by a constant to 5e-10 when cells sit on bin centres (dropping weights or the sigma term breaks it by 1e3-1e4); weighted bins vs per cell in PyMC: `parallel_stat` 7.31/7.45, 17.50/17.70; calibration (same 80 genes as Stage 1): nulls 0/60 at tau=0 and 0.1, crossing 90%/70%, peak shift 70%/40% (per cell 70%/60%), ~2.5 min vs ~10 min per config; default seed reproducible across reruns and worker counts | done |
| 3 | `bin_obs(sample=...)`: contiguous bins per slide (slide nested in condition; cells without a sample are left out), non-centred random level + random curve per slide (`sample_level_sd`, `sample_curve_sd`, HalfNormal(1); one scale across the curve coefficients), `parallel_curves_test(..., df_denominator=n_slides - 2)` with an F / noncentral F reference (the prototype's scale-mixture variant was not built); `test='parallel'` only, `fit_profiles` refuses slides; fitted curves are population-level (slide terms switched off on the grid) | legnini within-SHH split null (same 10 genes x 5 splits as the pooled check, 5 vs 5 slides, NUTS on bins): 4% rejected at tau=0 and 0% at tau=0.1, against 94% / 62% pooled; tau=0 p-values near uniform (mean 0.53). Real Ctrl vs SHH, 6 strong genes: GLI1, PAX7, HHIP, FOXA2 stay at p <= 7e-4 (W 90-150 instead of 1800-6200, shape_diff 0.77-1.47 vs 0.96-1.75), VAMP2 0.80 (pooled 0.069), DBX1 0.054. 8 unit tests, NUTS snapshot bit-identical | done. Caveats: NUTS on the hierarchical bins mixes modestly (R-hat 1.01-1.05, ESS ~200 of 2000 at 500+500 draws) and not at all for near-empty genes (ALDH1L1, ARG1: R-hat 2.4 / 1.2); per-cell ADVI with slides works with empirical-Bayes scales (`slide_scales`, default), see advi_plan.md |
| 4 | (optional) RW2 smoothing prior on `beta`, `beta_1`, sample curves; enables a derivative metric | power vs Stage 3 on the simulations | not started |
