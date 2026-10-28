# Variational inference (ADVI) for the per-cell fits: plan and evidence

Goal (user, 2026-10-21): see whether ADVI can replace NUTS on raw, unbinned cells, in particular together with the
parallel-curves test (`test="parallel"`), with raw/ZINB data as the eventual target. Earlier ADVI attempts gave bad
posterior predictives and coefficients that may have been noise. Quick checks on 1-2 legnini genes only.

## What was built

`fit_model(..., inference="advi")` (default stays `"nuts"`, bit-identical on seeded runs of the sliding, weighted
and per-cell paths). Per gene: full-rank ADVI started at the Laplace approximation (L-BFGS + Newton for the mode on
the unconstrained scale, central-difference Hessian of the analytic gradient), then Adam with an exponentially
decaying learning rate (`n=2000`, 1e-3 -> 1e-5), 4000 draws from the fit, then the usual predictive/test code.
Stats columns: `vi_khat`, `vi_ess` replace `n_divergent`/`rhat_max`/`ess_bulk_min`. Settings go in `sample_kwargs`
(`_VI_DEFAULTS` in `tl/model.py`; `n=0` is the plain Laplace approximation, `init=None` PyMC's own start).
`record_mu=False` for the VI model: the per-row `mu` deterministic would put draws x cells floats in the trace.
Not done on purpose: mean-field (below), `fit_profiles`, `check_stability` pass-through.

## Evidence (legnini, Ctrl vs SHH, `log1p_norm`, 41,630 cells; GLI1 strong, VAMP2 borderline)

Reference = the exact posterior of the Normal model (coefficients Gaussian given sigma, sigma integrated on a grid;
`advi_prototype/lab.py`), so no MCMC is needed to judge ADVI. Scripts and logs: `advi_prototype/`.

**Plain ADVI does not work here** (`naive_advi.py`, W relative to exact; GLI1 3130, VAMP2 41.9):

| fit | GLI1 W/exact | VAMP2 W/exact |
|---|---|---|
| mean-field, PyMC default, 10k steps | 0.002 | 0.007 |
| full-rank, PyMC default, 10k steps | 0.000 | 0.001 |
| mean-field, Adam, 10k | 0.053 | 0.032 |
| full-rank, Adam .05 -> 1e-4, 30k | 0.51 | 0.43 |

Cause: the coefficient posterior is a thin tilted ridge (sd ~0.007 in directions the data fix, ~0.3 in those they do
not; B-spline coefficients are strongly correlated, and the constant direction of `beta_1` is prior-only). A unit
Gaussian start must shrink into it by noisy gradients; mean-field cannot tilt (shape-direction variance up to 70x too
large, marginal sds of the weak directions 5-18% of true), PyMC's default optimiser has barely moved after 10k steps.

**Laplace start + polish works** (`null_permutation.py`, package `_fit_advi`; same-data ratios to exact):
- Real data, W/exact: default 0.99-1.02 per run; across 8 seeds W CV 3.6% (GLI1) / 4.4% (VAMP2); Laplace alone
  1.8% / 3.1% (that is mostly the 4000-draw noise). VAMP2 p(tau=0.1): exact 0.072, ADVI 0.061-0.144 over seeds,
  Laplace 0.053-0.093 - a borderline gene stays seed-noisy, a clear one does not (GLI1 p = 0 in all runs).
- Permuting `condition` across cells (identical curves; 30 permutations/gene): share of p < 0.05 at tau=0 exact
  6.7% / 6.7%, ADVI 6.7% / 10%, Laplace 6.7% / 6.7%; at tau=0.1 all 0%. W/exact on the same permutation 5-95%:
  0.95-1.06 (GLI1), 0.95-1.03 (VAMP2); shape_diff within 0.003.
- Timing at 41k cells, one thread: ~2 s Laplace, ~7 s for 2000 Adam steps, ~1-2 s diagnostics. Set
  `OMP_NUM_THREADS=1` (and friends) when running workers: BLAS oversubscription made it ~10x slower.
- For this Gaussian model the Laplace approximation alone is as good (it is nearly exact); the polish only adds
  noise. Its job is non-Gaussian likelihoods (ZINB) - untested.

**Diagnostics** (`_importance_diagnostics`, importance ratios target/q at 1000 draws; k-hat as Vehtari et al. 2024,
own implementation, checked against `arviz_stats` in a test - its `log_weights=True` path is buggy in 1.2.0):

| q | k-hat | ESS share |
|---|---|---|
| Laplace / Laplace + polish | 0.15 / 0.16 | 0.99 / 0.80 |
| full-rank 30k Adam (too wide) | 0.95-1.5 | 0.005-0.016 |
| mean-field | 80-90 | 0.0005 |
| Laplace covariance x0.8 / x1.25 | 0.24 / -0.07 | 0.51 / 0.62 |
| Laplace covariance x0.5 / x2 | 0.82 / 0.55 | 0.021 / 0.053 |

k-hat only sees a q that is too narrow; ESS sees too wide as well. Rule of thumb from the table: k-hat < 0.7 and
ESS share > 0.5 means the covariance (hence W) is within ~25%. `init=None` lands at ESS 0.001 (test covers it).

**Within-condition null through the package** (`advi_prototype/slide_split_null.py`, 2026-10-21): the 10 SHH slides
with an anchor split 5 vs 5 at random (no true difference), `inference="advi"` per cell, 10 genes x 5 splits:
p < 0.05 for 94% of fits at tau=0 and 62% at tau=0.1, median apparent `shape_diff` 0.27 (above tau), while the
variational diagnostics stayed healthy (k-hat <= 0.35, ESS share >= 0.68). So the over-rejection on real data is the
pooled model (no slide variation), not the approximation; matches the Stage 0 numbers (82% / 57%). Small sample.

## Slides (hierarchical model) and ADVI

With `bin_obs(sample=...)` the model has a random level and curve per slide (non-centred, half-normal scales). A plain
Laplace start fails: it sits at the *joint* mode of (z, log scale), which in the non-centred form follows the size of the
slide effects and the prior, not the posterior (simulated slides, NUTS level/curve sd 0.40/0.55: joint mode 1.27/2.30,
ADVI W 0.15 vs NUTS 4.6, vi_ess 0.001). Two remedies are built, chosen with the VI setting `slide_scales`
(`tl/model.py`, section "Slides with ADVI"):

- `"empirical_bayes"` (default): `_empirical_bayes_scales` fixes the two scales at the mode of their marginal posterior,
  the slide effects integrated out by a Laplace approximation (lme4/TMB style). Each round expands the log density once
  (finite-difference Hessian at the conditional mode), rewrites it in u = scale * z, where the scales only enter the
  Gaussian prior of u, and maximises the marginal over the log-scales with one Cholesky per evaluation; 2-3 rounds.
  Then the model is rebuilt with the scales fixed (`sample_scales`) and fitted with Laplace start + ADVI; the F
  reference allows for the plug-in. Start: `_guess_slide_scales`.
- `"data_start"`: scales stay random, ADVI starts at `_guess_slide_scales` (pooled least squares, then per-slide level
  and curve fits to the residuals; RMS of those) with log-scale sd 0.3. Kept for comparison: it does not work.

Polish with slides uses steps ten times smaller (`_VI_SLIDE_DEFAULTS`, 1e-4 -> 1e-6) and `vi_ess > 0.1` is the
threshold (`_VI_MIN_ESS`): with ~130-200 parameters the ESS share falls even for a good fit (0.45 while matching NUTS).

Evidence (`slide_scales_check.py`, `slide_eb_real.py`):

| fit | simulated, no effect: W / p / scales | simulated, shape effect 1.0: W / p | vi_khat / vi_ess |
|---|---|---|---|
| NUTS bins (4 x 1000) | 4.69 / 0.83 / 0.39, 0.55 | 7.69 / 0.59 | - |
| EB bins | 4.46 / 0.84 / 0.35, 0.54 | 7.64 / 0.60 | 0.23 / 0.45 (quiet polish) |
| EB cells | 4.42 / 0.85 / 0.36, 0.55 | 7.98 / 0.57 | 0.33 / 0.45 (quiet polish) |
| data_start bins | 6.87 / 0.65 / 0.30, 0.47 | 10.2 / 0.44 | 4.8 / 0.007 |
| data_start, lr 1e-2, 6000 steps | 15-17 / 0.20-0.24 / 0.19, 0.18 | 15-16 / 0.22-0.24 | 66-123 / 0.001 |

data_start's free scales drift too small, so its p-values are anti-conservative; its diagnostics flag it.
Real data, EB on cells (per cell, 30k-41k cells): within-SHH split null (10 genes x 3 splits, same halves as below)
6.7% rejected at tau=0 (mean p 0.49) and 0% at tau=0.1, vi_khat <= 0.35, vi_ess >= 0.30. Ctrl vs SHH (tau 0.1): W
GLI1 114, PAX7 93, HHIP 139, VAMP2 7.9, FOXA2 151, DBX1 27 (NUTS on bins: 126, 89, 140, 7.9, 146, 28), p 1e-4 to 4e-4
for the four pathway genes, VAMP2 0.80, DBX1 0.060. Cost ~40-60 s per gene and worker (vs ~5 s without slides).

**Analytic marginalisation (not built, note for later).** For the Normal likelihood, given sigma, the model is a linear
mixed model: y ~ N(X beta, sigma^2 W^-1 + Z G Z'), Z the slide-by-basis design, G = diag(level sd^2, curve sd^2 ...).
The slide effects can be integrated out exactly (Woodbury: only a (n_slides * (K + 1))-dimensional system per
evaluation), giving the exact marginal likelihood of the scales and sigma (REML or ML) and the exact Gaussian
posterior of beta given them - what the Stage 0 prototype (`shape_test_prototype/slide_curves.py`) did. It would be
faster and deterministic for the Normal model, but it does not carry over to ZINB, which is why empirical Bayes was
built on the generic Laplace machinery instead.

**Within-condition null through the package** (`advi_prototype/slide_split_null.py`, 2026-10-21): the 10 SHH slides
with an anchor split 5 vs 5 at random (no true difference), `inference="advi"` per cell, 10 genes x 5 splits:
p < 0.05 for 94% of fits at tau=0 and 62% at tau=0.1, median apparent `shape_diff` 0.27 (above tau), while the
variational diagnostics stayed healthy (k-hat <= 0.35, ESS share >= 0.68). So the over-rejection on real data is the
pooled model (no slide variation), not the approximation; matches the Stage 0 numbers (82% / 57%). Small sample.

## Slides (hierarchical model) and ADVI

With `bin_obs(sample=...)` the model has a random level and curve per slide (non-centred, half-normal sds). On a
small simulation (10 slides, no condition effect, 15 s of NUTS on bins: level sd 0.40, curve sd 0.55, W 4.6, p 0.83)
NUTS behaves; per-cell/bin ADVI does not (W 0.15, p 1, vi_ess 0.001, vi_khat 5-16). Cause: the Laplace start is the
*joint* mode of (z, log sd), which in the non-centred form sits at sd 1.27 / 2.30 (small z, large sd is cheap, so it
tracks sqrt of the effect size and the HalfNormal(1) prior rather than the posterior), 3-4 posterior sds away from the
bulk, and the SGD polish cannot cross that with a covariance taken at the wrong point. The diagnostics flag it and
`fit_model` warns when genes fail them. Options, not built: (i) empirical Bayes as in the Stage 0 prototype - choose
the sds by the Laplace-approximated marginal likelihood (nested: inner Newton over coefficients and z, outer over the
sds), plug them in, keep the F reference; (ii) start the sds at a data-driven value and let ADVI move them with a
larger step; (iii) integrate the Gaussian slide terms out analytically for the Normal likelihood. For ZINB none of the
bin tricks apply, so one of these is needed before cells + slides + counts can be combined.

## Decisions and caveats

- Full-rank only: the test is built on the covariance of `beta_1`.
- Per-cell NUTS was not used as a reference: it did not finish one gene in >14 min with 11 cores busy and keeps a
  `mu` trace of draws x cells (~1.3 GB/gene at 41k cells); a run of it exhausted RAM on the user's machine. If a
  NUTS comparison is ever wanted, subsample cells and build with `record_mu=False`.
- Seeded NUTS output of the weighted-bins path differs between a cold and a warm compile cache (first run only).
- `ruff format --check` flags `pl/plot_stability.py` and `tl/stability.py` (already so before this work).
- The `raw` layer of legnini is library-size normalised to 279 per cell, not integer counts. Counts are recoverable
  as `raw / min_nonzero` per cell (always integer-valued; assumes the smallest count of a cell is 1), offset
  `-log(min_nonzero)`; cells with zero total must be dropped.

## Stages

| Stage | What | Verified by | Status |
|---|---|---|---|
| 1 | `inference="advi"` in `fit_model` (Laplace start, Adam polish, `vi_khat`/`vi_ess`) | `tests/test_advi.py` (11), NUTS snapshot bit-identical, evidence above | done |
| 2 | ZINB / count likelihood on raw cells (log link + offset in the likelihood registry, priors for dispersion and zero inflation); check ADVI with the k-hat/ESS diagnostics, small cell subsample for any NUTS comparison | to discuss with the user | not started |
| 3 | slides (`bin_obs(sample=...)`, shape-test Stage 3): NUTS on per-slide bins; ADVI with `slide_scales="empirical_bayes"` (default) or `"data_start"` | split null NUTS 4% / 0%, EB-ADVI on cells 6.7% / 0%; real contrast EB matches NUTS; data_start fails (flagged); 11 slide tests | done |
| 4 | `fit_profiles` / `check_stability` pass-through; iterate averaging if seed noise matters | - | not started |
