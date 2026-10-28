# diffsvg: how the curve test and the ADVI model came to be

Reasoning and reference, written 2026-10-21. Plans with their status tables are `shape_test_plan.md` (the test) and
`advi_plan.md` (variational inference, slides); this file explains *why* the current design looks as it does and what
every quantity of the curve test is. Experiment scripts and logs: `advi_prototype/`, `shape_test_prototype/`.

---

## Part 1. The curve test (`test="parallel"`)

### The model, per gene

Rows `i` are cells, or bins of cells (then `y_i` is the bin mean and the row carries `n_i` cells). `x_i` is the anchor
coordinate in [0, 1], `c_i` the condition code (0 or 1), `j(i)` the slide.

```
y_i = alpha + sum_k B_k(x_i) beta_k                        # curve of condition 0
          + gamma * c_i                                    # level shift of condition 1
          + c_i * sum_k B_k(x_i) beta1_k                   # change of the curve in condition 1
          + [ u_j0 + sum_k B_k(x_i) u_jk ]                 # only with slides (sample=...), slide j = j(i)
          + eps_i,        eps_i ~ N(0, sigma^2 / n_i)      # n_i = 1 for single cells
```

* Curve of condition 0: `f0(x) = alpha + B(x) beta`. Curve of condition 1: `f1(x) = alpha + gamma + B(x) (beta + beta1)`.
* **Difference curve** `d(x) = f1(x) - f0(x) = gamma + B(x) beta1`. Everything the test says is about `d`.
* Priors: `alpha, beta_k, beta1_k, gamma ~ N(0, 1)`, `sigma ~ HalfNormal(1)` (`DEFAULT_PRIORS`, overridable with
  `setup_model(priors=...)`). They assume O(1)-scaled data, which is why the layer is `log1p_norm`.
* Slides: `u_j0 = s_level * z_j`, `u_jk = s_curve * z_jk`, `z ~ N(0, 1)` (non-centred), `s_level, s_curve ~ HalfNormal(1)`
  (`sample_level_sd`, `sample_curve_sd`) or fixed by empirical Bayes (Part 2). The slide terms are switched off on the
  prediction grid, so plotted curves are population-level.
* Weighted bins (`window_radius=None`): rows are weighted by their cell count, and a `Potential`
  `-0.5 (2 df log sigma + SS_within / sigma^2)` with `df = n_cells - n_rows` restores the cell-level noise, so a binned fit
  has the same likelihood as a fit on the cells (checked: log densities differ by a constant).

### Where each quantity comes from

| Quantity | Meaning | Source |
|---|---|---|
| `x` | normalised distance to the closest anchor cell, in [0, 1] | `design_matrix[anchor]`, e.g. squidpy `var_by_distance` ("True") |
| `c` | condition code | `design_matrix[condition]`; the two levels, sorted as strings, become 0 and 1 (`condition_categories`) |
| `j` | slide of a row | `design_matrix[sample]`; each slide must belong to one condition; slides without anchor cells drop out |
| `y` | expression | `adata.X` or `adata.layers[layer]`; cells (`use_binned=False`) or bin means from `pp.bin_obs` |
| `B_k(x)` | cubic B-spline basis, `K = n_bsplines = 10` columns, evenly spaced knots on [0, 1], columns centred (`center(bs(...))`) | `pp.bin_obs` (`n_bsplines`, `degree`), evaluated per cell by `pp.spline_basis_at` |
| bins | `n_bins = 50` (51 grid points); sliding windows of radius 0.02 (default, overlap) or `window_radius=None` (contiguous, each cell once) | `pp.bin_obs` |
| `alpha` | intercept (expression of condition 0 at the spline's mean) | fitted |
| `beta` | spline coefficients of condition 0's curve | fitted |
| `gamma` | `categorical_beta`: level shift of condition 1 | fitted; mean and 94% interval in `categorical_beta_*` |
| `beta1` | spline coefficients of the interaction: how condition 1's curve differs in shape and level | fitted; **the only parameters the test looks at** |
| `sigma` | per-cell noise sd | fitted |
| `s_level`, `s_curve` | between-slide sd of level and curve coefficients | fitted (NUTS) or empirical Bayes (ADVI); columns `sample_level_sd`, `sample_curve_sd` |
| `tau` | smallest shape difference that counts | user, `fit_model(tau=0.1)` |
| `nu` | denominator df with slides, `n_slides - 2` | `parallel_df_denominator` |

### The statistic

Level and shape of the gap separate: `d(x) = d_bar + (d(x) - d_bar)`, `d_bar = integral of d over [0, 1]`. The curves are
parallel exactly when the second part is 0.

* **`shape_diff`** `E = sqrt(12) * RMS(d - d_bar) = sqrt(beta1' Q beta1)`, with `Q = 12 * Gram(centred basis)` computed on a fine
  grid by `shape_metric` (it depends only on `n_bsplines`, `degree`). `Q 1 = 0`, so the level never enters. Units: slope
  equivalent - if the gap changes linearly across the range, `E` is that total change; otherwise the linear change with
  the same RMS departure from parallel. On `log1p` data 0.1 means the Ctrl/SHH fold change drifts by ~10% across the
  range. The averaging over x is uniform on [0, 1], not weighted by cell density. It is computed from the posterior mean
  of `beta1`, so noise inflates it for poorly measured genes (the p-value accounts for that, the raw number does not).
* **`slope_diff`** `g' beta1`, `g = 12 * integral (x - 1/2) B(x) dx`: least-squares slope of `d` over [0, 1], signed
  (positive: the gap grows with distance from the anchor); with `slope_diff_ci_low/high` the 3%/97% quantiles over draws.
  X-axis of the volcano plot.
* **`integral`**: the level difference, trapezoid area of the mean posterior-predictive difference curve. Not tested.
* **Hypotheses**: `H0: E <= tau` against `H1: E > tau`. `tau = 0` is the classic parallelism test.
* **`parallel_stat`** `W = z' Omega^-1 z`, with `Q = L L'` over its `K - 1` non-zero directions, `z = L' mean(beta1)`
  (so `|z| = E`), `Omega = L' V L`, `V` = posterior covariance of `beta1` (draws from NUTS or ADVI). `parallel_df = K - 1 = 9`.
* **`parallel_pval`**: `tau = 0`: chi-square(9). `tau > 0`: noncentral chi-square(9, `tau^2 / omega_min`), `omega_min` =
  smallest eigenvalue of `Omega` (the best-measured shape direction; exactly valid because under `E <= tau` the
  noncentrality is at most that). With slides: the same with F(9, `nu`) / noncentral F, because the covariance then holds
  scale estimates from few slides. `parallel_pval_adj`: Benjamini-Hochberg over the panel.
* Why a p-value near 1 for many genes at `tau > 0`: an interval null pushes genes with a true `E` below `tau` towards 1,
  so the histogram is two-peaked by design (permutation null: p ~ 0.98-1.0 at tau 0.1, uniform at tau 0).
* **Diagnostics columns**: NUTS `n_divergent`, `rhat_max`, `ess_bulk_min`; ADVI `vi_khat` (want < 0.7), `vi_ess` (want > 0.5,
  > 0.1 with slides).

---

## Part 2. How we arrived at the current ADVI model

**Goal.** Fit single cells (`use_binned=False`) fast enough to use, with the shape test, eventually with ZINB counts.
NUTS on 41k legnini cells takes >14 min per gene and keeps a `mu` trace of draws x cells (~1.3 GB per gene: it exhausted
the user's RAM once). Earlier ADVI attempts were reported as bad.

1. **Reference without MCMC.** For the Normal model with N(0, 1) priors the coefficients are Gaussian given sigma, so the
   exact posterior (sigma integrated on a grid) is cheap (`advi_prototype/lab.py`). Every comparison below is against it.
2. **Plain ADVI fails** (`naive_advi.py`): mean-field and full-rank with PyMC's default optimiser reach W at 0-1% of
   the exact value; tuned Adam, 30k steps, full-rank 43-51%. Cause: the coefficient posterior is a thin tilted ridge
   (sd ~0.007 along directions the data fix, ~0.3 along the others; B-spline coefficients are strongly correlated; the
   constant direction of `beta1` is prior-only). A unit Gaussian start must shrink into it by noisy gradients.
   Mean-field cannot tilt at all (shape-direction variance up to 70x too large), so it is not offered.
3. **Laplace start.** Mode (L-BFGS + Newton, on the unconstrained scale with Jacobians) and inverse Hessian
   (central differences of the analytic gradient, no second-order graph) put the full-rank Gaussian next to the optimum;
   Adam only polishes (2000 steps, learning rate 1e-3 -> 1e-5). Result: W within ~5% of exact, permutation null calibrated
   like the exact posterior, ~8 s per gene. For a Gaussian model Laplace alone is already near exact; the polish matters
   for non-Gaussian likelihoods (ZINB, untested).
4. **"Did it work?"** Importance ratios target/q at 1000 draws: Pareto k-hat (tail too heavy: q too narrow) and the ESS share
   (q too wide or off-centre). Both separate good from bad fits cleanly (`advi_plan.md`). k-hat is our own implementation:
   `arviz_stats.pareto_khat(log_weights=True)` fits the log-weights in 1.2.0. `fit_model` warns when genes fail.
5. **Pooled cells over-reject.** Within-SHH split (no true difference): 94% (tau 0) / 62% (tau 0.1) of genes significant.
   Cells of a slide share level and curve shape, the model credits `sigma^2 / n_cells` but the truth is
   `sigma^2 / n_cells + tau_slide^2 / J`; with 40k cells only the second term is left. Smoothing or penalties act along x,
   not across replicates, and a prior cannot supply a missing level of the data-generating process.
6. **Slides as random level + random curve**, F reference with `n_slides - 2` df (Stage 3). NUTS on per-slide bins:
   4% / 0% on the same split; strong SHH genes stay significant, with W of ~100 instead of thousands.
7. **ADVI with slides fails at first.** The Laplace start is the joint mode of (z, log scale). In the non-centred form
   a large scale with small z is cheap, so that mode follows the effect size and the HalfNormal(1) prior, not the posterior
   (simulated slides: scales 1.3 / 2.3 against NUTS 0.40 / 0.55), the covariance of `beta1` inflates, W comes out ~30x too
   small and the diagnostics say so (ESS 0.001).
8. **Two remedies, `slide_scales`.**
   * **`empirical_bayes` (default).** Integrate the slide effects out and put the two scales at the mode of their marginal
     posterior. Per round: mode of everything else at the current scales, expansion of the log density there rewritten in
     `u = scale * z`, where the scales appear only in the Gaussian prior of `u`:
     `log p(h | y) ~ max_u [ b'd - d'Ad/2 - sum_j u_j^2 / (2 s_j^2) ] - n_j log s_j - log|A + D|/2 + log p(h) + Jacobian`
     (`d = u - u_hat`, `D = diag(1/s^2)`), maximised over the two log-scales with one Cholesky per evaluation; 2-3 rounds.
     Then the model is rebuilt with the scales fixed and fitted by Laplace start + ADVI. The plug-in ignores the scales'
     uncertainty, which the F reference with `n_slides - 2` df makes up for (this is what the Stage 0 prototype did, with 1.6%
     false positives). Matches NUTS: W 4.4-4.5 vs 4.69, scales 0.35 / 0.54 vs 0.39 / 0.55; real split null 6.7% / 0%.
   * **`data_start`.** Keep the scales random, start ADVI at a least-squares guess. It does not work: the free scales drift
     too small (~0.2 vs 0.4 / 0.55), so p-values are anti-conservative; the diagnostics flag it (k-hat 5-120). The posterior
     over (z, scale) is a curved ridge (the data fix `scale * z`), and a single Gaussian fitted by stochastic gradients
     represents it badly. Empirical Bayes avoids the ridge by integrating `z` out and optimising only two scalars; what
     remains is close to Gaussian. (This mechanism is our explanation of the measured drift, not separately tested.)
   * With slides the polish uses 10x smaller steps (1e-4 -> 1e-6), and the ESS threshold is 0.1: the share falls with the
     number of parameters (~130-200) even for a good fit (0.45 while matching NUTS).

### Why simpler ways of accounting for slides do not work

* **A categorical slide variable (fixed effects).** Slide is nested in condition, so slide dummies are collinear with the
  condition term; a fixed-effects fit has no between-slide error term to judge the condition contrast against.
* **With N(0, 1) priors on the dummies** it is already a random effect, but with its sd frozen at 1: the posterior of the
  condition effect is then too wide or too narrow depending on the true sd. It has to be a *learned* scale.
* **Levels only is not enough.** Prototype, same split null: pooled 82% false positives, random intercept + slope 60%, a
  random spline *curve* per slide 11.5%, 1.6% with the F reference. Slides differ in shape, not just level - so the
  dummies must be crossed with the spline basis (`J x K` columns). That, with a learned scale, is the random curve we fit;
  empirical Bayes vs NUTS vs `data_start` only differ in how the scale is obtained.
* **A slide x condition interaction** adds nothing: with slides nested in conditions its columns equal the slide dummies.
  It would matter in a crossed (paired) design where the same slide is measured under both conditions; that would need
  `bin_obs` to allow such slides and a new term (not built).
* **Smoothing / penalised splines** act along the anchor axis. They reduce noise in one curve; they cannot create the
  between-slide variance term. (The package spline is not roughness-penalised anyway: independent N(0, 1) priors on the
  coefficients; difference penalties are the optional Stage 4 of the shape-test plan.)
* **Two-stage** (a curve per slide, then compare coefficient vectors across 6 vs 11 slides) is valid but loses partial
  pooling; **cluster-robust covariances** are unreliable with 17 clusters and 9 shape directions.

### Not built / open

* Analytic marginalisation (Normal likelihood only): given sigma the model is a linear mixed model
  `y ~ N(X beta, sigma^2 W^-1 + Z G Z')`; Woodbury gives the exact marginal of the scales and the exact Gaussian posterior
  of `beta` in an `n_slides (K + 1)`-dimensional system. Faster and deterministic, but no help for ZINB.
* ZINB / counts (Stage 2): needs a log link and offset in the likelihood registry; legnini's `raw` layer is library-size
  normalised to 279, counts recoverable as `raw / min nonzero` per cell. To be discussed.
* `fit_profiles` and `check_stability` do not take `inference` or slides.
* NUTS on the hierarchical bins mixes modestly (R-hat 1.01-1.05) and fails for near-empty genes.
