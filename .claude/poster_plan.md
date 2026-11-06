# Poster: open questions for the Xenium run

Added 2026-11-06. Draft: https://claude.ai/code/artifact/80ca0657-f4de-4f0b-91b3-6327debf4d9a

Dataset: `/home/lehnerl/Arbeit/data/HiWi/both_conditions_5_7_without_noise.h5ad`, wildtype vs. TgCRND8, one section per
condition, 347 genes, dentate gyrus cells as anchors. Run planned for 2026-11-07.

Figures: (1) two spatial scatter plots with the anchors, (2) volcano, (3) curves by condition for one top gene from the
volcano, (4) correlation of rankings between tests, maybe a p-value histogram, (5) curves by cluster.

Status: questions 1-4 decided by the user on 2026-11-06, see "Decisions" below. The text of the questions is kept as asked.
Notebook for the ADVI run: `docs/notebooks/xenium_advi.ipynb` (written 2026-11-06, the fit has not run).

## Decisions (user, 2026-11-06)

1. **Distance:** each section scaled 0-1 by its own range, as `var_by_distance` does, so neither condition has blank space at
   the far end. This goes against the recommendation in question 1; checked: both sections reach exactly 0 and 1. With
   `min_counts=10` the farthest cells are 3,962 vs. 4,028, so the x-offset is 1.7%, not 4.6% (the 4,144 was a low-count cell).
2. **Layer:** log1p, no `sc.pp.scale`.
3. **Fig. 4 tests:** classic Wald with the Gamma null, shape test on bins with NUTS, shape test on cells with ADVI: three pairwise
   comparisons plus one p-value histogram. No empirical Bayes: it needs at least 3 slides and there are two sections.
4. **Fig. 5:** as recommended, "sounds fine for now".

Fig. 1 is drawn with spatialdata-plot: anchor in red, other cells in viridis by distance. It needs `method="matplotlib"`: with
datashader, the overlapping circles of dense tissue come out unfilled and the pyramidal layer looks like an empty gap.

## To decide before the run

1. **How should distance be scaled?** `var_by_distance` scales each section by its own farthest cell (3,962 vs. 4,144,
   so the same physical distance gets an x about 4.6% apart between conditions). The test could read that as a shape
   difference in every gene. Recommendation: scale both sections by one common maximum.
2. **Which expression layer?** `example.ipynb` also runs `sc.pp.scale`, but tau = 0.1 is defined on log1p; on z-scored
   data it means something different for each gene. Recommendation: log1p without scaling, which the poster text assumes.
3. **Which tests go in Fig. 4?** Options: classic Wald with the Gamma null, shape test on bins with NUTS, shape test on
   cells with ADVI. The slide-effect test cannot run with one section per condition. Recommendation: all three if the ADVI
   run on about 113,000 cells finishes in time (its runtime on this data is untested).
4. **Which clustering goes in Fig. 5?** With 347 genes the legend will not fit and the flat cluster will dominate.
   Recommendation: `by="interaction"`, only the patterned clusters, `show_legend=False`, a few genes labelled by hand. If
   clustering on beta_1, say that the overlap with flagged genes describes them and does not validate them (same
   coefficients as the test).

## Needs the user's input

- Conference, authors, affiliations, funding, poster size.
- Are the coordinates in um? (Inferred from the extents, about 6,000 x 5,000.)
- Sample age and section plane (the filename suggests 5.7 months).
- Check the references against the published versions; they were written from memory.

## Offered, not started

- Fig. 1 needs no fit, so the two spatial scatter plots can be made before the run. Done 2026-11-06, in the notebook.

## Read of the Xenium ADVI run (2026-11-06, from the saved tables, no refit)

Results: `/home/lehnerl/Arbeit/results/06_11_results/xenium_advi/`. 152 of 347 genes p < 0.05 (149 BH), 343 pass the variational
diagnostics. Only 19 genes have p between 0.05 and 0.95: with 113,000 cells the test acts as a threshold on `shape_diff` at `tau`.
Share flagged by expression quintile: 3%, 16%, 35%, 71%, 90%. 38% (median) of a gene's squared shape difference lies in x > 0.9,
which holds 2.4% of the cells; PC1 of the difference curves (48% of their variance) sits at x = 1.
Rough moderation prototype on these tables (log `shape_diff` against a lowess trend on mean expression, scaled chi-square): 31-75
BH genes depending on the assumed df (3-9); the spread around the trend is wider than chi-square, so this is not calibrated.
The results file has no `dsvg_coefs_*` tables, so the cluster's diffsvg predates commit 5849667; Fig. 5 needs a rerun with the current code.
Options discussed with the user, none decided: more sections; moderation across genes by expression trend, validated on single
legnini slide pairs; angular sectors as pseudo-slides; trimming or cell-weighting the far end of x.
