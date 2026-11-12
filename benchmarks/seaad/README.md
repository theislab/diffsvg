# SEA-AD MTG MERFISH benchmark

**Correction (2026-11-12, afternoon):** the `lognorm` layer of the SpatialDataDB SEA-AD tables holds near-raw counts (maximum 516, correlation 0.48 with log1p of counts per 10k), not log-normalized values. Every result below that table used it. The analyses are being rerun on `log1p` = log1p(normalize_total(csc_counts)), and the numbers below are superseded until then.

Data: SpatialDataDB study `seaad:MTG-MERFISH` (64 sections, 27 donors, 140 genes). The tables of all sections are concatenated into one h5ad with the `sddb` client (`Catalog().query().filter(study_id="seaad:MTG-MERFISH").to_list(elements=["tables/table"])`). The distance axis is `Normalized_depth_from_pia` (0 = pia, 1 = white matter), present for 321,226 cells. The replicate unit is the donor (`sample="donor"`; the sections of a donor are pooled).

Contrasts: ADNC High + Intermediate (14 donors) vs Low + Not AD (13 donors); Dementia (10) vs No dementia (17); and five random donor splits with the ADNC group sizes, where no gene truly differs.

Results (2026-11-12, prototype/all-prs, exact inference, tau = 0.1 unless noted), genes at BH < 0.05 out of 140:

| Contrast | Pooled cells | Donor effects | Donor effects, nominal p < 0.05 |
|---|---|---|---|
| Random donor split, 5 seeds | 99 to 111 | 0 in every split | 0 to 5 (mean 1.6) |
| ADNC | 114 | 0 | 4 |
| Dementia | 109 | 0 | 3 |
| ADNC, tau = 0 | not run | 0 | 10 |
| ADNC, glutamatergic neurons only | 115 | 0 | 3 |

Pooling the cells of all donors flags about as many genes for a random donor split as for the real contrasts, so its hits measure donor-to-donor variation. With donor effects, the random splits give no hits at BH < 0.05, and the nominal rate (1.1%) is below 5%. The real contrasts give no hits at BH < 0.05 either. The same upper-layer genes rank first in the ADNC runs (LAMP5, RFX3, STXBP6, HS6ST3), which fits the reported vulnerability of layer 2/3 neurons, but none passes the correction.

Runtime per fit on a laptop: about 1.7 s pooled and about 2 minutes with 27 donors.

## Level test and power (2026-11-12)

The level test (`test="level"`, the mean gap between the curves) with donor effects gives 0 hits at BH < 0.05 on three random donor splits, and 0 for the ADNC and dementia contrasts, at both tau = 0 and tau = 0.1.

`power_spikein.py` adds known effects to one group of a random donor split, so the effects sit on top of the real donor variation: five sizes on 20 genes each, and 40 genes left as null controls, over 3 seeds. Detection rate at BH < 0.05, with genes split at the median of their estimated between-donor sd:

| Test | Donor variability | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|---|
| level, tau = 0 | below median | 0.18 | 0.48 | 0.72 | 0.90 | |
| level, tau = 0 | above median | 0 | 0 | 0 | 0 | |
| shape, tau = 0 | below median | 0 | 0.16 | 0.34 | 0.52 | 0.80 |
| shape, tau = 0 | above median | 0 | 0 | 0 | 0.04 | 0 |
| shape, tau = 0.1 | below median | 0 | 0 | 0 | 0.15 | 0.47 |

Effects are in log-expression units (0.2 is about 1.2-fold). The 480 null-gene tests gave no false positives. The spiked effects are estimated without bias (median 0.47 to 0.49 for a level shift of 0.5). Missed genes have a between-donor sd around 0.85, against 0.16 for detected ones, and they are largely cell-type markers, whose whole-tissue expression depends on how much of each cell type a section contains. With 27 donors the method detects moderate effects in genes that are stable between donors, and it cannot detect them in genes dominated by differences in cell-type composition.
