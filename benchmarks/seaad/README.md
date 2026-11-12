# SEA-AD MTG MERFISH benchmark

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
