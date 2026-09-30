# Full-sample versus small-sample model: practical recovery assessment

30 September 2026. The larger model predicts independent observations better overall. The evidence supports completing its replacement bundle, rather than restarting all density fits to clear convergence flags.

## Direct comparison

The same 30,000 held-out quasars and 30,000 held-out stellar-background sources were evaluated with identical photometry, masks, reference choices and transforms. QSO rows lie in 15 original reserved Galactic HEALPix cells and the new test role; stellar rows span 49 new test regions and exclude all original field cones. The independent pools contain 291,976 QSOs and 500,712 stellar-background sources. The sample cap applies only to this evaluation; training is uncapped.

The main comparison uses the complete saved 300-iteration density/spatial candidate against the active small-data model. Positive gain means the larger model gives higher average density to real held-out measurements. Gains are natural-log units (nats) per object, conditional on the measured reference band; they are not gains in classification accuracy or calibrated probability. Compare models within each row of this table, not absolute densities between different band sets.

| Measurements | QSO mean gain [95% sky-bootstrap interval] | Stellar mean gain [95% interval] |
|---|---:|---:|
| All available bands | +0.357 [+0.310, +0.398] | +0.231 [+0.161, +0.302] |
| SDSS + Legacy optical + Pan-STARRS | +0.246 [+0.219, +0.272] | +0.062 [+0.014, +0.107] |
| Legacy optical only | +0.051 [+0.041, +0.063] | +0.071 [+0.045, +0.098] |

All 60,000 objects have at least two usable bands in the first two comparisons. The Legacy-only comparison uses 29,966 QSOs and all 30,000 stars; 34 QSOs lack two usable Legacy optical bands. All evaluated densities are finite. Both models use their own saved spatial stellar weights.

## Training size and numerical health

| Quantity | Active small model | Full-data model |
|---|---:|---:|
| Unique QSO shape-training rows | 55,994 | 1,049,260 (18.7 times more) |
| Stellar shape-training rows | 20,000 | 1,975,894 (98.8 times more) |
| QSO slices meeting stopping criterion | 20/43 | 25/43 at 300; 27/43 after continuation |
| Stellar stopping criterion met | No | No |
| Stellar mixture components | 12 | 20 |
| Finite parameters, positive weights and positive-definite covariances | Yes | Yes |

The active model itself contains capped fits. Numerical stopping flags cannot by themselves justify retaining it over the larger model. Minimum covariance eigenvalues are 0.001173 (small) and 0.001005 (full), with mixture weights normalized to floating-point precision. These checks establish numerical validity, not convergence to a unique optimum.

## Specific limitations

- The bright stellar subset (reference luptitude below 20; 2,357 objects) worsens by 0.531 nats/object in combined optical bands, with sky-bootstrap interval [-0.755, -0.332]. Its all-band change is -0.193 with an interval spanning zero. Legacy-only bright stars improve by 0.032. This is a specific multi-survey optical weakness requiring a focused check, not a reason to discard the full-data training.
- Northern stellar combined-optical performance is approximately unchanged: -0.017 [-0.114, +0.079]. QSO mean gains are positive in both declination strata. These use declination 32 degrees as a simple sky split, not as an assertion about the exact Legacy release boundary.
- QSO all-band improvement at z >= 3 is modest and uncertain: +0.045 [-0.018, +0.103]; optical-only improves by +0.103. The common historical QSO holdout has only 1,334 northern objects, so it does not certify every northern region.
- Improvements are averages: with all available bands, 62.5% of QSOs and 54.2% of stellar sources improve. About 6.2% in each population lose more than one nat. These individual losses are not automatically failures, but the tails must be checked with the catch-all and hard support guard in place.
- The full stellar global mixture alone worsens in combined optical bands by 0.050; spatial conditioning changes the overall comparison to an improvement of 0.062. Preserve spatial weights when evaluating or releasing this model.
- This tests three practical band selections, not every sparse combination of 41 bands. The test sample has now been examined; it must not subsequently be described as an untouched final audit. Any parameter decisions should use the selection/calibration roles, accounting for the fact that final shapes already include selection rows.

## What the additional iterations bought

The saved continuation changes QSO held-out mean log density by only +0.000255 relative to the 300-iteration candidate. Its sky interval includes zero. The stellar global mixture improves by +0.0142 (95% interval +0.0115 to +0.0175). These are much smaller than the main full-versus-small gains. Continued shapes are individual fits without refreshed spatial weights; this report does not mix them with spatial weights fitted for the 300-iteration model. The isolated covariance-floor trial is not part of either evaluated production candidate.

## Bounded recovery plan

1. Use the existing assembled full-data density/spatial candidate as the replacement baseline. Stop treating universal convergence or another capacity search as prerequisites. Preserve the continuation and floor experiments as diagnostics; no blanket retraining is proposed.
2. Check the one demonstrated regression: bright stellar predictions with combined optical bands. Separate survey combinations and sky/reference-band effects, and check whether it causes excessive QSO evidence with the catch-all. Use a predeclared calibration-role diagnostic for follow-up; do not repeatedly tune against the test objects above. Change only a demonstrated cause if consequential for ranking.
3. Complete one coherent replacement bundle: retain matching density shapes and spatial weights, finish population priors under the documented area/completeness assumptions, and fit the catch-all on calibration-role data. Preserve arbitrary-band marginalization, both Legacy hemispheres and the hard support guard. No downloads are required.
4. Check withheld-QSO/stellar score tails and practical band subsets in that complete bundle; then replace the active pointer while retaining the previous bundle for rollback. Do not claim full probability calibration. A measured material ranking failure should trigger a targeted repair, not an open-ended refit of everything.

No fits or downloads were launched for this assessment; the active model pointer is unchanged. The paired evaluation took 38 seconds. Its fast paired-redshift calculation was checked against the existing scorer. Reproduce with `scripts/compare_full_sample_health.py`; numerical results are in `docs/FULL_VS_SMALL_HEALTH_2026-09-30.json`, and row identities and per-object predictions are under `models/multisurvey_psf/work/paired_health/20260930/`.
