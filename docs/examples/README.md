# Figure 10 versions for 12 survey combinations

[Download the 12-page PDF](multisurvey_examples.pdf). Each image below also opens at full resolution.

All versions use the same **5 held-out spectroscopic quasars** (Q1-Q5) and **5 unclassified field sources** (F1-F5). These are new examples selected for common coverage, rather than the objects in the original Figure 10, which remains unchanged.

The sample was drawn with seed 0, before evaluating scores, from 120 eligible quasars in 8 reserved sky blocks and 49 eligible field sources. Quasars come from 5 different blocks; all field sources come from the extra reserved overlap field (field 24). Selection requires the plotted bands to be measured and 19 < SDSS r luptitude < 21.5. This small common-coverage illustration is not a representative performance test; the [larger validation](../MULTISURVEY_VALIDATION.md) supplies those results.

Blue contours show the quasar model at the target redshift; green dashed contours show the all-source background. Both are normalised two-colour densities conditioned on the measured reference band, with all other bands marginalised and the plotted bands' measurement noise convolved before conditioning. Levels are 5%, 30%, and 80% of peak density, not enclosed probabilities. Circles mark measurements; bars and ellipses show one-sigma colour errors, including the covariance from the common reference. Axes are shared across all ten panels within each figure and include every point and its two-sigma error extent.

**Annotations use every available band in the selected surveys**, so two displayed colours need not explain the full score. `ln BF` is the natural-log quasar-at-target-z/background likelihood ratio. `P_z` is `p_zmatch_given_qso` for a +/-2000 km/s window and a flat redshift prior over model support (0.15-4.35). It assumes the object is a quasar and is not the probability of a physical companion. Population posteriors and `log R` remain unavailable without matching surface-density priors. The same target redshift is used in the top and bottom panel of each column.

Optical bands retain their native AB calibration; ALLWISE and VHS retain Vega calibration. Colours are differences of the model's saved luptitudes, not ordinary magnitude colours at low signal-to-noise. Usable negative fluxes remain measurements. DECaLS-only uses the southern grz background marginal; all other versions use the joint background. These figures show catalogue-object photometric diagnostics, without a per-object local background fit or a claim that the sources meet a close-companion blend policy.

## Fixed objects

| ID | RA (deg) | Dec (deg) | Target z | Reserved block/field |
|---|---:|---:|---:|---:|
| Q1 | 145.8397888 | -0.7945198 | 0.27166 | 38 |
| Q2 | 339.4651511 | -8.3335089 | 0.27454 | 131 |
| Q3 | 140.1365433 | -0.2993134 | 0.33837 | 35 |
| Q4 | 24.4082496 | -1.5926308 | 1.13569 | 145 |
| Q5 | 190.9469253 | -7.9503697 | 3.30415 | 62 |
| F1 | 151.5067084 | -3.3255632 | 0.27166 | 24 |
| F2 | 151.5652728 | -3.1711441 | 0.27454 | 24 |
| F3 | 151.5752661 | -3.3239194 | 0.33837 | 24 |
| F4 | 151.5453828 | -3.2446987 | 1.13569 | 24 |
| F5 | 151.6200268 | -3.2501954 | 3.30415 | 24 |

F1-F5 have assigned target redshifts, not measured spectroscopic redshifts.

## Scores on the same objects

Natural-log Bayes factors; signs do not provide spectroscopic class labels for the field sources.

| Survey inputs | Q1 | Q2 | Q3 | Q4 | Q5 | F1 | F2 | F3 | F4 | F5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SDSS | +1.4 | +4.4 | +1.9 | -2.7 | +9.0 | +0.2 | -4.6 | -3.2 | -20.3 | -2.5 |
| DECaLS | +1.0 | +1.9 | +6.5 | -4.1 | +1.7 | -0.7 | -4.6 | -1.9 | -16.1 | -4.7 |
| ALLWISE | +4.3 | +3.0 | -1.1 | +9.3 | +3.5 | +1.2 | -1.1 | -2.3 | -5.4 | -0.8 |
| PS1 | +1.2 | +1.6 | +4.4 | -2.0 | +2.4 | -3.1 | -9.2 | -2.2 | -21.6 | -3.1 |
| NSC | +1.7 | +1.0 | +0.8 | -3.2 | +3.3 | +0.1 | -8.5 | -4.6 | -17.7 | -3.6 |
| SkyMapper | +3.1 | +1.1 | +2.6 | +1.5 | +2.1 | +2.1 | -1.7 | +0.5 | -1.5 | +0.4 |
| VHS | +5.2 | +2.9 | +4.5 | +2.8 | +2.9 | +1.7 | -2.1 | -4.4 | +1.6 | +1.6 |
| DECaLS + ALLWISE | +14.6 | +7.8 | +1.7 | +14.8 | +5.5 | -1.1 | -7.0 | -8.4 | -16.3 | -7.1 |
| SDSS + ALLWISE | +10.4 | +8.8 | +2.4 | +10.5 | +14.1 | +0.6 | -14.9 | -16.0 | -33.1 | -4.3 |
| PS1 + ALLWISE | +9.0 | +6.6 | +4.1 | +10.9 | +5.3 | -1.8 | -14.2 | -15.1 | -40.0 | -4.5 |
| ALLWISE + VHS | +14.6 | +6.8 | +3.7 | +16.7 | +8.1 | +2.8 | -5.5 | -8.4 | -29.0 | +0.3 |
| All seven surveys | +22.9 | +22.9 | +22.3 | +28.7 | +34.3 | +5.0 | -47.1 | -39.4 | -115.7 | -9.7 |

These examples were retained as drawn, without replacing ambiguous cases. Positive evidence for an unclassified field source must not be read as a confirmed rejection failure.

## 1. SDSS

[![SDSS](../../plots/examples/multisurvey/sdss.png)](../../plots/examples/multisurvey/sdss.png)

## 2. DECaLS

[![DECaLS](../../plots/examples/multisurvey/decals.png)](../../plots/examples/multisurvey/decals.png)

## 3. ALLWISE

[![ALLWISE](../../plots/examples/multisurvey/allwise.png)](../../plots/examples/multisurvey/allwise.png)

## 4. PS1

[![PS1](../../plots/examples/multisurvey/ps1.png)](../../plots/examples/multisurvey/ps1.png)

## 5. NSC

[![NSC](../../plots/examples/multisurvey/nsc.png)](../../plots/examples/multisurvey/nsc.png)

## 6. SkyMapper

[![SkyMapper](../../plots/examples/multisurvey/skymapper.png)](../../plots/examples/multisurvey/skymapper.png)

## 7. VHS

[![VHS](../../plots/examples/multisurvey/vhs.png)](../../plots/examples/multisurvey/vhs.png)

## 8. DECaLS + ALLWISE

[![DECaLS + ALLWISE](../../plots/examples/multisurvey/decals_allwise.png)](../../plots/examples/multisurvey/decals_allwise.png)

## 9. SDSS + ALLWISE

[![SDSS + ALLWISE](../../plots/examples/multisurvey/sdss_allwise.png)](../../plots/examples/multisurvey/sdss_allwise.png)

## 10. PS1 + ALLWISE

[![PS1 + ALLWISE](../../plots/examples/multisurvey/ps1_allwise.png)](../../plots/examples/multisurvey/ps1_allwise.png)

## 11. ALLWISE + VHS

[![ALLWISE + VHS](../../plots/examples/multisurvey/allwise_vhs.png)](../../plots/examples/multisurvey/allwise_vhs.png)

## 12. All seven surveys

[![All seven surveys](../../plots/examples/multisurvey/sdss_decals_allwise_ps1_nsc_skymapper_vhs.png)](../../plots/examples/multisurvey/sdss_decals_allwise_ps1_nsc_skymapper_vhs.png)

## Reproduce

```bash
pip install -e ".[figures]"
python scripts/make_multisurvey_examples.py
```

The committed [sample](multisurvey_sample.json) contains all ten objects' native fluxes and variances, parent-cache hashes, source row numbers, and selection settings. The script uses the saved model and needs no WSDB access or training. `--reselect` deliberately regenerates the sample from the original local caches. The [configuration](../../configs/multisurvey_examples.json) declares the survey combinations, plotting bands, redshift window, and seed. [Exact scores](multisurvey_scores.json) record all 120 score rows, band lists, status and quality flags, plotted covariance matrices, and model/sample hashes. No model files are changed.
