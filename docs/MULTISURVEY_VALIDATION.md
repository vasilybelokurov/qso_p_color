# Multi-survey model validation

Run `9ca579b6b507`; model `models/multisurvey.json`.

The method and these tests are in the main text of the [method note](method/method.pdf). The [matched old/new comparison](MODEL_COMPARISON.md) and [fresh-object confirmation](MODEL_COMPARISON_FRESH.md) document performance relative to the original model. Survey-combination rows alone do not establish equivalence.

High-latitude availability-selected subsets; at most 200 objects per class and combination. Different survey rows use different available objects. Redshift PDFs use a flat z prior; coverage is a diagnostic, not a calibration claim. Population posteriors require separate matched surface-density priors.

The quasar fit uses 67,574 objects, with 14,766 reserved in the existing spatial holdout. It spans 43 slices over model support 0.15–4.35; the selected component counts (K: number of slices) are `{'2': 6, '4': 37}`.

The original field sample has 173,054 sources from 24 fields, with 41,241 sources in reserved fields. The background fit uses 20,000 sampled rows with weights restoring the training population and K=8.

An additional 1 overlap field contributes 6,825 exclusively held-out sources. It was selected for survey coverage before inspecting validation scores and is absent from all fitting and component selection. Its coordinates and selection rule are recorded in the validation configuration.

The requested EM tolerance was reached by 2/43 quasar slices; background convergence: False. Fits that reached the iteration cap are recorded as such, not declared converged. The held-out results below assess the saved fits at that stopping point.

The median absolute change in mean training log density over the final quasar-fit iteration was 0.00022 nats per object.

The public scorer was checked against a separate batched calculation for 127 real-data combinations. The unit tests exercise all 127 non-empty survey subsets, including two-band infrared-only inputs.

| Surveys | Held-out QSOs available | QSOs / field objects evaluated | QSO vs field AUC | True vs other z AUC | Median absolute dz/(1+z) | 68% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| sdss | 10137 | 200 / 200 | 0.943 | 0.923 | 0.048 | 0.705 |
| decals | 13685 | 200 / 200 | 0.935 | 0.886 | 0.089 | 0.700 |
| allwise | 8244 | 200 / 200 | 0.903 | 0.770 | 0.199 | 0.725 |
| ps1 | 14235 | 200 / 200 | 0.938 | 0.845 | 0.068 | 0.735 |
| nsc | 11765 | 200 / 200 | 0.902 | 0.847 | 0.089 | 0.740 |
| skymapper | 1941 | 200 / 200 | 0.883 | 0.772 | 0.220 | 0.655 |
| vhs | 2002 | 200 / 200 | 0.620 | 0.779 | 0.252 | 0.745 |
| sdss + allwise | 6291 | 200 / 200 | 0.994 | 0.961 | 0.035 | 0.760 |
| decals + allwise | 7898 | 200 / 200 | 0.986 | 0.928 | 0.052 | 0.770 |
| allwise + ps1 | 8190 | 200 / 200 | 0.986 | 0.918 | 0.046 | 0.745 |
| allwise + vhs | 1847 | 200 / 200 | 0.928 | 0.841 | 0.086 | 0.770 |
| sdss + allwise + ps1 | 6281 | 200 / 200 | 0.989 | 0.946 | 0.031 | 0.790 |
| sdss + decals + allwise + ps1 + nsc + skymapper + vhs | 616 | 200 / 145 | 1.000 | 0.974 | 0.022 | 0.810 |

Infrared-only inputs receive scores, but their information content differs. On the available held-out samples, ALLWISE-only gives QSO/field AUC 0.903; VHS-only gives 0.620, so VHS-only classification is weak. These rows use different objects and do not measure the improvement from adding a survey to the same sources.

The QSO-versus-field AUC uses colour evidence at the true redshift for quasars and assigned primary redshifts for field objects. The true-versus-other-redshift AUC scores each withheld quasar at its true redshift and at a primary redshift drawn from the withheld population outside the declared matching window, using the normalised quasar-only redshift density. These are discrimination checks, not physical-pair probabilities or substitutes for the field-quasar term in scoring.

The data use native observed photometry, with quality cuts and high-latitude selection recorded in `configs/multisurvey.json`. The background contains all source types and excludes known quasars. Whole fields were reserved before fitting; internal component selection used additional fields from the training partition.

The [full machine-readable report](MULTISURVEY_VALIDATION.json) includes all 127 combinations, per-band holdout counts, and fit diagnostics. A local copy is also kept in `data/multisurvey/validation.json`. The existing southern models were retained.

The declared southern Legacy grz background marginal uses 13,233 existing training objects; 2,279 of these were reserved in the original internal selection fields to choose K=16. The final fit converged: True. It is used only when it contains every observed input band. Quasar fits, transforms, and the full joint background are unchanged.
