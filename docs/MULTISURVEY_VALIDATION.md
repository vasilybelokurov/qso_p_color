# Multi-survey model validation

Run `22820c906d75`; model `models/multisurvey.json`.

The method and these tests are in the main text of the [method note](method/method.pdf). The [matched old/new comparison](MODEL_COMPARISON.md) and [fresh-object confirmation](MODEL_COMPARISON_FRESH.md) document performance relative to the original model. Survey-combination rows alone do not establish equivalence.

High-latitude availability-selected subsets; at most 200 objects per class and combination. Different survey rows use different available objects. Redshift PDFs use a flat z prior; coverage is a diagnostic, not a calibration claim. Population posteriors require separate matched surface-density priors.

The quasar fit uses 67,574 objects, with 14,766 reserved in the existing spatial holdout. It spans 43 slices over model support 0.15–4.35; the selected component counts (K: number of slices) are `{'2': 10, '4': 33}`.

The original field sample has 174,567 sources from 24 fields, with 42,493 sources in reserved fields. The background fit uses 20,000 sampled rows with weights restoring the training population and K=8.

An additional 1 overlap field contributes 6,825 exclusively held-out sources. It was selected for survey coverage before inspecting validation scores and is absent from all fitting and component selection. Its coordinates and selection rule are recorded in the validation configuration.

The requested EM tolerance was reached by 1/43 quasar slices; background convergence: False. Fits that reached the iteration cap are recorded as such, not declared converged. The held-out results below assess the saved fits at that stopping point.

The median absolute change in mean training log density over the final quasar-fit iteration was 0.00028 nats per object.

The public scorer was checked against a separate batched calculation for 127 real-data combinations. The unit tests exercise all 127 non-empty survey subsets, including two-band infrared-only inputs.

| Surveys | Held-out QSOs available | QSOs / field objects evaluated | QSO vs field AUC | True vs other z AUC | Median absolute dz/(1+z) | 68% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| sdss | 10137 | 200 / 200 | 0.944 | 0.921 | 0.052 | 0.705 |
| decals | 13745 | 200 / 200 | 0.985 | 0.927 | 0.047 | 0.750 |
| allwise | 8244 | 200 / 200 | 0.913 | 0.724 | 0.210 | 0.655 |
| ps1 | 14235 | 200 / 200 | 0.943 | 0.842 | 0.055 | 0.720 |
| nsc | 11765 | 200 / 200 | 0.890 | 0.843 | 0.079 | 0.765 |
| skymapper | 1941 | 200 / 200 | 0.839 | 0.794 | 0.191 | 0.685 |
| vhs | 2002 | 200 / 200 | 0.662 | 0.754 | 0.234 | 0.740 |
| sdss + allwise | 6291 | 200 / 200 | 0.987 | 0.948 | 0.040 | 0.820 |
| decals + allwise | 7898 | 200 / 200 | 0.990 | 0.895 | 0.055 | 0.745 |
| allwise + ps1 | 8190 | 200 / 200 | 0.986 | 0.920 | 0.046 | 0.745 |
| allwise + vhs | 1847 | 200 / 200 | 0.931 | 0.842 | 0.085 | 0.770 |
| sdss + allwise + ps1 | 6281 | 200 / 200 | 0.989 | 0.943 | 0.029 | 0.800 |
| sdss + decals + allwise + ps1 + nsc + skymapper + vhs | 616 | 200 / 145 | 1.000 | 0.981 | 0.022 | 0.790 |

Infrared-only inputs receive scores, but their information content differs. On the available held-out samples, ALLWISE-only gives QSO/field AUC 0.913; VHS-only gives 0.662, so VHS-only classification is weak. These rows use different objects and do not measure the improvement from adding a survey to the same sources.

The QSO-versus-field AUC uses colour evidence at the true redshift for quasars and assigned primary redshifts for field objects. The true-versus-other-redshift AUC scores each withheld quasar at its true redshift and at a primary redshift drawn from the withheld population outside the declared matching window, using the normalised quasar-only redshift density. These are discrimination checks, not physical-pair probabilities or substitutes for the field-quasar term in scoring.

The data use native observed photometry, with quality cuts and high-latitude selection recorded in `configs/multisurvey.json`. The background contains all source types and excludes known quasars. Whole fields were reserved before fitting; internal component selection used additional fields from the training partition.

The [full machine-readable report](MULTISURVEY_VALIDATION.json) includes all 127 combinations, per-band holdout counts, and fit diagnostics. A local copy is also kept in `data/multisurvey/validation.json`. The existing southern models were retained.

The declared southern Legacy grz background marginal uses 13,239 existing training objects; 2,597 of these were reserved in the original internal selection fields to choose K=32. The final fit converged: True. It is used only when it contains every observed input band. Quasar fits, transforms, and the full joint background are unchanged.
