# Bounded implementation: full-data candidate completion

The user authorized items 1–3 of the recovery plan on 30 September. Retain the
assembled 300-iteration density and spatial model, investigate the bright-star
optical regression, and finish a coherent prior/catch-all bundle. Activation
and final complete-score release checks are subsequent work.

1. Freeze the density model and spatial weights by content hash. The continuation
   and covariance-floor experiments remain separate. No density fitting or
   downloads are needed.
2. Build stellar surface-density priors from every fit+selection source and the
   saved cone areas. Use measured per-band Legacy optical areas. For other bands,
   retain the existing approximate convention (base usable cone area where the
   survey has measurements), explicitly flagged in each prior. This does not
   establish external-survey footprint accuracy. Retain the QSO abundance prior
   numerically: the new spectroscopic training selection is not a new abundance
   calibration. Keep the same reference grids, merging empty stellar bins only.
3. Recompute the broad Student-t shape from the frozen full-data stellar mixture.
   Retain the selected degrees of freedom, scale factor, exact noise convolution,
   three share bins and positive pooled-share estimator. Fit shares on **all**
   502,251 eligible calibration-role stellar sources, with resumable density
   evaluation. Calibration/test rows never enter counts or density shapes.
4. Diagnose the bright-star issue on a fixed random sample from the calibration
   role, with thresholds and redshifts declared before inspection. Compare
   constituent optical surveys, joint optical bands, hemisphere/reference bands,
   raw QSO/stellar evidence, catch-all-adjusted evidence and the existing hard
   support guard. This diagnoses evidence, not calibrated posterior probabilities.
   Do not change the density model unless this identifies a consequential cause.
5. Save a loadable, hashed candidate bundle and a completion report. Verify 41-band
   identity, prior provenance, role separation, positive catch-all shares,
   unchanged density/spatial contents and unchanged active pointer. Run the test
   suite and record the actual results in PROJECT_STATE.md and JOURNAL.md.

The missing facts are whether the bright optical density loss creates excessive
QSO evidence, and whether the completed bundle remains internally consistent.
Local arrays, frozen role manifests, prior artifacts and saved areas suffice.
Configuration: `configs/full_sample_completion.json`. No final-test tuning or
new optimizer/capacity search is part of this implementation.
