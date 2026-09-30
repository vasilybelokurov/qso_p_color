# Full-row covariance comparison

The user authorized this diagnostic, then asked to wait because other work
occupies ten CPU cores. It is prepared, **not running or automatically queued**.
The production method and active model remain unchanged.

The decision is whether constrained covariance updates preserve predictive
quality while removing the likelihood declines caused by additive broadening.
The earlier 2,048-row experiment cannot settle this for the large slices or
background. The high-redshift slice itself has only 2,152 training objects.

| Case | All fit/select rows | Why included |
|---|---:|---|
| QSO z=1.45 | 114,657 | Previously declining likelihood |
| QSO z=4.05 | 2,152 | Slow fit and poor miniature floor result |
| Empirical PSF background | 1,975,894 | Enters every QSO comparison |

Both updates use identical warm starts and all quality-eligible training rows,
including weak and negative measurements. One method adds 0.001 to every
covariance eigenvalue; the other imposes a lower bound of 0.001 mag squared.
This is an experimental comparison, not a change to the production optimizer.

The script uses four independent QSO workers and four cooperating background
workers. The two background methods run sequentially within each block.
Every full pass is checkpointed. Interrupted blocks can resume with row and
run identity checks. Completed blocks compare training likelihood, non-training
conditional density, covariance spectra and effective component membership.
The already computed boundary sufficient statistics are reused on continuation.

The budget is 60 updates per fit, in blocks of 20, and four hours after launch.
An E step already in progress finishes before the wall-budget stop; scoring
overhead can also take the run past the nominal deadline. A declared predictive
plateau can stop after at least two blocks. Reaching a budget without a plateau
is explicitly inconclusive on stopping. No production promotion is automatic.
The thresholds and seed are in `configs/unified_covariance_trial.json`.

## Evaluation and separation

Development density panels contain up to 512 objects per hemisphere per case,
with at least one five-sigma detection and at least two observed bands. This
is an evaluation domain, never a training cut. The complete scorer also uses
128 objects per case, an unstratified random panel of 1,024 detectable background
objects, and the existing North/South bright and intermediate colour grids.

The exact object lists are saved before fitting. Their union consists of
1,646 QSO and 2,045 background objects from role 3. Source-row indices in
`final_assessment_exclusions.npz` refer to the immutable full-training inputs.
**Every future independent final assessment must exclude these rows**, along
with previously inspected development subsets. The original role arrays are
not rewritten; role-2 rejection/calibration data are untouched. Merely selecting
role 3 in an older validation script does not provide that exclusion.

The complete scorer retains same-redshift and other-redshift QSOs, background,
catch-all, spatial dependence and existing guards. All 43 QSO slices start
from the full-data canonical parent; only the two trial slices and background
change. Pilot priors, catch-all and spatial component weights are held fixed
between methods to isolate shape changes. Consequently, this is a conditional
classification-sensitivity diagnostic, not a completed refit with refreshed
spatial weights or a probability-calibration claim.

Saved outputs include raw and guarded rankings, ranking AUC, score changes,
crossings of the existing high-QSO threshold, covariance spectra and the
best development-density checkpoint. A baseline density-quantile proxy records
high-QSO/low-density counts in the random background panel. This proxy is not
a calibrated support test, and these counts are not known contamination rates
or an area-weighted all-sky incidence estimate.

Zero target-redshift prior weights legitimately produce negative-infinite
ranking scores. They stay at the bottom of the ranking; changes into or out of
zero weight are counted separately. The initial launch check exposed a
diagnostic that rejected five such objects before any fitting began. It was
corrected and regression-tested; no density fit was lost.

## Locations and restart

Prepared root:
`models/multisurvey_psf/work/unified_covariance_trial/20260930/8176eab7484bf5e1`.

- `prepared.json`, `development_manifest.json`: row accounting and exclusions.
- `options.json`, `identity.json`: exact configuration and implementation hashes.
- Each case/method directory: full-pass checkpoint, progress, models and predictions.
- After launch: `execution.json`, `progress.json`, `PROGRESS.md`, `run.log`.
- After each completed block: `comparison_NNN.json`, `scores_NNN.json` and score arrays.
- At termination: `finished.json`; an exception instead records a failed execution.

`python scripts/trial_unified_covariance.py` only prepares.
`python scripts/trial_unified_covariance.py --fit` runs the diagnostic.
**Wait for the user's resumption before launching; recheck CPU availability.**

At the pause, the machine had 48 GiB memory, 14 CPU cores (10 performance),
1.2 TiB free disk, approximately 1 GiB allocated swap with no swapping observed
during the short sample, and 80--90% CPU use. Two unrelated Python jobs used
six and four workers. No other process was stopped or modified.

Validation: 372 tests passed in 94.86 seconds, including exact restart/full-row
accounting, training/calibration exclusion, and zero-prior ranking regression.
Log: `/tmp/unified_covariance_trial_fullsuite_final.log`.
