# Fixed-component convergence review and continuation

The user authorized the saved-fit audit and up to 100 further iterations on
30 September. The initial audit is in `CONVERGENCE_AUDIT_2026-09-30.md` and
the corresponding JSON. All 19 unfinished mixtures have finite parameters and
positive-definite covariances. Twelve have improving recent likelihoods: the
stars and QSO slices 00, 01, 02, 03, 04, 37, 38, 39, 40, 41, 42. Seven slices
have material declines: 13, 15, 22, 27, 31, 33, 35. These are held for diagnosis.
The 25 previously converged QSO results remain unchanged.

Configuration: `configs/convergence_continuation.json`.
Entry point: `scripts/review_fit_convergence.py`; default audit only, `--fit`
explicitly enables the authorized work. All original scientific settings come
from the parent's identical training configuration. Component counts and
selection results are fixed during this review. No data are downloaded.

Parent: `models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/`.
Output: `models/multisurvey_psf/work/convergence_continuation/20260930/`.
The output contains individual fits and diagnostics, not an assembled release.
`lineage.json` records original input, implementation, configuration and parent
artifact hashes, together with the new execution settings and engine hashes.
Both directories are locked during fitting; the parent is never overwritten.

The coordinator runs four QSO processes, sharing that pool between slice
continuations and seven full-data decline diagnostics. It closes this pool
before starting the stellar fit with four E-step workers and one global M step.
Every training pass uses the exact original fit+selection row IDs, verified by
hash. No calibration or final-test rows enter training or diagnostic samples.
All continuations start at saved iteration 300 and stop at convergence or 400.
A material likelihood reversal also stops continuation for investigation.
Checkpoints retain the full history, so restart does not reset the iteration
budget. Existing saved final results are reused on a matching restart.

For each declining slice, one full-data E step supplies sufficient statistics
for two single-update diagnostics: the configured added covariance diagonal,
and a zero-diagonal counterfactual. Both updates are evaluated on all of that
slice's original training rows. Neither diagnostic mixture replaces a saved
fit. The comparison tests whether this covariance regularization explains the
observed likelihood declines; it does not authorize removing regularization.

Each continued fit has a deterministic diagnostic sample of up to 8,192
training rows. This sample limits only the before/after sensitivity calculation,
never an EM pass. Saved rows and measurements are identical in both evaluations.
Reports contain joint-density and magnitude-conditioned density changes;
conditional changes are grouped by exact observed band mask and reference
magnitude quantile. Lower-density tails are defined within those groups, not by
comparing raw densities across different dimensionalities. Small strata retain
their counts. These are sensitivity diagnostics, not independent validation or
certification of all sparsely sampled band combinations.

`execution.json` records the coordinator and phase. Per-fit `*.status.json`
records iterations and state. `*.final.checkpoint.json` stores restart points;
`*.decline_diagnosis.json` stores one-step comparisons. Probe arrays and
`*.prediction_change.json` retain before/after evidence. `after_audit.json`
summarizes all returned fits. Completion of this run is not convergence of
every fit or release readiness.

The monitor command is `scripts/review_fit_convergence.py --monitor`. It writes
`monitor_progress.json` once per minute and exits when the coordinator finishes
or disappears. Progress weights additional saved EM iterations by row count and
component count, using actual iterations for finished fits and 100 for ongoing
fits. It excludes decline diagnostics, prediction comparisons, spatial weights,
population priors, catch-all fitting and reserved validation. The original
density run remains 100% complete; this continuation has a separate budget.

Before restarting, read PROJECT_STATE.md and verify all recorded PIDs. Restart
only the same configuration and implementation; changed lineage is rejected.
After results are reviewed, spatial weights and the density candidate must be
rebuilt from the settled shapes before population completion and validation.
