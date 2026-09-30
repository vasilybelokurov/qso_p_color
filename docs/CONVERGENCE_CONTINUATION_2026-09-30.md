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

## Covariance-floor investigation and queued trial

The seven full-data one-step diagnostics finished. Additive covariance
regularization reduced the likelihood at the next step for six slices; slice
13 improved slightly at that next step despite its preceding declining history.
All seven zero-addition counterfactuals improved. A constrained M step that
replaces only eigenvalues below 0.001 with that existing minimum variance also
improved every slice: gains were 0.114996, 0.122493, 0.097767, 0.107922,
0.125420, 0.142047 and 0.180281 nats/object for slices 13, 15, 22, 27, 31, 33
and 35, respectively. Minimum resulting covariance eigenvalues stayed at or
above the declared floor, to floating-point precision. The full-data results
are saved as `covariance_floor_diagnostic.json` in the continuation directory.

This is the exact constrained Gaussian covariance M step, applied to the XD
conditional sufficient statistics. It retains protection against covariance
collapse without adding variance to every direction at every iteration.
See the derivation in the method note. Related constrained mixture updates are
described in [Browne, Subedi and McNicholas, section 3.1](https://arxiv.org/html/1306.5824).
The XD extension here follows from the conditional-scatter objective and is
checked with missing dimensions, correlated measurement errors and row weights.
No production trainer has been switched to this update.

Next is a separate 20-iteration diagnostic trial for the seven held slices,
using every original fit+selection row and the original component counts.
Configuration: `configs/covariance_floor_trial.json`; entry point:
`scripts/trial_covariance_floor.py --fit`. It waits for the existing stellar
continuation to complete normally before using four QSO workers. Failure or
disappearance of that coordinator blocks the trial rather than starting an
unrelated job. Output: `models/multisurvey_psf/work/covariance_floor_trial/20260930/`.
`queued_identity.json` pins code, configuration and audit while waiting;
`lineage.json` records parent/input hashes. Each iteration checks the full-data
likelihood and saves a restart checkpoint. A decrease beyond configured
floating-point tolerance stops the trial. Twenty iterations is a diagnostic
budget, not a convergence claim. Trial convergence histories begin with the
new covariance update; the original iteration 300 remains explicit provenance.

The queued job saves per-slice prediction comparisons, `results.json`, and a
readable `REPORT.md` automatically. The next decision is whether this update
should enter the candidate training path, followed by fit-only/selection-role
model-selection checks. The final test sample remains untouched. The active
bundle, old checkpoints and ongoing stellar fit remain unchanged.

## Completed additive-update continuation

The authorized continuation finished at 03:13 UTC. Both QSO and stellar
additional-EM budgets are 100% complete. Two further QSO slices converged
(01 at 398 iterations; 40 at 352). Nine other continued slices and the stars
reached 400 without convergence. The 25 already-converged QSO fits remain
unchanged, and the seven declining slices remain held. Detailed outcomes are
in `docs/CONVERGENCE_CONTINUATION_RESULTS_2026-09-30.json`.

On the fixed 8,192-row stellar diagnostic sample, the median absolute change
in conditional log density was 0.0249 nat, the 95th percentile was 0.1625 and
the maximum was 4.678. For QSO slice 04 these values were 0.1972, 0.9450 and
12.063 nats. These changes compare iterations 300 and 400 on identical
measurements. They are not posterior-probability changes or independent
validation. Large changes for particular objects mean that completion of the
iteration budget cannot be described as blanket predictive stability.

The first queued covariance-floor trial failed before fitting because the
sandbox prohibited its process-health check. The launch record is retained as
`launch_failed_sandbox.json`. It was restarted with the required permission at
05:48 UTC, after verifying stellar completion; its execution record
identifies the restarted coordinator. No completed fitting work was repeated.

## Completed covariance-floor trial and recommendation

All seven slices completed 20 constrained updates. Every one of the 140
updates increased the full-data likelihood; total gains ranged from 0.3995
to 0.7070 nats/object. Covariance eigenvalues remained at or above the 0.001
bound within floating-point precision. None met the original convergence
tolerance in this short trial. The summary and paths to detailed prediction
comparisons are in `docs/COVARIANCE_FLOOR_TRIAL_2026-09-30.json`. All workers
from both diagnostic runs have exited.

The recommendation is to integrate this constrained covariance update in the
shared candidate training path, retaining additive updates as an explicit
historical reproduction option. It addresses the measured optimization defect
without removing the minimum-variance safeguard. Component selection must then
be checked with fit-only shapes evaluated on the selection role; the final
fit+selection models cannot supply independent selection scores. Previously
converged additive fits also need review under the consistent update, rather
than treating their old stopping flags as convergence of a different update.
All scientific settings and changed update identities must remain explicit.

After component selection and final-shape convergence, rebuild spatial weights
and complete priors, calibration-role catch-all fitting and reserved validation.
These training diagnostics do not establish probability calibration or support
for every sparsely sampled band combination. The active bundle remains unchanged.
