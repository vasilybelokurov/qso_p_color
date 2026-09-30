# Current project state

**CONVERGENCE REVIEW AND CONTINUATION LAUNCHED (30 September, 00:22 UTC).**
The user authorized a saved-fit diagnostic and up to 100 further iterations.
Coordinator 2282 and monitor 2283 use
`configs/convergence_continuation.json` and `scripts/review_fit_convergence.py`.
Current output: `models/multisurvey_psf/work/convergence_continuation/20260930/`.
Check its `launch.json`, `execution.json`, `run.log`, per-fit status and
`monitor_progress.json` before acting on recorded PIDs. No other trainer should
be started. The QSO phase shares four workers between continuation and decline
investigation; the stellar phase then uses four E-step workers.

The audit found 12 improving unfinished fits: stars and QSO slices
00, 01, 02, 03, 04, 37, 38, 39, 40, 41, 42. These resume at iteration 300,
with unchanged K, data, regularization and stopping tolerance, up to iteration
400. Seven QSO fits (13, 15, 22, 27, 31, 33, 35) have declining likelihoods and
are held: full-data one-step diagnostics compare the configured covariance
addition with a zero-addition counterfactual, without replacing either fit.
All 19 mixtures have finite parameters and positive-definite covariances.
The 25 previously converged QSO fits are retained unchanged. Some of those
converged histories also contain declines; their flags are retained in the full
audit and their convergence is not a certification of predictive stability.

**Follow-up queued (30 September, 00:45 UTC).** The 11 improving QSO continuations
have finished: slices 01 and 40 converged at iterations 398 and 352; the other
nine reached iteration 400 without convergence. Their fixed-row prediction
comparisons are saved. Stellar continuation is active. All seven one-step
covariance-floor diagnostics completed: every full-data likelihood improved
while retaining minimum eigenvalue 0.001 (within roundoff). Evidence:
`docs/COVARIANCE_FLOOR_DIAGNOSTIC_2026-09-30.json`.

The next diagnostic is a separate 20-iteration constrained-update trial on the
seven held slices. `configs/covariance_floor_trial.json` and
`scripts/trial_covariance_floor.py --fit` define it. Its output is
`models/multisurvey_psf/work/covariance_floor_trial/20260930/`; read that
`execution.json` and `launch.json` before scheduling anything. Coordinator 3588 is
waiting for
the current stellar coordinator to finish normally, then use four workers and
save `REPORT.md` and fixed-row prediction comparisons automatically. It does
not change the active model or the existing continuation. This tests whether
the one-step improvement persists before adopting the covariance update in
candidate training and reassessing component selection on the correct roles.

The original scheduled run finished at 23:08 UTC on 29 September; its density-EM
compute is 100% overall, QSO and stellar. This continuation has a separate
row-times-components-times-additional-iterations budget. Prediction comparisons
and decline diagnostics are not included in that EM percentage. The parent
`models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/` remains unchanged,
including its previously assembled density/spatial candidate. Continuation
outputs are individual fits and diagnostics, not an assembled release.

The active model is unchanged. Convergence/capacity review, refreshed spatial
weights after final shapes settle, population priors, calibration-role catch-all
fitting and reserved validation remain. No further acquisition is needed.
Details and restart rules: [continuation record](docs/CONVERGENCE_CONTINUATION_2026-09-30.md).
Initial 19-fit table: [convergence audit](docs/CONVERGENCE_AUDIT_2026-09-30.md).

Updated 30 September 2026 after the user authorized and launched the convergence review.
Read this file before scheduling work. It supersedes
older progress statements in the dated preparation documents; those documents
retain the selection rationale and provenance. `JOURNAL.md` records the history.

## Objective and authorization

Prepare uncapped QSO and stellar training data for the active PSF model, which
supports any nonempty subset of 41 bands, spatial stellar variation and local
refitting. Preserve held-out data. The readiness report was delivered and the
user explicitly authorized retraining on 29 September.
The current model is unchanged; full probability calibration remains unfinished.

**AllWISE and Pan-STARRS are complete; acquisition has stopped as requested.**
Both have saved results for all 1,657,444 targets, including nonmatches. The
worker exited after these two surveys. No acquisition or independent audit
worker is running. NSC, SkyMapper and VHS acquisition remains paused.
Retain and use existing measurements from those surveys, with unavailable
measurements masked; the model's 41-band interface is unchanged. Do not turn
their incomplete acquisition into a requirement to resume downloads. The
independent association auditor remains stopped. The scheduled uncapped density fits are complete; see the run record below.

## Completed data

| Product | Verified state | Location |
|---|---|---|
| Combined SDSS+DESI master | 2,051,328 deduplicated objects; original memberships retained | `~/data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/` |
| Eligible QSO target list | 1,657,444 objects in frozen order | QROOT `targets.npz` |
| QSO SDSS photometry | Results for every eligible target saved, including nonmatches | QROOT; `progress_sdss.json` and the existing batch reader locate old and new caches |
| QSO Legacy photometry | Complete; all 34 batch hashes and identities rechecked; correct hemisphere throughout | QROOT `legacy_local/4d418c295df8aa17/photometry.npz` |
| QSO AllWISE and Pan-STARRS photometry | Complete; all targets covered, row identities and band counts checked in 34 saved batches per survey | QROOT `acquired/allwise_*.npz`, `acquired/ps1_*.npz`; completion manifest linked below |
| Stellar sample | 2,978,857 cleaned entries; seven-survey photometry for all 274 regions | SROOT `photometry/`, `cleaning_report.json` |
| Stellar Legacy selection areas | All 274 regions measured; external-survey effective areas remain a separate issue | SROOT `areas/`, `coverage_report.json` |
| Spatial roles | Fit, selection, calibration and test roles saved, including northern test regions | SROOT `spatial_roles.json` |

QROOT: `models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/`.
SROOT: `models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0/`.
The full location index is [LOCAL_DATA_LOCATIONS.md](docs/LOCAL_DATA_LOCATIONS.md).

## Partially acquired data

NSC, SkyMapper and VHS remain paused and incomplete by the user's decision.
The original inventory established reusable results for about 68,400 targets
per survey, including nonmatches. Further queries and corrected PS1 ID matches
are saved under QROOT `gap_acquisition/`.

Each of these three paused surveys has a 50,000-target assembled prefix, plus
reusable results elsewhere in the target list. This prefix is not total local
coverage or the number of detections. Their full acquisition is not a gate for
the current data effort.

## Association checks: disposition

The stopped Legacy count query includes all catalogue releases, without the
required target hemisphere. It cannot distinguish overlapping survey entries
from alternative counterparts. Its `multiple_matches` flags must not become
training exclusion masks. Preserve the results as historical diagnostics;
do not restart this bulk audit or replace it with another blanket query.

Local checks have already verified Legacy source provenance, target identities,
separations, hemispheres and file hashes. A further local check found two target
rows sharing an exact Legacy release and counterpart position; their identities
are saved in QROOT `legacy_local/4d418c295df8aa17/shared_counterpart_review.npz`
for focused review. No selection masks were changed. This local check does not
certify the absence of every possible nearby alternative counterpart.

Use saved candidate counts from new positional acquisitions and existing
association results where applicable. Keep unknown counts explicit. Before any
additional query, establish which missing fact affects the training selection,
why local evidence cannot supply it, and the smallest necessary target set.

## QSO fitting completed (29 September, 15:28 UTC)

All 43 QSO final slice files are saved: 25 meet the configured convergence
criterion; 18 reached the 300-iteration limit and require convergence review.
The QSO worker pool has exited. The subsequent four-worker stellar fit has also finished.
Verified per-slice row counts and frozen roles, 41-dimensional finite mixtures,
positive-definite covariances and unchanged active pointer. Evidence is in
`models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d/qso_completion.json`.
At QSO completion, density-EM compute was about 53.0% overall and 22.3% stellar.
The final run is now 100% for both populations.
This completes the scheduled QSO fits, not scientific validation or release.
See the completed stellar run details below.

## Training run and remaining work

The requested local assembly and uncapped density-training integration are
complete. Inputs are in
`models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8/`, with
checksummed memory-mapped arrays and an assembly report. The final shape fits
used all 1,049,260 QSO fit+selection rows and 1,975,894 stellar fit+selection
rows. Calibration and test roles remain separate. No new downloads were launched.

The current launch record, log and preflight remain in
`models/multisurvey_psf/work/full_training_runs/20260929T045901Z/`.
The parent `current.json` locates this run. `stellar_parallel_train.log` is
now the training log; the earlier serial and QSO-parallel logs and launch
records are retained as history. The original coordinator and its four idle
QSO workers exited at a checkpoint-safe handover.

Historical handover command: `scripts/train_stellar_parallel.py --workers 4 --task-rows 8192
--resume-from models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d`
was used to continue the saved fit. This run is complete; do not relaunch it
from this historical command. Both parent and destination run locks reject
duplicate fitting. The completed continuation directory is
`models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/`.
Its `checkpoint_lineage.json` records all transferred hashes. The changed
streaming-engine implementation has its own identity; the old directory is
preserved. The statistical model, frozen roles, selected K=20, regularization,
convergence tolerance and total iteration limit are unchanged.

Four workers evaluate disjoint batches, then their sufficient statistics are
combined before one global M step. All 1,975,894 stars enter every iteration.
A real-data benchmark measured 26.84 seconds serial versus 7.81 seconds with
four workers (3.43x), with only floating-point rounding differences. Tests
cover serial-checkpoint continuation, worker failure without partial updates,
and unchanged completed QSO fits. Details: `docs/STELLAR_PARALLEL_2026-09-29.md`.
The active model pointer remains unchanged.

Progress monitoring ran independently every 60 seconds through
`scripts/monitor_full_training.py --watch`. In the run directory,
`PROGRESS.md` is the readable live summary, `progress.json` is its structured
form, and `progress_history.jsonl` records successive checks. The monitor
reports checkpoint iterations, likelihoods, completed fits, CPU/memory,
the four stellar workers, within-pass row counts and worker exit; it stops when the density candidate completes or the worker exits.
Every progress report must include estimated overall compute completion,
plus QSO and stellar percentages. Weight saved EM iterations by fitted rows
and component count; budget unfinished fits at their iteration limits and
maximum K until selection is known. Clearly label this as density-EM work,
not wall-clock progress or completion of priors/catch-all/validation.
It does not deliver chat notifications. Check `monitor.lock` and `monitor.log`
for the monitor PID and errors before starting another monitor.

The unchanged scientific configuration is `configs/full_sample_training.json`.
`scripts/train_full_sample.py` remains the serial/preflight entry point, but
the current continuation uses `scripts/train_stellar_parallel.py` as above. Read
[the readiness report](docs/FULL_TRAINING_READINESS_2026-09-29.md) before launch.
The old capped recovery scripts are historical and are not this run's entry point.

1. Review convergence of the 18 capped QSO fits and the capped stellar fit,
   together with component selection at the edge of the tested capacity grid.
   The authorized continuation and diagnostic run is recorded at the top of this file;
   preserve its fixed-K scope before beginning a separate capacity review.
2. After density fits, complete population priors and calibration-role catch-all
   fitting consistently with the new shapes. The historical population-completion
   script needs adaptation to the new role/input layout; do not run it unchanged.
   Review external-survey count-prior area assumptions at that stage. This is
   separate from the completed colour-density input preparation.
3. Assess convergence, selected capacity and reserved validation before any
   promotion. Full probability calibration is not established by these steps.

## Working rules

- Read this state, the location index and recent journal entries before acting.
- Before expensive work, record the missing fact, its purpose, local evidence,
  proposed query and measurable completion criterion. An old checklist item is
  not sufficient justification.
- Update this file after substantive progress or a changed decision. Keep
  historical findings in the journal rather than appending conflicting status
  paragraphs to several documents.
- Report acquisition, local assembly and validation separately. Use saved
  output counts and measured timings; a running process alone is not progress.
- Acquisition is finished for the agreed scope. Do not restart downloads from
  an older checklist. The readiness gate was satisfied and the user authorized this training run.

Evidence: [local reconciliation](docs/PREPARATION_RECONCILIATION_2026-09-28.json).
Download completion: [AllWISE and Pan-STARRS verification](docs/ALLWISE_PS1_COMPLETION_2026-09-29.json).
