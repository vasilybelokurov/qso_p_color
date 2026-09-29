# Current project state

Updated 29 September 2026 after the user authorized full-sample retraining
and the training worker was launched. Read this file before scheduling work. It supersedes
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
independent association auditor remains stopped. Uncapped density retraining is now launched; see the run record below.

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

## Training run and remaining work

The requested local assembly and uncapped density-training integration are
complete. Inputs are in
`models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8/`, with
checksummed memory-mapped arrays and an assembly report. The final shape fits
will use all 1,049,260 QSO fit+selection rows and 1,975,894 stellar fit+selection
rows. Calibration and test roles remain separate. No new downloads were launched.

Training worker **48795** was launched at **2026-09-29 04:59:01 UTC** from
commit `be2a5fb`, using the unchanged prepared configuration. Its durable run
record, log and launch preflight are in
`models/multisurvey_psf/work/full_training_runs/20260929T045901Z/`.
The parent `current.json` locates this run. Verify the process and saved
checkpoints before reporting subsequent progress; do not launch a duplicate.
The output directory is
`models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d/`.
Launch preflight passed and both full-population initializations are saved;
the worker is fitting stellar component-selection candidates first. The active model remains unchanged.

Use `scripts/train_full_sample.py` and `configs/full_sample_training.json`.
The default runs preflight only; `--fit` explicitly launches uncapped shape
and spatial-weight fitting. Read
[the readiness report](docs/FULL_TRAINING_READINESS_2026-09-29.md) before launch.
The old capped recovery scripts are historical and are not this run's entry point.

1. Monitor the authorized density fit through saved iteration checkpoints and
   completed component-selection records. At iteration limits the uncapped
   workload can take several days; every iteration is checkpointed.
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
