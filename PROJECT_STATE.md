# Current project state

Updated 28 September 2026 after the user limited further acquisition to
AllWISE and Pan-STARRS. Read this file before scheduling work. It supersedes
older progress statements in the dated preparation documents; those documents
retain the selection rationale and provenance. `JOURNAL.md` records the history.

## Objective and authorization

Prepare uncapped QSO and stellar training data for the active PSF model, which
supports any nonempty subset of 41 bands, spatial stellar variation and local
refitting. Preserve held-out data. Report readiness before launching retraining.
The current model is unchanged; full probability calibration remains unfinished.

**Acquire AllWISE and Pan-STARRS only, then stop.** The user considers these
additions sufficient for the current data effort. The downloader runs with
`--surveys allwise ps1`, completing AllWISE first and then Pan-STARRS; it exits
after those two surveys. NSC, SkyMapper and VHS acquisition remains paused.
Retain and use existing measurements from those surveys, with unavailable
measurements masked; the model's 41-band interface is unchanged. Do not turn
their incomplete acquisition into a requirement to resume downloads. The
independent association auditor remains stopped. Retraining still waits for
the readiness report.

## Completed data

| Product | Verified state | Location |
|---|---|---|
| Combined SDSS+DESI master | 2,051,328 deduplicated objects; original memberships retained | `~/data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/` |
| Eligible QSO target list | 1,657,444 objects in frozen order | QROOT `targets.npz` |
| QSO SDSS photometry | Results for every eligible target saved, including nonmatches | QROOT; `progress_sdss.json` and the existing batch reader locate old and new caches |
| QSO Legacy photometry | Complete; all 34 batch hashes and identities rechecked; correct hemisphere throughout | QROOT `legacy_local/4d418c295df8aa17/photometry.npz` |
| Stellar sample | 2,978,857 cleaned entries; seven-survey photometry for all 274 regions | SROOT `photometry/`, `cleaning_report.json` |
| Stellar Legacy selection areas | All 274 regions measured; external-survey effective areas remain a separate issue | SROOT `areas/`, `coverage_report.json` |
| Spatial roles | Fit, selection, calibration and test roles saved, including northern test regions | SROOT `spatial_roles.json` |

QROOT: `models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/`.
SROOT: `models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0/`.
The full location index is [LOCAL_DATA_LOCATIONS.md](docs/LOCAL_DATA_LOCATIONS.md).

## Partially acquired data

QSO AllWISE, Pan-STARRS, NSC, SkyMapper and VHS results remain incomplete.
The original inventory established reusable results for about 68,400 targets
per survey, including nonmatches. Further queries and corrected PS1 ID matches
are saved under QROOT `gap_acquisition/`.

At the pause, the assembled prefix contains 100,000 AllWISE targets and 50,000
targets in each of the other four surveys. Additional query results are saved
beyond these prefixes. **These assembly counters are not total local coverage
or the number of detections.** Do not derive download gaps by subtracting them
from the full target count. The downloader uses the saved inventory and query
caches to identify the actual gaps.

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

## Remaining data work

1. Finish only the demonstrated QSO gaps in AllWISE and Pan-STARRS, then stop
   acquisition. Do not proceed to NSC, SkyMapper or VHS.
   Completion means a recorded measurement or nonmatch for every eligible target
   in each of these two surveys, with no repeated acquisition of known results.
2. Assemble QSO training inputs from the available results in the 41-band
   representation, retaining missing-band masks for incomplete surveys. Verify identities,
   quality and association flags, band masks and counts by the frozen spatial
   roles. Review the identified Legacy shared-counterpart rows locally first.
   Unresolved association questions must be stated, not silently passed.
3. Review external-survey coverage and effective-area assumptions needed for
   count priors, using the saved stellar coverage report. Distinguish the needs
   of colour-density fitting from those of surface-density priors. Do not
   prescribe new downloads before establishing the missing information.
4. Produce the training-readiness report: selection counts, disjoint roles,
   band coverage, uncapped trainer accounting, runtime/memory and output paths.
   Training, promotion and probability calibration are separate later steps.

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
- Continue the authorized two-survey acquisition; do not repeatedly
  ask for permission. Preserve the explicit readiness gate before retraining.

Evidence: [local reconciliation](docs/PREPARATION_RECONCILIATION_2026-09-28.json).
