# Five-survey QSO gap acquisition

Current operational status is maintained in [PROJECT_STATE.md](../PROJECT_STATE.md).
Current authorization is AllWISE and Pan-STARRS only, then stop. The Legacy bulk count audit is stopped and its
cross-release counts must not be used as training ambiguity masks. Status
statements below describe earlier preparation stages.

The user authorized completing preparation, including acquiring missing results.
No further approval is required for these downloads. Retraining still requires
the agreed readiness report. The active model is unchanged.

## Inputs and reuse

Use `docs/LOCAL_DATA_LOCATIONS.md` and its committed manifests. All 1,657,444
eligible QSO targets remain in the frozen list. The local inventory covers
68,398 AllWISE target results and 68,399 each for PS1, NSC, SkyMapper and VHS,
including nonmatches. These have now been assembled into checked `reuse.npz`
files, with hashes and masks verified, under:

`models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/gap_acquisition/<survey>/`

Each directory also contains `reuse.json`, `status.json`, `position_queries/`
and, for PS1, `id_queries/`. Every query has its SQL, actual plan, input count,
backend PID, start time, duration and result hash in a JSON sidecar. Completed
queries survive interruptions. Raw batches in the acquisition root's
`acquired/<survey>_<start>.npz` retain exact target identities and order and
are consumed by the existing reader and association auditor.

## Query choice and live findings

Schema discovery is saved in `gap_acquisition/schema_preflight.json`. All five
survey tables have Q3C indexes. PS1 also has an indexed bigint `objid`.
The simple spatial LEFT JOIN planned a survey-wide PS1 sequential scan;
the plan guard rejected it before execution. The production positional query
uses an INNER JOIN, with the local target coordinates first in Q3C and
the survey coordinates second. NumPy restores nonmatches, selects the nearest
counterpart and retains the number of candidates. No target is lost. PS1's
flat INNER JOIN also chose a full scan. Its verified production form uses
CROSS JOIN LATERAL with OFFSET 0 to preserve the parameterised Q3C bitmap
lookup; it returns all candidates, without a nearest-only LIMIT. Other surveys
use the simple flat join. Both rejected PS1 plans were stopped before execution.

AllWISE's first three new 10,000-target blocks took 44.4, 50.0 and 50.8 seconds,
using its Q3C index. The first complete aligned batch contains 50,000 rows:
1,995 cached results plus 48,005 newly queried targets. NSC's first new
10,000-target block took 60.1 seconds. These are measured production times,
not cold-cache benchmarks or guarantees for other sky regions/surveys.
The 10,000-target limit is an operational checkpoint size, not a sample cap.
The worker first saves one 50,000-row aligned batch for each survey, then
continues all remaining batches without waiting for user input.

The separate association auditor now reuses saved candidate counts and queries
only rows whose counts remain unknown. PS1 identifier joins do not pretend to
measure the number of spatial neighbours: those counts remain unknown until
checked. No science-quality or blend policy is relaxed.

## PS1 ID correction

The earlier inventory treated integer storage as proof of identifier precision.
Live indexed retrieval exposed the mistake. In the 174,842-row Parquet index,
166,447 IDs equal the float64-rounded versions of the original FITS integers;
147,823 of the 155,286 eligible links need correction. All original integer
values are recoverable locally from the eleven `qso_ps1_full_c*_xmatch.fits`
exports, joined by SDSS name and exact counterpart coordinates. Regression
tests cover this failure. The corrected candidate file and its hashes are saved
in the inventory, and the original Parquet/FITS files remain unchanged.
The first 17 valid results from the old ID probe are retained as well. A live
check of 225 corrected IDs returned 224 valid associations in 4.6 seconds;
those measurements were also merged into reuse, leaving one positional fallback.

## Run and monitor

```bash
python scripts/fetch_qso_photometry_gaps.py \
  --cache models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff \
  --inventory models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/local_inventory/2026-09-28 \
  --surveys allwise ps1
```

The running log is `/tmp/qso_gap_acquisition.log`; durable status and timings
are in `gap_acquisition/`, not only in that temporary log. A single-worker lock
prevents concurrent gap downloaders. The general full-sample fetcher delegates
to this same implementation and refuses a blind full-list pull of these surveys.
Restart with the same command: completed query files and aligned batches are
reused. Do not delete cached nonmatches or infer missing measurements from
missing quality-approved bands.

A zero-match SkyMapper block exposed float-typed empty arrays from sqlutilpy.
The reducer now handles fully empty results explicitly, and a regression test
requires every input target to survive as a recorded nonmatch. Saved preceding
blocks were retained when the worker resumed.
