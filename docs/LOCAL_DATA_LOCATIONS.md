# Local data locations and reuse inventory

Current operational status is maintained in [PROJECT_STATE.md](../PROJECT_STATE.md).
AllWISE and Pan-STARRS acquisition is complete and the worker has stopped (29 September). The Legacy bulk count audit is stopped and its
cross-release counts must not be used as training ambiguity masks. Status
statements below describe earlier preparation stages.

Updated 28 September 2026. Read this before any photometry acquisition.
The inventory made no database queries and changed no model or acquisition output.

## Roots and current products

Repository root:
`/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/qso_p_color`

Permanent catalogue store: `/Users/vasilybelokurov/data/qso_p_color/`.
Repository `data/` resolves to this store's `project_data/`; these are the same
files, so do not count them twice. Paths beginning `models/` below are relative
to the repository root. These local data products are gitignored; the location
map, inventory summaries and file manifests in `docs/` are committed.

| Contents | Exact location |
|---|---|
| Deduplicated SDSS+DESI master, all memberships and provenance | `/Users/vasilybelokurov/data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/` (`objects.npz`, `members.npz`, `manifest.json`) |
| Expanded DESI identities | `/Users/vasilybelokurov/data/qso_p_color/catalogues/desi_dr1_qso_all_redshifts.npz` |
| Reusable DESI Legacy photometry | `/Users/vasilybelokurov/data/qso_p_color/catalogues/desi_dr1_qso.npz` and `.json` |
| Reusable SDSS-QSO Legacy photometry | `/Users/vasilybelokurov/data/qso_p_color/catalogues/dr16q_dr9.npz` and `.json` |
| Full QSO acquisition root, called **QROOT** below | `models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/` |
| Frozen 1,657,444 eligible QSO targets | QROOT `targets.npz`, `selection.json`, `provenance.json` |
| Complete SDSS raw batches | QROOT `acquired/sdss_*.npz`; positional sub-results in `sdss_without_ids/`; identifier provenance in `id_links/` |
| Complete Legacy measurements | QROOT `legacy_local/4d418c295df8aa17/photometry.npz` and `report.json`; raw batches in `acquired/decals_*.npz` |
| Independent QSO association audit | QROOT `association_audit/64b33f5755fae833/` (current); earlier count caches in `16f14c32960e2fb4/` remain reusable |
| Fresh stellar preparation | `models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0/` (region products, `queries/`, `coverage_report.json`) |
| This inventory, called **INV** below | QROOT `local_inventory/2026-09-28/` |

## Five remaining surveys

The search inspected 5,511 distinct NPZ headers under `~/data` and repository
models/data, identified 1,679 native measurement caches, and inspected 59
external FITS/Parquet catalogue schemas. This is a record of those locations,
not a claim to have searched every directory on the computer. The native
caches contain all measurement, error and quality columns required by the
current adapters.

| Survey | Reusable target results | Of these: no counterpart | At least one usable band | Targets still unresolved |
|---|---:|---:|---:|---:|
| AllWISE | 68,398 | 28,401 | 39,994 | 1,589,046 |
| Pan-STARRS | 68,399 | 1,285 | 67,114 | 1,589,045 |
| NSC | 68,399 | 1,236 | 58,700 | 1,589,045 |
| SkyMapper | 68,399 | 60,534 | 7,714 | 1,589,045 |
| VHS | 68,399 | 60,287 | 6,991 | 1,589,045 |

A reusable result includes a recorded non-detection or rejected measurement;
neither should trigger another photometry download. An unresolved target has
no verified result in these caches. It may lie outside a survey footprint;
these counts do not predict detections or mandate blind all-sky queries.
Multiple/shared-association auditing remains separate from photometry reuse.

The main QSO caches are under
`/Users/vasilybelokurov/data/qso_p_color/project_data/multisurvey/queries/`:

| Survey | Native QSO cache | Required quality fields |
|---|---|---|
| AllWISE | `allwise_1edf9211d64b0827.npz` | `cc_flags` |
| Pan-STARRS DR1 stack | `ps1_354cb44e0956878f.npz` | `gnframes` through `ynframes`; primary-detection selection in saved SQL |
| NSC DR2 | `nsc_04058f4e0bbf0c22.npz` | `flags` |
| SkyMapper DR4 | `skymapper_d6e74ae79002cb7c.npz` | per-band `ngood`, `flags` |
| VHS DR5 | `vhs_e2660852e27e529f.npz` | per-band `pperrbits` |

The adjacent `.json` files store SQL. The input coordinates are in the parent
`targets.npz` (82,489 historical targets). Equivalent copies are in
`project_data/multisurvey_lsw/`; byte-identical copies are counted once.
`cone_<survey>_*.npz` in those query directories retain complete local cones,
including quasars subsequently removed from background samples. Only targets
whose entire matching aperture lies inside a cone qualify for complete reuse.

Stellar query caches contain additional compatible measurements, but were
centred on stars. They cannot establish a QSO nonmatch or nearest counterpart.
Among unresolved QSOs, they and other nearby cached rows offer measurement
candidates for 183 AllWISE, 10 PS1, 14 NSC, 4 SkyMapper and 2 VHS targets.
These remain flagged for association verification, not silently certified.

## Other useful catalogues and traps

- `/Users/vasilybelokurov/data/qso/ps1/qso_ps1_object_index.parquet`:
  174,842 rows. **Correction from live verification:** 166,447 derived IDs were
  rounded through float64 despite the integer column type. Recover IDs from
  `casjobs_exports/qso_ps1_full_c*_xmatch.fits` (original FITS `K` columns),
  joining by SDSS name and counterpart coordinates. These recover links to
  155,286 eligible targets, all within 1 arcsec, with no conflicting PS1 IDs.
  147,823 of the 155,286 eligible links needed numerical correction; all retain
  the same counterpart positions. 150,273 targets initially lacked reusable
  stack photometry. Retain the corrected IDs for
  possible indexed lookups; indexed retrieval. WSDB has the matching bigint `objid` index; retrieval
  verifies primary detection and association radius. See QROOT
  `local_inventory/2026-09-28/ps1_id_recovery.json` for all original file hashes.
- `/Users/vasilybelokurov/data/qso/ps1/qso_ps1_forced_full.parquet`:
  14,439,972 forced time-series measurements. These do not supply the required
  DR1 stack PSF magnitudes, errors and frame counts. Original exports remain
  under the same root's `casjobs_exports/`.
- `/Users/vasilybelokurov/data/qso/wise/snapshots/qso_wise_full_events.parquet`:
  45,084,818 W1/W2 time-series rows. It lacks the required AllWISE W1–W4 fluxes
  and flux errors. Its `allwise_cntr` values are missing or rounded scientific-
  notation strings (e.g. `4.541378013510278e+17`); none qualifies as an exact ID.
  Do not recover guessed integers by casting through floating point. Companion
  `qso_wise_full_catalog.parquet` and `qso_wise_full_objects.parquet` contain
  219,830 object summaries, not replacement catalogue photometry. Provenance is
  in `qso_wise_full_meta.json` and `../irsa_exports/full_run_manifest.json`.
- `/Users/vasilybelokurov/data/desi/agnqso_desi_photometric_xmatch_allsky_style_wsdb.parquet`:
  17,995,599 rows, but its columns cover Legacy, Gaia, SDSS and GALEX. It supplies
  none of the five survey measurement sets inventoried here.
- `/Users/vasilybelokurov/data/gaia/sdssdr16q_gaia_source.fits` and
  `/Users/vasilybelokurov/data/qso/qso_lightcurve_prepare/data/` contain embedded
  SDSS/forced-WISE measurements. W1/W2 columns alone cannot replace AllWISE's
  separate W1–W4 catalogue system and quality fields.

## Machine-readable references and reproduction

- `docs/LOCAL_PHOTOMETRY_INVENTORY_2026-09-28.json`: counts, source/output hashes,
  verification results and identifier coverage.
- `docs/LOCAL_PHOTOMETRY_FILES_2026-09-28.json`: every native cache's resolved
  path, SHA256, duplicate status and stable `file_index`.
- `docs/LOCAL_EXTERNAL_CATALOGUES_2026-09-28.json`: external catalogue locations,
  sizes, row counts and actual column names.
- INV `native_files.json`: detailed native-file manifest including SQL.
- INV `<survey>_target_inventory.npz`: exact full target identities and order,
  reusable source file/row, observed-band masks, nearby measurement references,
  and `unresolved_target_index`. A reusable source-file index of -1 means
  unresolved. A reusable row of -1 with a valid file index means a complete cone
  established no counterpart. Direct-query nonmatches retain their actual row.
- INV `ps1_identifier_candidates.npz`, `allwise_identifier_candidates.npz` and
  `identifier_summary.json`: separate candidate-ID inventory; the AllWISE
  candidate list is empty because its stored identifiers are inexact.

Run `scripts/inventory_qso_photometry.py --cache <QROOT> --inventory <INV>`
for photometry and `scripts/inventory_qso_identifier_links.py --cache <QROOT>
--out <INV>` for external identifiers. Both read local files only. A new empty
inventory directory triggers NPZ-header discovery under `~/data` and `models/`;
an existing directory reuses its saved `discovered_native.json`.

The completed AllWISE/PS1 acquisition used QROOT `gap_acquisition/`, with cached
results and corrected PS1 IDs. Acquisition has stopped. See `docs/QSO_GAP_ACQUISITION_2026-09-28.md`
for the command, checkpoint locations and measured query behaviour.
This inventory does not authorize retraining or certify the full 41-band sample.

## Prepared full-sample density inputs (29 September)

`models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8/` contains
`qso/` and `stars/` memory-mapped NPY arrays, `report.json` and the checksummed
`manifest.json`. The parent `current.json` points to this version. All eligible
objects remain present; masks and frozen roles determine which measurements
and rows enter each stage. See `FULL_TRAINING_READINESS_2026-09-29.md`.

## Authorized density-training run (29 September)

`models/multisurvey_psf/work/full_training_runs/20260929T045901Z/` contains
`launch.json`, `train.log` and `preflight.json`. The parent `current.json`
records the run path and initial worker PID. Model-selection records and
per-iteration checkpoints are under
`models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d/`. This run leaves the active
model pointer unchanged.

Live monitoring files in the same run directory: `PROGRESS.md`, `progress.json`,
`progress_history.jsonl`, `monitor.log` and `monitor.lock`. Updated every minute
by `scripts/monitor_full_training.py --watch`; checkpoint/fit files remain
the source of truth. No training process is restarted by the monitor.

Parallel continuation uses `parallel_train.log` and the updated `launch.json`
in the same run directory. `serial_launch.json` preserves the earlier process
record. In the unchanged fit directory, `parallel_execution.json` records the
coordinator and queue; `parallel_worker_<pid>.json` records slice assignments.
