# Full-sample density training readiness

The two requested preparation steps are complete: the available measurements
have been assembled locally, and an uncapped trainer reads those inputs and
the frozen spatial roles. No production fit, download or model promotion was
launched. The current model remains unchanged.

## Inputs and splits

Prepared arrays live in
`models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8/`.
`manifest.json` records the input configuration, provenance and output hashes;
`report.json` records band counts, masks, negative measurements and association
diagnostics. `current.json` in the parent directory points to this version.

| Population | Fit | Model selection | Catch-all calibration | Final test | Final shape fit: fit + selection |
|---|---:|---:|---:|---:|---:|
| QSO | 854,202 | 195,058 | 239,371 | 368,813 | 1,049,260 |
| Stellar PSF field | 1,616,421 | 359,473 | 502,251 | 500,712 | 1,975,894 |

All 1,657,444 eligible quasars and 2,978,857 cleaned field entries retain at
least one usable band. There is no magnitude, signal-to-noise, complete-band
or random-count cut added by this assembly. All 41 labelled bands remain in
the data layout; unavailable measurements are masked. No rows are removed to
meet a computing budget. Quasars may appear in two overlapping redshift slices;
this does not increase the unique population count above.

SDSS, Legacy, AllWISE and PS1 have recorded query outcomes for the whole QSO
target list. Each paused survey (NSC, SkyMapper and VHS) has 116,404 known
outcomes after combining the initial acquired prefix and reusable results.
Its remaining outcomes are unknown, rather than claimed non-detections.

Known multiple or shared counterparts mask only that survey's measurements.
Stored SDSS counts were reused without querying; new AllWISE and PS1 positional
counts came with acquisition. Unknown counts are retained explicitly, including
ID-based associations. The invalid Legacy cross-release counts are ignored.
Two QSO target rows share one Legacy counterpart: indices 820451 and 820452,
`sdss:022613.77+035835.8` and `sdss:022613.78+035837.2`. They are distinct master
objects about 1.4 arcsec apart; both keep their other measurements while their
shared Legacy measurement is masked. No parent catalogue or redshift was edited.

Sparse bands remain a scientific limitation. The least-populated QSO band has
171 fit and 19 selection measurements. This preparation establishes correct
missing-band handling and accounting, not validated support for every possible
band combination. Calibration and test roles remain reserved for assessment.

## Trainer and verification

Use `scripts/train_full_sample.py` with `configs/full_sample_training.json`.
The earlier recovery scripts and their capped configurations are historical;
do not use them to launch this full-sample run.

The new trainer uses all fit-role rows for component selection, evaluates on
the selection role, and refits on their union. Neither calibration nor test rows
enter shape or spatial-weight fits. Both component grids are configurable;
the current grid is 4, 8, 12 and 20. A winner at the top of a grid and a fit at
its iteration limit are recorded for review, not labelled validated.

Every EM iteration holds the mixture fixed while accumulating sufficient
statistics over every batch, then performs one M step. This reproduces the
existing full-data objective. The expected row count is checked on every pass.
Checkpoints include iteration history and the updated mixture; restarting
continues the same iteration sequence. Completed selection fits and final
parts are reused only under the same input/configuration/implementation identity.

Regression tests compare batched and existing EM with missing dimensions,
correlated errors, unequal weights and large coordinate offsets. They also
check interrupted-fit continuation, local-cache-only acquisition, incomplete
survey reuse, calibration/test exclusion, spatial-weight equivalence and
end-to-end candidate save/load without changing the active pointer.

The full test suite passes: **329 tests** (60.24 seconds). The method PDF also
builds successfully. The real-data preflight reads every role, verifies the row counts, checks all
43 redshift slices and hashes the prepared arrays. It includes timing probes
that calculate E-step statistics without updating any production mixture.
The numerical report is [FULL_TRAINING_PREFLIGHT_2026-09-29.json](FULL_TRAINING_PREFLIGHT_2026-09-29.json).

## Memory and time

Input fluxes and variances are memory-mapped. The 41 by 41 noise covariance is
created only for a 256-row batch. This avoids a 62.35 GB full-population
covariance allocation. Prepared arrays occupy 3.71 GB on disk. The conservative
batch-workspace estimate at K=20 is 0.55 GB; the measured real-data preflight
stays within the configured 4 GiB RSS budget (exact measurement in the report).
Spatial fitting stores an N by K component-likelihood array, rather than an
N by 41 by 41 covariance array. Fits run sequentially to bound memory use.

Short, separated timing probes imply **several days of EM work** if every
selection fit takes all 40 iterations and every final fit takes all 300 with
K=20. These are workload estimates, not a completion-time promise: component
choice, convergence, band patterns and machine load matter. The report records
the measured per-batch times and extrapolation assumptions. No row caps are
used to shorten that work.

## Commands and stopping point

Read-only preflight (no production fit):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/train_full_sample.py --benchmark
```

After the readiness review, the explicit fitting command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/train_full_sample.py --fit
```

This writes a separately versioned density candidate under
`models/multisurvey_psf/work/full_training_fits/<identity>/`, including the
quasar shapes, stellar shapes and HEALPix weights. It never promotes a model.
Population priors and the calibration-role catch-all must subsequently be
fitted consistently with these new shapes, followed by reserved validation.
The old population-completion script still describes the earlier training
layout and must not be used unchanged on these new inputs. Effective-area
assumptions for external-survey count priors remain to be assessed at that
stage; they are not an additional download requirement or a prerequisite for
fitting the colour densities. Full probability calibration remains a separate
scientific validation task.
