# Unified full-data refit: launch readiness, 30 September 2026

**Ready for the requested 4+4 launch after the launcher update. No production
fits have started.** The audit found operational loose ends in the pilot
launcher; it found no new reason to reconsider unification or acquire data.

## Loose ends resolved

1. **Pilot scope was still active.** Added `configs/unified_full_training.json`
   and `scripts/train_unified_full.py`. The full launcher requires all sky
   cells and the frozen fit/select roles; it refuses the pilot footprint.
   Default execution only prepares/verifies. Production fitting requires
   explicit `--fit`. The full preparation has completed.
2. **The stellar fit was serial.** Four workers now accumulate disjoint
   stellar batches with the same current mixture and observation operator;
   one global update uses all their sufficient statistics. Four independent
   QSO slice workers run concurrently. Each worker has one numerical thread.
   A coordinator thread manages the stellar pool; it is not another stellar
   fit. Fixed-order reduction and bounded queued tasks control reproducibility
   and memory. This retains one shared stellar/background model.
3. **Saved checkpoints were not being restored.** Each iteration now records
   the updated model, complete likelihood history, iteration and input/row
   identity. Restarts verify those identities and continue from the saved
   iteration. Completed fits are reused. A failed worker cannot commit a
   partial global update. A small likelihood decline is no longer labelled
   convergence; the final report also records declines and iteration limits.
4. **Bundle completion assumed the pilot directory.** Completion now accepts
   an explicit run/config and full-footprint selection. Spatial weights,
   counts and catch-all remain subsequent stages, using their existing roles.

## Inputs and fitting settings checked

- All **39 frozen input files** match their recorded SHA-256 hashes.
- **1,049,260 QSO** and **1,975,894 stellar/background** fit/select objects.
  No eligible fitting object has zero usable bands. No S/N training cut or
  row cap is added; 211,188 QSO and 1,241,696 background training objects
  retain at least one negative measured flux.
- All training QSOs enter the existing **43 slices**; slice populations range
  from 640 to 119,828. Overlapping slices deliberately reuse rows.
- Calibration/test remain separate: QSO 239,371/368,813; background
  502,251/500,712. Previously inspected test data remain development/regression
  checks, not untouched confirmation.
- All **41 input bands** have full-sample measurements. The 36 latent
  coordinates, fixed North/South relation, component counts and canonical
  full-data warm starts are retained from the tested design.
- Maximum **300 iterations**, relative tolerance **1e-5**, regularization
  **0.001 mag²**; these follow the previous full-data fitting settings rather
  than the pilot's eight iterations. Early convergence can stop a fit. A
  budget limit is reported, not treated as convergence. No component search.
- Four QSO workers plus four stellar workers; batch size 128 and stellar
  task size 8,192 rows. Every selected row enters each iteration. Progress
  reports row × component × iteration budget fraction and live stellar
  row progress; this is an approximation to compute, not a wall-time promise.
- Disk audit found approximately 1.2 TiB free. All inputs are local. There
  are no downloads or changes to the active model pointer.

## Checks performed

The projected numerical tests compare four workers with serial fitting,
including missing bands, correlated errors, different observation operators
and interrupted continuation. Model parameters and likelihoods agree to
1e-10. A deliberate worker failure leaves no partial checkpoint. Disk-level
restart tests reproduce uninterrupted results and reject changed identities.
The 4+4 launcher test completes both populations, preserves the active pointer
and reuses completed results on a second call.

Targeted tests: 14 projected/unified tests and 3 launcher integration tests
passed. The full suite reports **366 passed in 78.45 seconds**, recorded in
`/tmp/unified_full_launcher_suite.log`. Full-data preflight passed with
**44 prepared tasks and zero fit-result files**.

## Deferred items that do not block fitting

- Rejection-cut calibration and full probability calibration remain separate.
  Rejection rules will be tuned on calibration roles after the densities are
  fitted; the active model is replaced only after focused candidate checks.
- Rare-survey support remains uneven: SkyMapper u/v have only 190/281 QSO
  training measurements, with zero in 18/9 slices respectively. Keep sparse
  support flags and existing measurements. This does not justify more data
  acquisition or delaying the main refit; it limits claims for those subsets.
- External-survey usable areas and abundance priors retain their previously
  documented approximations. They affect posterior calibration, not the
  ability to pool the photometric density fits. Preserve flags and provenance.
- Fixed regularization can cause likelihood declines; monitor the recorded
  curves and returned-model likelihoods. The successful short pilot does not
  guarantee that every long fit will converge.

## Reproduction and next action

Prepared candidate directory: `models/multisurvey_psf/work/unified_full/20260930/5fd837e5cafee6c1`.
Read `preflight.json`, `config.json`, `identity.json` and `tasks.json` there.
Input audit: `docs/UNIFIED_PRELAUNCH_INPUT_AUDIT_2026-09-30.json`.
Preparation log: `/tmp/unified_full_preflight.log`.

Prepare/verify only:

```bash
python scripts/train_unified_full.py
```

The proposed launch, **not executed**, is:

```bash
python scripts/train_unified_full.py --fit
```

After density completion, use the same candidate explicitly:

```bash
python scripts/complete_unified_pilot.py --config configs/unified_full_training.json
```

Then check fit health and held-out density/ranking, calibrate rejection on the
reserved calibration role, and verify the replacement before changing the
active pointer. The recommendation is to launch this prepared refit; no
additional exploratory fitting campaign is needed first.
