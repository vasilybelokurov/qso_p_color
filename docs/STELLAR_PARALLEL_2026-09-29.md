# Four-worker stellar continuation

The user authorized closing the idle QSO workers and continuing the final
stellar fit with four workers. All 43 QSO fits are preserved byte-for-byte;
25 converged and 18 reached their iteration limit. Their validation status is
unchanged.

The serial stellar fit stopped immediately after saving iteration 18. Its
coordinator, four idle QSO workers, multiprocessing helper and monitor exited.
The replacement coordinator resumed that mixture and its complete likelihood
history. Its first full parallel iteration, 19, has been saved with exactly
1,975,894 fitting-plus-selection stars. No source-count cap or data acquisition
was introduced.

## Calculation and checks

Each worker evaluates the existing extreme-deconvolution E step on disjoint
chunks of 8,192 rows, retaining the existing 256-row transformation batches.
At most eight tasks are in flight. The coordinator adds sufficient statistics
in chunk order and makes one global M step after every row is accounted for.
Workers reopen read-only memory maps; cached arrays are not copied during
worker startup. Seeds, K=20, regularization, convergence tolerance and the
300-iteration total limit are unchanged.

On 131,072 identical diagnostic stars drawn from 16 distributed windows:

| Calculation | Seconds |
|---|---:|
| Serial E step | 26.84 |
| Four workers, including startup | 8.78 |
| Four workers, already started | 7.81 |

The measured steady speedup is **3.43x**. These diagnostic windows are a timing
experiment, not a training cap. Statistics agreed within rounding precision;
the largest absolute difference among the accumulated sums was 2.7e-9.
Repeated parallel passes were identical. Exact measurements and hashes are in
[the benchmark report](STELLAR_PARALLEL_BENCHMARK_2026-09-29.json).

The **332-test suite passes**. Tests cover missing bands, correlated errors,
unequal weights, large offsets, continuation from a serial checkpoint, worker
failure before a complete pass, preservation of completed QSO fits, and the
active-pointer invariant. The method PDF also builds successfully.

## Locations and restart

Current output:
`models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/`.
The streaming implementation has changed, so this is a new implementation
identity. `checkpoint_lineage.json` records the parent identity and each copied
file hash. The previous output, `full_training_fits/5d1429d22b40d24d/`, remains
an unchanged handover snapshot. Only this specifically verified serial engine
is admitted by the continuation compatibility check.

The run directory remains
`models/multisurvey_psf/work/full_training_runs/20260929T045901Z/`.
`launch.json`, `stellar_handover.json` and `stellar_parallel_train.log` identify
the current process and handover. The live report now shows the four stellar
workers and the count of stars accumulated during the current E step.

After confirming the current coordinator has exited, restart with:

```bash
python scripts/train_stellar_parallel.py --workers 4 --task-rows 8192 \
  --resume-from models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d
```

Existing destination checkpoints take precedence over the copied parent.
Both run locks prevent concurrent writers. Checkpoint history continues rather
than receiving a fresh iteration budget. The active model is not promoted;
convergence review, population completion and reserved validation still follow.
