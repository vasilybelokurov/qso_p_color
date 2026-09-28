# Resumable SDSS acquisition blocks

New SDSS queries use at most **10,000 targets per block**, processed sequentially.
This is an operational checkpoint limit: every eligible target remains in the
sample. The scientific selection and association radius are unchanged.

`configs/photometry_sky_blocks.json` specifies the target limit, parent HEALPix
resolution (`nside=1`) and ordering resolution (`nside=1024`), both nested and
in equatorial coordinates. Targets are sorted by the fine pixel and grouped
within a parent cell, with smaller blocks at cell boundaries. Blocks therefore
have variable sky area. The parent pixels are acquisition geometry, unrelated
to the model's Galactic spatial dependence or its training/validation split.

Only the target list is partitioned. The query searches the full indexed SDSS
table, so a counterpart across a block boundary is not lost. Exact photometric
IDs remain the preferred route; positional fallback retains the same nearest
primary detection within the declared radius. Unmatched rows, negative fluxes,
photometric errors and quality fields are retained.

## Checkpoints and restart

`src/qso_pcolor/sky_acquisition.py` records the query identity, input coordinates
and IDs, block geometry, per-block target identities, counts, matched rows and
elapsed time. Each photometry file is written atomically. A failed query leaves
earlier blocks reusable; retry repeats only the unfinished block. Input row
identities are checked before saving and the final arrays are restored to their
original order. A lock prevents two workers using the same block cache.

The main acquisition and the independent no-ID script accept `--block-config`.
The latter publishes completed-target counts in `sdss_without_ids/status.json`
after each block. Per-block SQL plans and detailed checkpoints live under
`sdss_without_ids/blocks/`; ID blocks and failed-ID fallback blocks live under
`queries/`. Both scripts reuse completed whole-list results before issuing new
queries. The other six surveys retain their existing acquisition schedule.

Use one new SDSS query worker at a time. At the user's request at 17:58 CEST,
the unsaved whole-list positional download was stopped and its database backend
cancelled. Its partial in-memory results were unavailable as checkpoints, so
all 857,578 targets in that job are now queried through the block downloader.
The earlier 350,000 completed targets and the full ID result are reused.
The main worker waits for the combined block output, then queries failed ID
associations. No retraining is launched by these scripts.

## Sizing measurements

The earlier seven 50,000-target positional queries took 400.85–903.31 seconds
each; together, 350,000 targets took 4,696.32 seconds (74.53 targets/second).
The later 449,866-target ID query completed in 7,910.34 seconds
(56.87 targets/second). These are different samples and access paths; their
comparison does not isolate the effect of query size or concurrent load.

The reproducible sizing script is `scripts/benchmark_sdss_sky_blocks.py`.
Its current run uses 5,000, 10,000 and 20,000 targets, seed 20260928, and requires
at least five known SDSS QSO members per candidate nside=64 cell. It excludes
all production target cells and their neighbours, and positions within 0.6
degrees of stellar query cones. Each trial uses different, non-touching cells.
The targets lie outside the current acquisition footprint and are timing inputs
only; they never enter the training sample. A 25,000-target covered trial was
not available under these exclusions.

The first execution uses `EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)` after checking
that Q3C is indexed. This measures server execution and shared-buffer reads,
not network delivery of the photometry. A subsequent warm retrieval runs the
actual block downloader and checks that every target has exactly one output
row. Both timings are reported separately. Their sum, plus preparation, is a
conservative two-execution cost, not a measured single cold download time.
Shared-buffer reads do not prove physical disk reads: the operating-system
cache remains uncontrolled. The production positional query was still running
during these trials; backend activity is saved alongside each measurement.

Raw selections, SQL, plans, photometry and timing JSON files are retained under
`models/multisurvey_psf/work/sdss_block_benchmark/2026-09-28_covered/`.
The measurements were:

| Targets | First server execution | Subsequent warm download | Matched | Shared buffers read |
|---:|---:|---:|---:|---:|
| 5,000 | 77.23 s | 2.01 s | 4,696 | 10,732 |
| 10,000 | 158.02 s | 4.52 s | 8,904 | 26,500 |
| 20,000 | 295.03 s | 3.66 s | 17,838 | 48,697 |

Including preparation and both executions, the trials took 89.37, 169.04 and
317.15 seconds, respectively. First-execution throughput was 63–68 targets per
second, without a substantial size trend. These regional timings assess
checkpoint duration; they do not establish an optimal throughput or a causal
speedup from smaller queries.

The **10,000-target default** gives roughly three-minute checkpoints at these
rates; 20,000 targets gives roughly five minutes. This is a checkpoint-frequency
choice, not a scientific threshold or guaranteed runtime. Applied to the frozen
857,578-target no-ID list, it would produce 89 blocks (some smaller at parent
boundaries). The user subsequently requested this switch; the 89-block run began at
17:59 CEST, saving its first 50,000 rows in five blocks of about three seconds
each. Those initial warm timings are not a forecast for untouched regions.

The completed ID result contains 449,866 distinct target rows: 449,843 pass the
existing ID association check and 23 require positional fallback. The fallback
uses the new block path after the positional block download completes.

The full suite passes: 301 tests in 61.36 seconds. Tests cover interrupted runs,
completed-block reuse, reordered and unmatched outputs, negative fluxes, exact
64-bit IDs, duplicate/missing-row refusal, sky boundaries and cache identity.
The machine-readable timing report is
`SDSS_SKY_BLOCK_BENCHMARK_2026-09-28.json`.

A live correctness check retrieved the same 20,000 targets in one block and
in four 5,000-target blocks. Every returned column was exactly identical,
including errors, quality fields, negative measurements and unmatched rows.
The four smaller retrievals took 3.2, 2.7, 2.2 and 2.1 seconds with warm caches;
these are correctness timings, not fresh-sky performance measurements.
