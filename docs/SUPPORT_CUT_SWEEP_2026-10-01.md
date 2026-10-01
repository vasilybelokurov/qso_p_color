# Extinction-corrected baseline: release checks and the support-cut threshold (1 October 2026)

## Conclusion

The extinction-corrected, magnitude-independent baseline
(`models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059/bundle`) ranks quasars
better than the active model in all eight band/hemisphere panels when no support cut is
applied. With the support cut at its original, untuned setting it fails four panels, by
at most 0.013 in AUC. **The failures come from the crude cut, not the model.** Loosening
the cut from 98% to 99.5% calibration retention passes all eight panels and leaves one
stray point on the artificial colour grids, against 28 with no cut.

Practical consequence: the baseline is usable, with a support cut. The cut is genuinely
needed for this model, more than for the earlier magnitude-dependent one. Its threshold
should be set at about 99.5% retention, once it has been confirmed on data that did not
choose it.

## What the support cut is

For each candidate, the cut asks how unusual its colours are for a quasar at the target
redshift. It does this by ranking the candidate against simulated quasars drawn from the
model, with the candidate's own errors. A threshold set on reserved calibration quasars
(role 2) rejects the rarest fraction. A rejected object gets no ranking score. This removes
false quasar scores in empty colour space, but it also removes rare real quasars (reddened,
broad absorption lines, variable, bad measurements). Those rejected real quasars are what
lowers AUC.

## Release checks at the original threshold (98% retention)

Clean role-3 test rows outside every recorded development exclusion; 500 per class per
hemisphere. AUC = quasar versus background ranking. "Fail" means more than 0.005 below
the active model.

| Bands | Active | Baseline, no cut | Baseline, cut at 98% | 30 Sep refit, cut at 98% |
|---|---:|---:|---:|---:|
| All, South | 0.945 | 0.965 | 0.953 | 0.971 |
| All, North | 0.941 | 0.977 | 0.976 | 0.977 |
| Legacy optical, South | 0.978 | 0.988 | 0.977 | 0.978 |
| Legacy optical, North | 0.986 | 0.990 | 0.977 fail | 0.977 fail |
| SDSS, South | 0.976 | 0.979 | 0.969 fail | 0.977 |
| SDSS, North | 0.967 | 0.979 | 0.971 | 0.971 |
| PS1, South | 0.941 | 0.945 | 0.928 fail | 0.929 fail |
| PS1, North | 0.952 | 0.958 | 0.946 fail | 0.940 fail |

Full report: `docs/UNIFIED_RELEASE_BASELINE_2026-10-01.json`.

## Threshold sweep (from cached scores; no rescoring)

Each object's support percentile and raw score were already saved, so any threshold can be
applied exactly. Reproduces the release check at 98% (all bands, South: 0.952956).

**Column definitions:**
- **Target retention:** fraction of role-2 calibration quasars kept.
- **Worst ΔAUC:** smallest AUC minus active AUC over the eight panels.
- **Failed:** number of panels below −0.005.
- **Minimum retention:** the lowest fraction of test quasars kept, over the eight panels.
- **Grid stray points:** high-QSO scores (p > 0.5) in the original low-density mask of the artificial north/south colour grids at r = 18.5 and 21.

Baseline (extinction-corrected, magnitude-independent):

| Target retention | Threshold | Worst ΔAUC | Failed | Minimum retention | Grid stray S18.5 / S21 / N18.5 / N21 |
|---:|---:|---:|---:|---:|---|
| 0.98 (current) | 0.0234 | −0.0127 | 4 | 0.960 | 0 / 0 / 0 / 0 |
| 0.99 | 0.0156 | −0.0075 | 1 | 0.972 | 0 / 0 / 0 / 0 |
| **0.995** | 0.0078 | −0.0039 | **0** | 0.986 | 1 / 0 / 0 / 0 |
| no cut | 0 | +0.0025 | 0 | 1.000 | 8 / 13 / 3 / 4 |

30 September refit (uncorrected, magnitude-dependent), for comparison:

| Target retention | Threshold | Worst ΔAUC | Failed | Minimum retention | Grid stray |
|---:|---:|---:|---:|---:|---|
| 0.98 | 0.0234 | −0.0123 | 3 | 0.964 | 0 / 0 / 0 / 0 |
| 0.99 | 0.0156 | −0.0106 | 1 | 0.968 | 0 / 0 / 0 / 0 |
| 0.995 | 0.0078 | +0.0009 | 0 | 0.988 | 0 / 0 / 0 / 0 |
| no cut | 0 | +0.0048 | 0 | 1.000 | 1 / 3 / 1 / 0 |

Numbers: `docs/SUPPORT_CUT_SWEEP_2026-10-01.json`.

## Caveats

- **Data that chose the threshold:** the sweep used the same test rows that report the
  result. A 99.5% threshold must be confirmed on rows that did not choose it before it is
  quoted as performance.
- **Granularity:** with 255 simulated draws, thresholds come in steps of 1/256 = 0.0039.
  0.0078 is the second step, and 99.9% retention is already the same as no cut. More draws
  would give finer control.
- **What AUC measures:** AUC compares spectroscopically selected quasars with random point
  sources. It is not a calibrated probability or a labelled contamination rate.
- **Grid stray points:** these come from artificial colour grids. They are stress tests, not
  measured contamination.

## Open question, and next step

The baseline is weaker than the magnitude-dependent model in two ways. Its held-out
density is about 0.6 nats per object lower. Without the cut, it has more stray grid points
(28 against 5). Three effects could contribute: the magnitude-independence rule, the
extinction correction, or the cut. The cut is now addressed above. The prepared
unconstrained refit (`5f4002492dbb849c`: extinction-corrected, magnitude-dependent) separates
the other two. It has not been launched; it awaits the PI's decision.
