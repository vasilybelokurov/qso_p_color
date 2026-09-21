# Maintenance and validation, 2026-09-21

The approved maintenance pass is complete. The current quasar model, background
model, background surface density and quasar prior are retained unchanged.
No training or background fitting was run in this maintenance pass. The later
[seven-survey extension](MULTISURVEY_VALIDATION.md) adds a separate model and
retains these four files. Clustering correction is outside this
work and is not an outstanding maintenance finding.

## Changes

- The scorer integrates the match window and total quasar intensity on shared
  nodes inside the model/prior support. Exact boundaries and interpolation knots
  are included. This fixes dependence on grid extent and probabilities above
  one when a coarse grid missed a narrow prior peak. Public redshift helpers
  also respect model support and resolve narrow windows.
- An empty quasar prior retains the colour likelihoods and Bayes factor. The
  posterior remains unavailable with an explicit status. Background magnitude
  extrapolation is flagged.
- Feature subsets now carry the corresponding per-object flags. Validation
  saves every `PairScore` field, including `dz_match_eff` and quality flags.
- Validation applies the survey mask and the declared reference-band fracflux
  limit, recomputes labels for the requested velocity window, and records cut
  counts, the random seed and hashes of the saved models. Bayes-factor metrics
  do not require an available posterior.
- Future redshift extensions inherit the saved model's maskbits policy. This
  fixes the extension script; no extension was run.
- README and the method note describe the corrected integration, the validation
  selection, the just-outside-window comparison, and the effect of a uniform
  prior rescaling. Such a rescaling cancels within the quasar-only probability,
  but can change ranking through competition with the background.

## Validation

```bash
python scripts/validate_pairs.py --max-fracflux 0.2 --hard-negative-max-kms 6000 10000
```

Cuts: release 9010, separation 3–30 arcsec, primary inside model support,
`maskbits = 0`, finite `fracflux_r <= 0.2`, at least three usable feature
dimensions, and 17 <= r < 22.5. The non-quasar class is subsampled to 30,000
with seed zero after the mask/fracflux cuts and before the photometry cut.
The match window remains ±3,000 km/s. All models are loaded from their saved
files; `--prior` replaces the old validator's `--plateau` reconstruction.

| Selected sample | Same-redshift quasars | Field quasars | Non-quasars |
|---|---:|---:|---:|
| All, 50,746 | 3,308 | 26,154 | 21,284 |
| Held out, 14,221 | 880 | 7,293 | 6,048 |

| Comparison on held-out objects | AUC |
|---|---:|
| Same-redshift vs all field quasars, log R | 0.815 |
| Same comparison, log Bayes factor | 0.759 |
| Same comparison, p_zmatch | 0.855 |
| Quasar vs non-quasar, log Bayes factor | 0.970 |
| Same-redshift vs 3,000–6,000 km/s field quasars, log R | 0.488 |
| Same restricted comparison, p_zmatch | 0.508 |

The previous held-out log R AUC was 0.812. The new sample has consistent cuts;
this small change is not evidence of a modelling improvement. The comparison
just outside the window contains 151 held-out negatives and 880 positives. Its
near-chance performance limits claims about velocity resolution, while leaving
the broad ranking result intact. No retraining is indicated by this pass.

## Verification and outputs

- Baseline: 117 tests passed. Updated suite: 127 tests passed.
- Numerical regressions use analytic uniform and triangular-prior integrals,
  plus checks of support, narrow windows, empty priors, labels and flag alignment.
- All 50,746 saved rows were checked for the declared cuts, IDs, flag alignment,
  full score fields, probabilities in [0, 1], and both posterior identities.
- SHA-256 hashes of all four deployed model/prior files match the pre-edit hashes.
- The validation PNG and rebuilt method PDF were visually checked.

The machine-readable outputs are `data/pair_validation_results.npz` and
`data/pair_validation_report.json`; the figure is
`plots/validation/pair_validation.png`. Earlier NPZ/JSON results are preserved
locally with `_before_20260921` suffixes. The method note is
`docs/method/method.pdf`.
