# Spatial background in the active PSF model

`XDQSOBaseline.load("models/legacy_psf_xdqso/current")` now loads a background
whose colour distribution and surface density depend on reference magnitude
and Galactic position. It combines immutable base bundle `8e2a27c013ed` with
`spatial_7bbdd562534bcc68.json`. The quasar model and its prior are unchanged.

## Model and connection to the legacy implementation

The existing `BackgroundColourModel` supplies the Galactic NESTED HEALPix
hierarchy, magnitude bins, cell/parent/global pooling and position-dependent
scoring. This implementation fits **component weights**, sharing the existing
Gaussian means and covariances over the sky. It is the requested simplification
of the older code, which could refit full mixtures within each cell.

The field intensity is

`Sigma_B(m,l,b) * [(1-eta(m)) * sum_k w_k(m,l,b) N_k(c; V_k+S) + eta(m) p_U(c)]`.

Colour estimates use `n/(n+n0)` pooling from cell to parent to hemisphere.
Surface densities use non-quasar counts divided by measured usable area, with
their own pooling strength. Empty observed cells contribute zero measured
counts; unsampled cells fall back. The broad component and eta are inherited;
preserving all Gaussian shapes preserves the original covariance envelope.

Both hemispheres selected `nside=4`, parent `nside=2`, colour pooling `n0=1000`
and density pooling `n0=100`. These choices came from withholding one cone in
each populated fitting stratum. The nside grid, parent ratio, pooling grids,
seed and convergence settings are in `configs/legacy_spatial.json`. Changing
nside requires a new fit; it cannot reinterpret the cells of an existing fit.

The saved model contains 184 southern and 68 northern cell/magnitude fits,
plus 100 southern and 32 northern parent/magnitude fits. All 384 converged.
Each scored row records its composite model ID, background mode, nside, match,
blend and support policies, and a policy hash. The scorer converts catalogue
RA/Dec to Galactic coordinates before evaluating the background.

## Geometry and field validation

The original cone design checked the centre's cell, allowing some cones to
straddle partition boundaries. This run excludes such cones in full, retaining
their area estimates only when the entire cone is accepted. An inclusive
HEALPix overlap query checks geometry; catalogue positions are independently
checked against the retained cell. Latitude-edge cones are also excluded.
274 of 304 cones remain. Sampling weights are recomputed within each fitting
split from the retained cone areas.

The southern test contains **97,041 field sources in 38 cones across 15 cells**.
Those cones and cells do not fit or select the new spatial parameters.

| Southern test measurement | Pooled background | Spatial background |
|---|---:|---:|
| Observed/predicted total counts | 1.433 | 1.253 |
| Mean absolute fractional count error per cone | 0.776 | 0.442 |
| Observable-population Poisson log score gain | reference | +0.190 nat/source |

The 95% sky-cell bootstrap interval for the score gain is **+0.039 to +0.300**.
The score includes colour density, source counts and the adopted contribution
of unrecognised quasars. Count errors improve but remain substantial.

Synthetic tests independently recover changes in colour-population proportions
and surface density. Tests of the active saved model hold photometry fixed
while changing position: background likelihoods, densities and ranking change,
while quasar colour likelihoods remain equal. They also check covariance and
missing-band handling, fallback, boundary exclusion, immutable component
shapes, artifact integrity, and scoring provenance.

Global component shapes and priors retain their original training provenance;
this is conditional validation of spatial adaptation, not a fresh independent
validation of the entire model. The north has selection/calibration cones but
no reserved field test cells and remains provisional. The existing quasar-prior,
tail and full probability-calibration limitations remain. In particular, the
unrecognised-quasar probabilities used for fitting are calculated against the
fixed pooled baseline; they are not re-estimated jointly with the spatial fit.

Software verification: the complete suite passes, **247 tests in 57.87 seconds**.
The method PDF was rebuilt and its updated opening pages visually checked.

## Reproduction and use

The active pointer includes both the base directory and spatial artifact. Loading
the immutable directory `models/legacy_psf_xdqso/8e2a27c013ed` explicitly recovers
the historical pooled model. An alternative spatial fit can be loaded with
`XDQSOBaseline.load(base_directory, background=path)`; mismatched base bundles,
selections, systems, feature layouts and Gaussian shapes are refused.

```bash
python scripts/fit_spatial_psf_background.py \
  --cache /tmp/qso-spatial-cones-contained \
  --output /tmp/qso-spatial-contained.json \
  --report /tmp/qso-spatial-contained-fit.json
python scripts/validate_spatial_psf_background.py \
  --adaptation /tmp/qso-spatial-contained.json \
  --fit-report /tmp/qso-spatial-contained-fit.json \
  --output /tmp/qso-spatial-contained-validation.json
```

The cached catalogue sample is required; no database queries are made.
`FIT_legacy_psf_spatial_2026-09-28.json` preserves the choices and per-cone
predictions. `VALIDATION_legacy_psf_spatial_2026-09-28.json` records the checks,
intervals and excluded-cone list. The scientific validation command exits
non-zero if a required spatial check fails.

The optional `fit_local_psf_background.py` refits the same weights and counts
inside a supplied cone, with source exclusions, measured area and scope checks.
It is separate from the survey-wide default; the active spatial artifact is
not a collection of per-candidate fits.
