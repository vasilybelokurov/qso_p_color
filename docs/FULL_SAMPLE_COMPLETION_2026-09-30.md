# Full-data candidate completion: items 1–3

30 September 2026. The assembled full-data model is retained, its prior/catch-all
bundle is complete, and the bright-star optical weakness has been investigated.
The active model has not been changed; complete-score release checks are next.

## Saved candidate

`models/multisurvey_psf/work/full_sample_completion/e894d3c77baab446/`

Load it with `PSFMultiSurveyBaseline.load(...)`, or use the explicit candidate
pointer `models/multisurvey_psf/work/full_sample_completion/candidate.json`. This is separate from the active
`models/multisurvey_psf/current` pointer. No downloads or density refits occurred.

- The density model and matching spatial weights are byte-identical to the
  assembled full-data candidate. Individual continuation fits were not mixed in.
- Stellar surface-density priors use all **1,975,894** fit/selection sources from
  **180** regions. Calibration and test rows enter none of these new counts.
- All **41** reference bands have prior pairs. QSO abundance grids and the
  inherited completeness constant are numerically unchanged, with explicit
  source provenance. Larger spectroscopic training counts are not a new
  sky-abundance calibration. The inherited priors retain their original
  selection/footprint limitations and are not a fresh frozen-role abundance fit.
- Six Legacy optical bands use measured per-band usable areas. The remaining
  **35** use the explicitly marked historical approximation: base Legacy usable
  cone area where the survey has usable measurements. This includes Legacy
  W1/W2, whose separate coverage was not measured. External-band area precision
  and probability calibration remain unfinished; the bundle does not conceal
  these assumptions or require more downloads.
- Catch-all shares use **all 502,251 calibration sources**. The broad Student-t
  is rebuilt from the new stellar-mixture moments while keeping nu=4, kappa=1,
  exact noise convolution, three share bins and the inherited positive pooling.
  The pooled share is **0.000436441**; all stored shares are
  strictly positive (range 5.09448e-05–0.000918971).
  The three bins describe catch-all shares; stellar colours still depend
  continuously on reference magnitude.

## Bright-star diagnosis

A predeclared random sample of 30,000 calibration-role sources contains 1,648
bright objects, defined once by combined-optical reference luptitude below 20.
The cohort is held fixed when removing surveys. The density loss reproduces:
combined optical -0.233 nats/object, SDSS-only -0.325, PS1-only -0.270,
Legacy+PS1 -0.194. Legacy-only and Legacy+SDSS are approximately unchanged.
All-available-band density improves by +0.454 on this calibration cohort.
The loss is therefore concentrated in the SDSS/PS1 optical predictions, rather
than a general failure of the Legacy stellar locus. This localizes the issue;
it does not establish a unique causal explanation for the fitted-shape change.

For the same objects, compare the maximum QSO log density at five fixed redshifts
(0.5, 1.0, 1.8, 2.5, 3.5) against the stellar plus catch-all density. The declared
high-evidence threshold is log ratio > 5. Apply the existing four-sigma hard
support guard. The following counts are evidence diagnostics, not probabilities
or labelled false-positive rates:

| Measurements | Objects | Small model high evidence | Full model high evidence |
|---|---:|---:|---:|
| Combined optical | 1648 | 10 | 4 |
| Legacy optical | 1648 | 2 | 4 |
| SDSS only | 749 | 5 | 8 |
| PS1 only | 1186 | 6 | 5 |
| Legacy + SDSS | 1648 | 16 | 5 |
| Legacy + PS1 | 1648 | 4 | 5 |
| All available | 1648 | 7 | 7 |

Combined optical falls from 10 to 4 high-evidence rows; all four were already
high in the small model. All-band counts remain 7, with no newly high rows.
SDSS-only rises from 5 to 8, including five newly high objects; their exact input
row identities are saved for the subsequent complete-score check. Legacy-only
and Legacy+PS1 also have small increases, recorded rather than hidden.
The sample is star-dominated, not spectroscopically certified stars, so a high
score does not prove a misclassification. Calibration rows also enter the
catch-all share fit: this is a focused diagnosis, not an independent release
validation. No parameter was tuned against these results.

**Disposition:** retain the full-data density model. The combined-optical density
loss did not translate into an increased high-QSO-evidence tail in this check.
There is no demonstrated basis for another global density refit. Carry the
specific SDSS-only and other newly high rows into item 4, using the complete
priors and scoring interface before activation.

## Verification and reproducibility

The bundle loads with its hash-checked manifest; all 41 QSO abundance grids
match their source numerically. All 41 singleton inputs score with finite
posteriors, zero colour evidence and the no-colour-information flag. This is
functional verification, not probability calibration. Unit tests cover role
separation, conservation of counts under empty-bin merging, area normalization,
rejection of counts without area and exact QSO-abundance reuse.

Reproduce with `scripts/complete_full_sample.py`, then
`scripts/diagnose_bright_stars.py`. The configuration, per-checkpoint row lists,
densities, hashes, count/area reports and diagnostic predictions are saved
beside the candidate. The completion takes about 46 seconds on this machine
with prepared local inputs. Final numerical evidence is in
`docs/FULL_SAMPLE_COMPLETION_2026-09-30.json`.
