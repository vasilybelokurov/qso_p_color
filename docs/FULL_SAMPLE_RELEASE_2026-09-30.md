# Complete-score release audit: activation held

30 September 2026. The full-data candidate improves real-object ranking and
passes numerical checks, but fails the northern low-density colour-grid check.
It has **not** replaced the active model. This is a measured release failure,
not a requirement that every EM fit converge or every metric improve.

## Real-object complete scores

The paired sample contains 256 quasars and 256 stellar-background sources,
with 128 per population in each declination stratum. Objects use the frozen
test role; quasars also lie in the original reserved sky cells. Band selections
use the same available measurements in both models. Fixed clean-blend fixtures
isolate the photometric calculation; this is not close-pair validation or a
claim that all sampled objects have independently certified blend measurements.
Stellar primary-redshift values match the QSO redshift distribution. AUC measures
ranking separation for target-redshift quasars against this star-dominated
sample; rejected objects remain at the bottom of the ranking. Neither this
sample nor AUC establishes probability calibration.

| Bands | Small-model AUC | Full-model AUC |
|---|---:|---:|
| all available | 0.9429 | 0.9743 |
| optical | 0.9829 | 0.9962 |
| legacy optical | 0.9864 | 0.9859 |
| legacy wise | 0.9993 | 0.9995 |
| sdss only | 0.9794 | 0.9884 |
| ps1 only | 0.9707 | 0.9673 |

No AUC decrease exceeded the predeclared 0.03 review threshold. All-band QSO
eligibility improves from 240 to 249 of 256; high total-QSO scores in the stellar
sample remain 1/256. Combined-optical high-QSO stellar counts fall from 2 to 0.
Wrong-primary-redshift checks retain the competing field-QSO hypothesis: the
fraction with higher true-redshift rank is 0.984 with all bands, versus 0.975
for the old model. All eligible outputs are finite and bounded, and the exact
window/ranking identity holds. Flagged outside-both objects are rejected as
specified. This checks implementation, not the adequacy of the flag itself.

The five newly high-evidence SDSS-only calibration cases were revisited through
the full scoring interface. Their new maximum total-QSO scores are approximately
0.006, 0.116, 0.756, 0.785 and 0.226. Only two exceed 0.5, and those two already
exceeded 0.5 with the old bundle. Their maximum same-redshift scores are below
0.045 for the declared velocity window. High colour evidence did not create five
new near-certain classifications. These remain uncalibrated model outputs and
unconfirmed background labels.

## The release blocker

On the northern Legacy g/r/z grid at reference luptitude 18.5, **72** common
low-density points receive total-QSO scores above 0.5 after the existing guard,
versus **0** for the old model. At reference luptitude 21, the corresponding
counts are **79 versus 12**. These are the points below 1% of both models' grid
peak QSO and stellar intensities. At the stricter 0.01% level, the counts are
32 versus 0 and 37 versus 2. All exceed the predeclared two-percentage-point
review threshold. Southern grids pass; the northern faint grid has no eligible
high-QSO points. The complete results, including rejections, are in the JSON.

An explicit failing example has northern native luptitudes (g,r,z) =
(21.5,18.5,22.5), i.e. colours (g-r,r-z) = (3,-4). The new model returns a
**0.9951 total-QSO score**. Its conditional nearest-component distances are only
0.256 sigma for QSOs and 0.803 sigma for the stellar component, so the existing
four-sigma guard accepts it. Yet among training objects with northern
18 <= r < 19 and all three measurements, the nearest QSO is 3.914 magnitudes
away in this two-colour plane (12,124 rows), and the nearest stellar-background
source is 4.668 magnitudes away (5,829 rows). Gaussian distance is not evidence
of empirical support in this example.

The grid location (Galactic l=180,b=45 degrees) is at equatorial declination
41.65 degrees; its HEALPix cell contains a northern training cone. This is not
simply a northern photometric system tested at a southern-only location.

## Isolated cause

Low-redshift mixture component 8 admits the unsupported northern point. Using
full observed-band likelihoods on the actual frozen fit+selection rows, summed
responsibilities show:

| Slice centre | Global component weight | Effective total members | Effective northern members | Bright northern members |
|---|---:|---:|---:|---:|
| 0.25 | 0.160 | 403.2 | 1.11 | 6.4e-05 |
| 0.45 | 0.100 | 1179.9 | 1.82 | 1.07e-05 |
| 0.65 | 0.023 | 725.2 | 2.67 | 1.03e-16 |

An effective member is the sum of fitted posterior component responsibilities,
not an integer count of uniquely assigned sources. The northern predictions of
these globally significant components have almost no direct northern support,
especially at the bright reference magnitude. Parameters for missing bands
still exist, but their Gaussian distance must not be taken as training support.
This supplies a concrete mechanism for the observed extrapolation failure.

Small diagnostic substitutions at the northern bright grid do not remove it:
using the old stellar prior leaves 70 of the 72 high-QSO tail points; dropping
spatial weights leaves 66; substituting the old catch-all leaves 49. These were
read-only calculations, not candidate changes. Arbitrarily increasing the
catch-all share or replacing the abundance prior is not an established repair.
A joint-distance check retaining reference magnitude also leaves all 72; merely
changing the distance formula is not sufficient evidence of a fix.

## Disposition and bounded next repair

Keep the completed full-data candidate, the measured real-data gains and all
training artifacts. Keep the active small-data bundle only as the temporary
fallback. Do not restart global fitting or raise release thresholds.

The next repair should make support depend on actual component-level observed
training information for the requested bands. First compute those support
counts from the frozen training rows, then prevent components whose relevant
coordinates are effectively imputed from certifying support. Assess the rule
on calibration rows before rerunning the same real-object and grid checks.
This should preserve the joint 41-band marginal likelihood and expose unreliable
scores, rather than silently disabling northern data or fitting separate band
subsets. If a shape correction is also necessary, target the implicated
components/slices; a whole-model refit is not presently justified.

Reproduce the audit with `scripts/validate_full_sample_release.py`. The focused
training checks are `scripts/check_northern_tail_support.py` and
`scripts/diagnose_northern_component_support.py`, controlled by
`configs/northern_tail_diagnostic.json`. Their numerical evidence is saved in
`docs/NORTHERN_TAIL_SUPPORT_2026-09-30.json` and
`docs/NORTHERN_COMPONENT_SUPPORT_2026-09-30.json`. No production model,
catch-all, prior, support threshold or active pointer was changed during this
release assessment. Full probability calibration remains unfinished.
