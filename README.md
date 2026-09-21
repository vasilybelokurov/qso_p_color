# qso_pcolor

Is the photometric companion of a known quasar itself a quasar at the same
redshift?

Given a quasar with a spectroscopic redshift *z*₀ and a companion a few
arcseconds away with broadband photometry only, this package weighs three
competing explanations and reports how strongly the colours support the first:

| hypothesis | meaning |
|---|---|
| `same_z` | a quasar whose redshift matches *z*₀ |
| `field_q` | a quasar at some other redshift |
| `bkg` | anything else in the imaging at that brightness and sky position |

The third row is why the package has this shape. A quasar-versus-star
calculation cannot tell a genuine companion from a foreground quasar at
*z* = 2.6, and for a binary-quasar search that is the failure mode that matters.

---

## Choose a saved model

**Both model families ship with the repository and work offline.** The original
southern Legacy Surveys models remain at their existing paths, with unchanged
trained parameters and loading APIs. The seven-survey model is an additional
choice; installing the update does not switch an existing script to it.

| Model family | Photometry and outputs | Start here |
|---|---|---|
| Original southern Legacy DR9 | Dereddened `grz` and Legacy forced W1/W2; colour evidence, `p_zmatch_given_qso`, and `p_sameq` / `log_r_per_unit_z` using the supplied priors | [Original offline example](#scoring-a-candidate--offline-straight-from-a-clone) |
| Seven-survey extension | Any subset of SDSS, DECaLS/Legacy DR9, ALLWISE, PS1, NSC, SkyMapper, and VHS, including infrared-only; observed native photometry, at least two measured bands; evidence and `p_zmatch_given_qso`, with population posteriors requiring matched priors | [Multi-survey offline example](#combining-surveys-including-infrared-only-inputs) |

The saved files are:

| File | Contents |
|---|---|
| [qso_south_full.json](models/qso_south_full.json) | Original southern quasar colour model |
| [background_south_global.json](models/background_south_global.json) | Original southern background colour model |
| [background_density_south_global.json](models/background_density_south_global.json) | Original background surface density |
| [sigma_q_south.json](models/sigma_q_south.json) | Original quasar surface-density prior |
| [multisurvey.json](models/multisurvey.json) | New joint quasar/background model, transform, and band schema |

Clone the repository and run the examples from its root so these relative paths
resolve. Model files are included in Git; they are not bundled into the Python
wheel. The multi-survey model uses different flux and extinction conventions,
so follow the example for the model you load.

## State of play — read this before trusting a number

**What is solid.** The statistical machinery, checked against independent routes
(quadrature, Monte Carlo, closed forms) by 149 tests. The original quasar colour model,
trained on 1,106,986 spectroscopic quasars — 917,489 DESI DR1 and 189,497 SDSS
DR16Q, all with `maskbits = 0` — with 20 % of nside=4 sky blocks reserved before
fitting. The
prior-independent Bayes factor.

**What is measured** (`scripts/validate_pairs.py`, rerun 2026-09-21): 50,746
spectroscopically labelled companions at 3–30″ with `maskbits = 0`, finite
`fracflux_r ≤ 0.2`, and 17 ≤ r < 22.5. There are 14,221 companions in the
reserved sky blocks. Quasar-ranking results below use those held-out objects;
star/galaxy results use the full sample.

| question | result |
|---|---|
| same-*z* quasar vs wrong-*z* quasar, ranked by `log_r_per_unit_z` | **AUC 0.815** (0.855 by `p_zmatch_given_qso` alone) |
| same, ranked by the Bayes factor alone | AUC 0.759 |
| quasar vs spectroscopic star, by Bayes factor | **AUC 0.988**; 0.1 % of stars above the median same-*z* quasar |
| quasar vs spectroscopic galaxy, by Bayes factor | **AUC 0.964**; 0.3 % above |
| `p_zmatch_given_qso` calibration | right in shape, **low by 3.3× (20–30″) to 10.0× (3–5″)** |

That last row is not a bug: the scorer assumes the companion's redshift is drawn
from the field, and physical pairs cluster. The factor is the measured
clustering excess; multiply the odds by it if you want a probability at a given
separation. Rank on `log_r_per_unit_z`; do not read `p_sameq` as calibrated
without that factor.

The broad comparison is useful, but it does **not** resolve the velocity
boundary. Against held-out quasars just outside it (3,000–6,000 km/s), AUC is
0.488 by `log_r_per_unit_z` and 0.508 by `p_zmatch_given_qso`: effectively
chance. This limits what can be inferred from colours; it is not a reason to
retrain the current model.

Reproduce this validation using the saved models and cached pairs:

```bash
python scripts/validate_pairs.py --max-fracflux 0.2 --hard-negative-max-kms 6000 10000
```

The report records the selection counts and model hashes. The saved rows include
`dz_match_eff`, quality flags and all `PairScore` fields. `--prior` selects a
saved prior; the validator no longer rebuilds one through `--plateau`.

`log_bayes_factor_qz_bkg` is **not** an alternative ranking statistic. It
compares "a quasar at *z*₀" against "background" and has no `field_q` term.
Measured: it separates same-*z* from wrong-*z* quasars with AUC 0.759 against
0.815 for `log_r_per_unit_z` — worse, not useless, because p(c | Q, *z*₀) is
itself redshift-dependent. Use it to reject stars, not to order candidates.
Ranking needs a prior; without one the package returns NaN for
`log_r_per_unit_z` rather than substituting something that looks similar.

**Known open issues**, all recorded in `AGENTS.md`:

- `tune_shrinkage` cannot see its own parameter (it holds out at the parent
  HEALPix resolution), so the background pooling constant is fixed, not tuned.
- The selection-bias measurement is unresolved: DESI- and SDSS-selected quasars
  cannot be compared by raw density, because they differ by 0.7 mag in
  brightness, which changes the density mechanically.
- The original five-band pipeline covers the southern photometric system.
  The multi-survey extension below has separately labelled northern and
  southern Legacy bands.
- Blends are out of scope: `BlendPolicy` refuses companions below a stated
  separation rather than scoring them badly — and only when you pass one.
- Close-pair validation currently covers the southern model. The multi-survey
  checks use reserved individual quasars and field sources. `p_zmatch_given_qso`
  is conditional on the companion being a quasar *inside the trained range*; a
  candidate whose colours are best explained beyond z ≈ 4.4 is reported via
  `frac_norm_outside_support`, not scored as if it were inside.

---

## Install

```bash
source ~/Work/venvs/.venv/bin/activate      # or your own environment
pip install -e ".[dev,wsdb]"                 # dev = pytest, wsdb = sqlutilpy
python -m pytest -q                          # 149 tests, ~43 s
```

Python ≥ 3.11 with numpy, scipy, astropy, healpy, matplotlib. `pip install -e .`
alone installs neither pytest nor `sqlutilpy`, so use the extras above: the test
command needs the first and everything that touches data needs the second.

The two examples using the saved models run offline. Fitting a local background
or rebuilding the training samples needs WSDB, `sqlutilpy`, and credentials
(`PGUSER` / `PGHOST` / `~/.pgpass`).

---

## Scoring a candidate — offline, straight from a clone

This example loads the **original southern Legacy DR9 model**. Everything
it needs ships in `models/`: the quasar colour model, a
footprint-average background colour model with its surface density, and the
quasar surface density. No database access is required for this.

```python
import numpy as np
from qso_pcolor.background import BackgroundColourModel
from qso_pcolor.features import RelativeFluxTransform, deredden
from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
from qso_pcolor.score import BlendPolicy, score_candidates

BANDS = ("g", "r", "z", "w1", "w2")
qso   = SlicedColourRedshiftModel.load("models/qso_south_full.json")
bkg   = BackgroundColourModel.load("models/background_south_global.json")
dens  = BackgroundSurfaceDensity.load("models/background_density_south_global.json")
prior = GridQSOPrior.load("models/sigma_q_south.json")
tr = RelativeFluxTransform(reference_band="r")

# Legacy Surveys fluxes, inverse variances and transmissions (nanomaggies)
flux  = np.array([[1.9, 2.6, 3.1, 11.0, 14.0]])
ivar  = np.array([[120.0, 150.0, 60.0, 8.0, 3.0]])
trans = np.array([[0.97, 0.98, 0.99, 1.0, 1.0]])
f, v = deredden(flux, ivar, trans)
feat = tr(f, v, BANDS)

rows = score_candidates(
    feat, z_primary=np.array([1.8]),
    l_deg=np.array([276.337]), b_deg=np.array([60.189]),   # (RA,Dec)=(180,0)
    qso_model=qso, background_model=bkg, background_density=dens,
    qso_prior=prior, match=RedshiftMatch(half_width_kms=2000.0),
    blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
    separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]),
)
s = rows[0]
print(s.log_bayes_factor_qz_bkg)   # +3.42   evidence: quasar at z0 vs background
print(s.log_r_per_unit_z)          # -2.52   the ranking statistic
print(s.p_sameq)                   # 3.0e-03 posterior for the ±2000 km/s window
print(s.status)                    # 'ok'
```

`tests/test_shipped_models.py` is this example, executed; if the numbers drift
the suite fails. The shipped background is a footprint average from eight
0.5° fields at |b| > 32° (provenance in the file's `meta`).

**Global or local?** Measured in 30 cones from |b| = 20° to 78°
(`scripts/compare_background_modes.py`): above |b| ≈ 45° the local model is
indistinguishable from the shipped one; between 20° and 45° it describes the
field better by 0.1–0.2 nats per object, but neither the false-positive rate
nor the quasar scores move by more than ~0.1 nat. **Use the shipped global
model by default.** Fit a local one for |b| ≲ 25°, for fields denser than
~3×10⁴ sources/deg², or when a candidate's neighbourhood is known to be odd.

## Scoring a candidate — with a local background (needs WSDB)

Same candidate, but the field population is fitted in the candidate's own 0.5°
neighbourhood. The Bayes factor moves from +3.42 to +4.91: the background at
(180°, 0°) is sparser in this part of colour space than the footprint average,
and the scorer says so.

```python
from qso_pcolor.background import fit_local_background

bkg, dens, info = fit_local_background(
    ra=180.0, dec=0.0, radius_deg=0.5, transform=tr, bands=BANDS,
    mag_edges=np.array([17.0, 19.5, 20.5, 21.5, 22.5]),
    system=qso.system, max_ref_mag=22.5, seed=0, max_iter=200,
)

rows = score_candidates(
    feat, z_primary=np.array([1.8]),
    l_deg=np.array([276.337]), b_deg=np.array([60.189]),
    qso_model=qso, background_model=bkg, background_density=dens,
    qso_prior=prior, match=RedshiftMatch(half_width_kms=2000.0),
    blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
    separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]),
)
print(rows[0].log_bayes_factor_qz_bkg)   # +4.91
```

Leave out `qso_prior` and `p_sameq` and `log_r_per_unit_z` come back `nan` with
`status='no_prior_posterior_unavailable'` — **by design**: the package returns
the prior-independent evidence rather than inventing a posterior.
An empty prior at the candidate magnitude likewise retains the likelihoods and
Bayes factor, with `status='qso_prior_empty_at_this_magnitude'`.

Every row also carries `log_r_per_unit_z` (the ranking statistic),
`dz_match_eff`, the three log intensities, an out-of-distribution score,
`frac_norm_outside_support` (how much of the redshift normalisation the scorer
discarded as lying outside the trained range), quality flags and a status code.
§10 of the method note lists them all.

### Things that will bite you

- **Mixing photometric systems.** North and south Legacy Surveys are different
  systems; the scorer raises rather than silently combining them.
- **Reading `p_sameq` as "probability of a binary".** A ±2000 km/s window is
  Δ*z* = 0.037 against a photometric redshift width of ~0.6, so `p_sameq` stays
  small even for a perfect candidate. Rank on `log_r_per_unit_z`.
- **Ranking on the Bayes factor.** See above: AUC 0.759 against 0.815.
- **A background model with no galaxies in it.** The first validation used an
  eight-field background fitted to `type = 'PSF'` sources only, and reported
  galaxies as a serious contaminant (AUC 0.80, 10.7 % above the same-*z*
  median). With the shipped all-source background those numbers are 0.96 and
  0.3 %. The background must contain everything a chance neighbour can be.
- **Feeding it blended pairs.** Below ~3″ the survey photometry does not give two
  independent measurements. `BlendPolicy` exists to refuse them, not to
  down-weight them.

---

## What is in here

| path | what |
|---|---|
| `src/qso_pcolor/` | the package: mixtures, extreme deconvolution, features, models, priors, scorer |
| `scripts/train_qso_model.py` | train the quasar colour model (DESI ± SDSS, spatial holdout, K selection) |
| `scripts/score_examples.py` | worked example: 10 real objects, per-object local backgrounds, figure |
| `scripts/redraw_optical_examples.py` | redraw Figure 10 from saved inputs in luptitude colours; no queries or fitting |
| `scripts/plot_colour_redshift.py` | ten intrinsic quasar colour densities versus redshift, including four PS1 colours; saved model only |
| `scripts/build_pair_validation.py` | labelled close-pair sample from DESI DR1 (579,572 companions with spectra) |
| `scripts/validate_pairs.py` | the validation: ROC, reliability, contaminants by spectype, figure |
| `scripts/make_method_figures.py` | the method note's figures |
| `scripts/build_global_background.py` | the shipped footprint-average background, with provenance |
| `scripts/compare_background_modes.py` | global vs local field model in 30 cones: when does local matter? (measured: |b| ≲ 25°) |
| `scripts/recover_holdout_blocks.py` | recover a trained model's spatial holdout and record it |
| `scripts/extend_qso_model_redshift.py` | widen a trained model's redshift range by appending slices |
| `scripts/check_redshift_extension.py` | did that extension buy anything? (measured: +1.9 nats) |
| `scripts/check_em_convergence.py` | does the EM iteration cap matter? (measured: no) |
| `scripts/smoke_real_data.py` | end-to-end plumbing check against live WSDB |
| `tools/journal.py` | JOURNAL.md updater; `hook-install` journals every commit |
| `docs/method/method.pdf` | **the method note** — formalism, derivations, 9 figures |
| `docs/REVIEW_OF_PLAN.md` | why the design departs from the original plan |
| `configs/example_ls_dr9.yaml` | every tunable, with nothing defaulted that changes meaning |

`python scripts/<name>.py --help` for options.

## Models and data

Four files are committed, and together they are everything the scorer needs:

| file | what |
|---|---|
| `models/qso_south_full.json` | the quasar colour–redshift model |
| `models/background_south_global.json` | footprint-average background colour model (all source types, `maskbits = 0`, known quasars removed) |
| `models/background_density_south_global.json` | its surface density Σ_B, mask-corrected area |
| `models/sigma_q_south.json` | the quasar surface density Σ_Q(z, m), global and isotropic |

`models/qso_south_full.json` is 1,106,986 training quasars
(917,489 DESI DR1 + 189,497 SDSS DR16Q, de-duplicated at 1″, `maskbits = 0` on
both channels), Legacy Surveys DR9 south (`release` 9010), 43 redshift slices
covering 0.15 < z < 4.35 trained natively over 0.1–4.4. K = 20 where a slice
has ≥ 10,000 objects; below that K is chosen per slice by held-out density
(12, 8 and 4 at the high-redshift end). 16 of the 81 populated nside=4 sky
blocks were reserved before fitting, holding out 338,170 objects (23.4 % of the
sample; the 20 % is a fraction of *blocks*, not of objects). Those 16 block IDs and the seed are recorded in the file's `meta`, so
downstream code reads the split instead of re-deriving it — re-deriving it is
what produced a figure caption claiming five held-out quasars when three of
them were in the fit. Enough to score candidates without retraining.

History: the range was first widened from 0.4–3.6 by *appending* 11 slices
(`scripts/extend_qso_model_redshift.py`; +1.90 nats on 15,552 held-out quasars
outside the old range), then the whole model was retrained natively on
2026-09-20 with the `maskbits` cut applied to both channels and the same 16
holdout blocks read in. On 40,000 identical mask-clean held-out quasars the
retrained and appended models agree to +0.0008 nats; the validation AUCs moved
by ≤ 0.006. The retrain bought consistency and provenance, not accuracy — as
expected once we measured that masked quasars sit 0.85 nats off the clean
locus but are only 5.4 % of the sample.

Large catalogue caches and intermediate fits are gitignored. Rebuilding them needs **WSDB access**
(`sqlutilpy`, credentials via `PGUSER` / `PGHOST` / `~/.pgpass`):

```bash
python scripts/train_qso_model.py --system south --select-k   # ~4 h
python scripts/make_method_figures.py                          # figs 1-8, ~8 min
python scripts/redraw_optical_examples.py                       # fig 10, saved objects/models/scores
python scripts/score_examples.py                               # rebuild the original sample and scores; cached fits
make -C docs/method                                            # rebuild the PDF
```

The Figure 10 redraw uses committed inputs and runs offline; rebuilding the
PDF also needs no database access.

Queries cache to `data/`, keyed by a hash of the query text, so a rerun reuses
the same selection. SDSS training matches also carry the parent query in their
cache identity. The fitted `models/method_*.json` reload unless `--refit` is
passed.

## Combining surveys, including infrared-only inputs

The extension in `qso_pcolor.multisurvey` uses **SDSS, DECaLS/Legacy DR9,
ALLWISE, PS1, NSC, SkyMapper, and VHS**, individually or in combination.
Every survey/filter has its own label. Northern and southern Legacy filters
remain separate, and Legacy forced WISE measurements are not relabelled as
ALLWISE. The original southern model and its offline example above are retained.

The saved `models/multisurvey.json` uses **67,574 quasars** in 43 redshift
slices (support 0.15–4.35), with **14,766 quasars reserved** in the existing
spatial holdout. All **127 non-empty survey combinations** were checked on
reserved quasars and field sources. See the [validation report](docs/MULTISURVEY_VALIDATION.md)
for the samples, fit stopping criteria, and measured performance. Supporting
a combination does not imply equal precision: VHS-only QSO/field separation
is weak in this validation (AUC 0.620), while ALLWISE-only gives 0.903.

An [identical-data comparison](docs/MODEL_COMPARISON.md) uses southern Legacy
DR9 **g,r,z only** in both models. It identified a background-model discrepancy
on these bands. A separate grz background, selected using reserved
training fields, restores close overall discrimination without refitting the
quasar model. It applies only when every observed band belongs to southern
Legacy grz; other inputs keep the joint background.

On a [fresh confirmation sample](docs/MODEL_COMPARISON_FRESH.md) of 2,000
held-out quasars and 2,000 different stars/galaxies, original/new AUC is
**0.962/0.959** for quasars versus non-quasars and **0.790/0.791** for redshift
discrimination. Star rejection remains slightly weaker (**0.970/0.953**),
and individual scores are not interchangeable. All 127 survey combinations
were rechecked; the other 126 rows are unchanged. The method and both sets of
tests appear in the method note's **main text**. The original southern models
and the [initial joint-only extension](models/multisurvey_joint_20260921.json)
remain available.

Twelve [Figure 10-style comparisons](docs/examples/README.md) show the same
five held-out quasars and five unclassified field sources using each survey
alone, DECaLS+ALLWISE, SDSS+ALLWISE, PS1+ALLWISE, ALLWISE+VHS, and all seven.
The gallery includes a twelve-page PDF, full-resolution PNGs, the fixed sample,
and exact scores. Two-colour contours accompany scores from all available bands
in each combination. The field examples share one reserved overlap field;
these ten objects illustrate behaviour rather than measure survey-wide accuracy.
Reproduce them offline with `python scripts/make_multisurvey_examples.py`
(the `figures` installation extra supplies the PDF dependency).
Figure 10 now uses luptitude-colour axes too, retaining its original objects,
models and scores. Its previous [flux-ratio rendering](plots/examples/optical_only_examples_flux_ratio.png)
is archived; the redraw includes the density Jacobian and full colour errors.

Figures 12–13 in the method note show the model's **colour density versus
redshift**: [six optical/infrared colours](plots/method/colour_redshift_core.png)
and [PS1 g−r, r−i, i−z and z−y](plots/method/colour_redshift_ps1.png).
Brightness and unused bands are integrated out, with no added measurement
noise. Every redshift column is a unit-integral colour density; both figures
share one absolute density scale and show the median and 16th–84th percentiles.
The colours retain the model's observed native calibration (optical AB,
ALLWISE/VHS Vega). Reproduce them offline with
`python scripts/plot_colour_redshift.py`; band choices and display settings
are in [the configuration](configs/colour_redshift.json), with model provenance
and integration checks in [the numerical report](docs/COLOUR_REDSHIFT.json).

The new mixtures learn the joint band distribution. For each object, the
scorer conditions on an available reference band and marginalises the absent
bands. Thus ALLWISE-only, VHS-only, and mixed optical/infrared inputs use the
same three-hypothesis calculation. Survey likelihoods are evaluated jointly;
they are not multiplied as independent evidence. Two measured bands provide
one colour, with the minimum band count supplied explicitly.

The inputs use **native calibrated flux units**: magnitude 22.5 has flux one.
Optical and infrared zero-point conventions are recorded separately; ALLWISE
and VHS retain Vega calibration. `catalogue_photometry` converts the verified
WSDB columns, preserves negative raw fluxes, and applies the same configured
quality cuts for training, field sources, and candidates. The current new
sample uses observed photometry at high Galactic latitude, without dereddening.
Pass observed measurements to this model. Its metadata records this convention.

For example, a diagnostic using only two infrared bands is:

```python
import numpy as np
from qso_pcolor import MultiSurveyModel, Photometry, RedshiftMatch

model = MultiSurveyModel.load("models/multisurvey.json")
phot = Photometry(
    flux=np.array([[400.0, 830.0]]),
    variance=np.array([[40.0**2, 100.0**2]]),
    bands=("allwise:w1", "allwise:w2"),
)
row = model.score(
    phot, z_primary=np.array([1.8]),
    l_deg=np.array([180.0]), b_deg=np.array([45.0]),
    match=RedshiftMatch(half_width_kms=2000.0), min_bands=2,
)[0]
print(row.log_bayes_factor_qz_bkg, row.p_zmatch_given_qso)
print(row.reference_band, row.bands_used, row.status)
```

The returned row retains the existing evidence/posterior contract and adds
the reference band, bands used, and surveys used. Input columns can arrive in
any order; absent bands are represented by missing entries. A real companion
score must also supply a `BlendPolicy` and its required measurements. Mark
contaminated bands unusable rather than borrowing a primary's infrared flux.

**Population posteriors require matching priors.** The old southern
r-magnitude prior cannot be applied to an infrared reference or to the new
luptitude coordinate. Without an appropriate prior pair, the new model returns
colour evidence and `p_zmatch_given_qso`, with posterior fields and `log R`
unavailable. To supply priors, pass
`priors={reference_band: (qso_prior, background_density)}`; each prior's metadata
must contain that `reference_band` and `transform_id=model.transform_id`.
Those surface densities must describe the same photometric selection and be
normalised per reference-band luptitude. Relabelling the old priors is invalid.

The reproducible build uses the shared environment and caches the catalogue
queries and individual fits. Rebuilding needs WSDB and the DESI/SDSS parent
caches named in `configs/multisurvey.json`; using the saved model needs neither.
The commands write the extension to `models/multisurvey.json` and preserve all
four original model/prior files:

```bash
python scripts/build_multisurvey_sample.py --config configs/multisurvey.json
python scripts/train_multisurvey_model.py --config configs/multisurvey.json
python scripts/fit_background_marginal.py --config configs/background_marginal.json
python scripts/build_multisurvey_sample.py --part validation
python scripts/validate_multisurvey.py --config configs/multisurvey.json
python scripts/compare_old_new_models.py --config configs/model_comparison_joint.json
python scripts/compare_old_new_models.py --config configs/model_comparison.json
python scripts/compare_old_new_models.py --config configs/model_comparison_fresh.json
```

The configuration records survey matching, quality selection, the seed, and
fit settings. Quasar validation reuses the saved spatial holdout; background
validation reserves whole fields. Component selection uses a separate split
inside the training sample. See [the method note](docs/method/method.pdf),
sections “Extension to arbitrary survey combinations” and “Validation:
design and results”, for the conditional likelihood, its normalisation,
the matched old/new test, and the survey-combination results.

## Conventions

- Figures are PNG under `plots/`, written with `qso_pcolor.plotting.save_figure`.
- `JOURNAL.md` is local and gitignored; `python tools/journal.py hook-install`
  makes every commit journal itself.
- `AGENTS.md` is the working contract — read it before changing anything.

## Citing

Queries use q3c (Koposov & Bartunov 2006,
<https://ui.adsabs.harvard.edu/abs/2006ASPC..351..735K>). Training data are DESI
DR1 and SDSS DR16Q (Lyke et al. 2020). The method note's bibliography has the
rest.

**No licence file yet** — add one before sharing outside the group.

## Acknowledgements

The code, the method note and the analyses in this repository were written with
**Claude** (Anthropic) in Claude Code, working from the scientific design and
under the review of the author. **OpenAI Codex** was used as an independent
reviewer of the pipeline and the write-up.

Responsibility for the method and for what is claimed of it rests with the
author.
