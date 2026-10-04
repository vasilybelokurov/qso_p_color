# qso_pcolor

Is the photometric companion of a known quasar itself a quasar at the same
redshift?

Given a quasar with a spectroscopic redshift *z*₀ and a companion a few
arcseconds away with broadband photometry only, this package weighs four
competing explanations and reports how strongly the photometry supports the
first:

| hypothesis | meaning |
|---|---|
| `same_z` | a quasar whose redshift matches *z*₀ |
| `field_q` | a quasar at some other redshift |
| `bkg` | anything else in the imaging, as the field model describes it |
| `out` | a broad, fitted share of the field that neither model describes |

The second row is why the package has this shape. A quasar-versus-star
calculation cannot tell a genuine companion from a foreground quasar at
*z* = 2.6, and for a binary-quasar search that is the failure mode that matters.
The fourth stops an object unlike anything either model was fitted to from
being called a certain quasar because of how two Gaussian tails fell.

**The method is written up in [docs/method/method.pdf](docs/method/method.pdf).**
`AGENTS.md` is the working contract for changing the code.

---

## Current status and model choice

**Method write-up:** `docs/method_unified/method_unified.tex` (built with `make -C docs/method_unified`)
describes the candidate bundles of 3 October below (figures from `scripts/method_unified/`; set
`NOTE_MODELS=promoted` to draw them from the promoted pointers). The earlier note, `docs/method/method.tex`, is retired: it describes
models before 2 October 2026 and is kept unchanged for reference.

**Two models are active (2 October 2026).** Both use one shared North/South latent model over
all 41 bands, Galactic-extinction-corrected photometry (SFD98; the scorer corrects catalogue
input from position), a MAP covariance update with predictive early stopping, and a calibrated
QSO-support cut at 99.5% calibration retention. They differ only in the quasar colour model:

| Pointer | QSO colours | Bundle |
|---|---|---|
| `models/multisurvey_psf/current` | independent of magnitude (XDQSO design) | `13866e45ef794059` |
| `models/multisurvey_psf/current_magdep` | free to depend on magnitude | `5f4002492dbb849c` |
| `models/multisurvey_psf/previous` | rollback: the 28 September bundle | `630f47f63b6f0694` |

```python
from qso_pcolor import Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.unified import UnifiedPSFModel

model = UnifiedPSFModel.load("models/multisurvey_psf/current")   # or current_magdep
# Catalogue fluxes as observed; extinction is corrected internally from RA/Dec.
phot = Photometry([[1.9, 2.6, 3.1]], [[1/120, 1/150, 1/60]],
                  ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z"))
scores, decision = model.score(phot, ra_deg=180., dec_deg=0., z_primary=1.8, morphology=["PSF"],
    match=RedshiftMatch(half_width_kms=2000.), blend_policy=BlendPolicy(3., .2),
    separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.)
```

At 99.5% retention, both pass all eight release ranking panels against the previous bundle.
The magnitude-dependent model fits held-out quasars about 0.67 nats/object better, but ranks
quasars against contaminants only about 0.002 AUC better on average. The magnitude-independent model relies
more on the support cut. The cut threshold was chosen on the reporting rows and still needs
confirmation on independent rows. Probability calibration is not established. See
[the comparison](docs/SUPPORT_CUT_SWEEP_2026-10-01.md) and the
[extinction coefficients](configs/extinction_coefficients.json).

### Candidate bundles (3 October 2026, not yet promoted)

**Two models on an equal footing.** `independent` (quasar colours independent of apparent magnitude,
the XDQSO assumption) and `dependent` (quasar colours free to depend on it) share everything else.
Neither is the default. The magnitude-dependent model is better on the (DESI/SDSS-selected) test
quasars, most clearly for bright ones; the magnitude-independent one makes the weaker assumption for
quasars outside that selection. Score candidates with both; see the method note, Section 6.2.

`models/multisurvey_psf/work/stellar_binned/20261003/bundles/{independent,dependent}` keep the
quasar models of the promoted bundles and change the non-quasar side:

- **Background (stars and other non-quasars):** XDQSO-style, eight independent 20-component fits in
  bins of Legacy r (15.5-24.5), joined into one 160-component mixture; training sample cleaned of likely
  quasars (Quaia members and WISE W1-W2 > 0.8 Vega; `scripts/clean_stellar_qso_contamination.py`);
  component weights vary smoothly with Galactic position (softmax gate); catch-all and background counts
  redone on the cleaned sample.
- **Faint limit (new behaviour).** Only sources with Legacy DR9 r (South, else North) at S/N >= 10 are
  scored, and Legacy r is always the reference band. Others are returned with `decision['eligible']`
  False and `decision['reason']` = `too_faint` (S/N < 10) or `no_faint_limit_band` (no Legacy r, e.g.
  SDSS-only or PS1-only input). This removes 2.8% of test quasars.
- **Results** (held-out, same rows as the promoted bundles with the same limit): AUC unchanged
  (0.976 / 0.980), more quasars with p_quasar > 0.5 (+2.3 / +1.9 points), fewer background objects
  above 0.5. p_quasar is consistent with observed quasar fractions in the mid range and mildly
  overconfident above 0.97 (method note, Section 10.7). `qso_pcolor.unified.quasar_probability` computes it
  from a score record.

```python
model = UnifiedPSFModel.load("models/multisurvey_psf/work/stellar_binned/20261003/bundles/independent")
scores, decision = model.score(phot, ra_deg=..., dec_deg=..., z_primary=..., ...)
p_q = model.quasar_probability(scores)          # NaN for unscored rows
```

### The previous bundle (28 September 2026)

The description below is the previous PSF model, `PSFMultiSurveyBaseline`, now loaded from
`models/multisurvey_psf/previous`. It accepts **any nonempty subset of all 41
bands**, including optical-only and infrared-only input. Missing bands are
marginalised from one joint model. No band is compulsory. A single band has
no colour information and is flagged accordingly.

The stellar/background component depends on colour, magnitude and Galactic
position, using shared Gaussian shapes with HEALPix weights (`nside=4`, parent
`nside=2`). Both Legacy hemispheres are active. Population priors cover all 41
reference bands; sparse priors are flagged. `baseline.fit_local(...)` refits
the same weights and counts in a declared cone with measured usable areas.

The active catch-all is a Student-t with pooled, strictly positive magnitude-bin
weights and Gaussian measurement-noise convolution. A fitted broad Gaussian is
retained for comparison. Both use the same joint 41-band marginalisation;
neither changes the stellar model's continuous magnitude dependence. See the
[catch-all comparison](docs/CATCHALL_COMPARISON_2026-09-28.md) for the held-out
results and colour-plane stress tests. The outside-both-models exclusion remains
required: a catch-all alone is not a certificate of reliable colour-space support.

The [recovery validation](docs/VALIDATION_multisurvey_psf_2026-09-28.json)
checks all 41 single bands, 820 pairs, larger subsets and 127 survey selections.
On reserved PSF sources, mean predictive log density improved by 0.069 nats
for 12,072 quasars and 1.161 nats for 14,511 field objects against the earlier
joint model. These checks establish functionality and predictive improvement;
they do not establish probability calibration.

```python
from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy

baseline = PSFMultiSurveyBaseline.load("models/multisurvey_psf/previous")
# Native observed fluxes, in each band's original photometric system.
phot = Photometry([[1.9, 2.6, 3.1]], [[1/120, 1/150, 1/60]],
                  ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z"))
scores, decision = baseline.score(
    phot, morphology=["PSF"], z_primary=1.8, l_deg=276.337, b_deg=60.189,
    match=RedshiftMatch(half_width_kms=2000.),
    blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
    separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.,
)
# To choose a different subset, pass a shorter Photometry or use keep_bands().
print(scores[0].bands_used, scores[0].status, decision["eligible"])
```

Morphology is supplied separately from the chosen photometry. Unknown or
extended morphology is excluded, as are missing blend measurements and
unsupported scores. Rank only `decision['eligible']` rows by
`log_r_per_unit_z`; quote `p_sameq` together with `dz_match_eff`.

### Earlier models retained for comparison

The Legacy DR9 PSF baseline has returned to dereddened relative fluxes and
improves broad candidate ranking. **Its full release validation fails:** the
observable-field tail prediction and quasar score-stability checks do not pass.
Its science interface now excludes missing blend information and objects
outside both fitted populations. Use it for exploratory ranking with these
limits; do not treat the earlier validation PASS as a complete certification.
See the [recovery report](docs/RECOVERY_2026-09-28.md).

The earlier Legacy-only PSF model includes a spatial background: Gaussian component
weights and source surface densities vary with magnitude and Galactic HEALPix
cell. Both hemispheres use `nside=4`, parent `nside=2`, with cell-to-parent-to-global
pooling. Component means and covariances remain shared. The
[spatial validation report](docs/SPATIAL_BACKGROUND_2026-09-28.md) records the
field-only tests, configuration and remaining limits.

| Input and scope | Interface | Saved files |
|---|---|---|
| Legacy DR9 PSF companions, 17 <= dereddened r < 22.5, Galactic \|b\| >= 25 degrees | `XDQSOBaseline` | `models/legacy_psf_xdqso/current` (base `8e2a27c013ed` + spatial background); north provisional |
| Other survey combinations, earlier multi-survey release | `MultiSurveyModel` | `models/multisurvey*.json`; native observed fluxes, separate field-model limitations |

### Legacy PSF scoring

Supply native catalogue rows, including `mw_transmission_*`; the adapter
dereddens flux and variance internally. Declare both policies. Missing separation
or `fracflux` excludes the companion; outside both populations the diagnostic
row remains, but its ranking and posterior fields are NaN.

```python
import numpy as np
from qso_pcolor import BlendPolicy, RedshiftMatch
from qso_pcolor.baseline import XDQSOBaseline

baseline = XDQSOBaseline.load("models/legacy_psf_xdqso/current")
# Synthetic catalogue row; unit transmissions mean zero Galactic extinction.
rows = dict(ra=np.array([180.]), dec=np.array([0.]), release=np.array([9010]),
            type=np.array(["PSF"]), maskbits=np.array([0]))
for band, flux in zip(("g", "r", "z", "w1", "w2"), (8., 10., 12., 40., 60.)):
    rows[f"flux_{band}"] = np.array([flux])
    rows[f"flux_ivar_{band}"] = np.array([100.])
    rows[f"nobs_{band}"] = np.array([3])
    rows[f"mw_transmission_{band}"] = np.array([1.])
scores, decision = baseline.score_rows(
    rows, z_primary=1.8, match=RedshiftMatch(half_width_kms=2000.),
    blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
    separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.,
)
print(decision["eligible"], scores[0].status, scores[0].model_manifest_id)
# Rank only eligible rows. A posterior always travels with its window width.
print(scores[0].log_r_per_unit_z, scores[0].p_sameq, scores[0].dz_match_eff)
```

The support threshold is a declared selection policy, not a calibrated outlier
fraction. Quasar densities and Gaussian component shapes retain their original
fits; the active background uses the fitted sky-dependent weights and counts.
Scores record the composite model ID, background mode, nside and scoring policy.
Load `models/legacy_psf_xdqso/8e2a27c013ed` explicitly to reproduce the earlier
spatially pooled baseline.

## Multi-survey model (earlier release)

This model accepts **any combination of 41 bands** from seven
surveys, including infrared-only input:

The evaluator marginalises the same joint distribution for every subset;
there is no compulsory band. Use `phot.keep_bands(("sdss:u", "allwise:w2"))`
to choose individual measurements. The default `min_bands=1` accepts a single
band and flags `no_colour_information`: its colour Bayes factor is one, and
only population priors can supply a posterior. Separate fitted subset models
are used only with `MultiSurveyModel.load(..., legacy_field_fits=True)` for
historical reproduction. The earlier population fits and calibration below
have not thereby become a validated PSF/spatial multi-survey release.

| survey | bands |
|---|---|
| SDSS | *u g r i z* |
| Legacy Surveys DR9, south and north separately | *g r z* and the forced unWISE *W1 W2* |
| AllWISE | *W1 W2 W3 W4* |
| Pan-STARRS1 | *g r i z y* |
| NSC | *u g r i z y* VR |
| SkyMapper | *u v g r i z* |
| VHS | *Y J H K*s |

Filters from different surveys are never treated as the same filter. The
Legacy forced *W1, W2* and the AllWISE *W1, W2* are different measurements and
different bands; the Legacy ones are the single strongest quasar/star
discriminant available.

Three files, all in `models/`, are the model:

| file | contents |
|---|---|
| [multisurvey.json](models/multisurvey.json) | quasar and field densities over native band luptitudes, the luptitude transform, the band schema, dedicated Legacy-only field fits, training manifest |
| [multisurvey_priors.json](models/multisurvey_priors.json) | surface densities Σ_Q(z, u_a) and Σ_B(u_a) for 37 of the 41 possible reference bands |
| [multisurvey_outlier.json](models/multisurvey_outlier.json) | the unmodelled hypothesis: a heavy-tailed Student-t (ν = 2) at the field's own scale, and its fitted share of the field |

Without the priors you get the colour evidence and the quasar-only redshift
probability; with them, the posterior and the ranking statistic *R*.

The quasar densities are 43 redshift slices (support 0.15 < *z* < 4.35) fitted
to **67,574** DESI DR1 and SDSS DR16Q quasars drawn flat in redshift, with
**14,766** more reserved in sixteen sky blocks before fitting. The field density
is fitted to every source, of every type, in 24 cones at |*b*| ≥ 25°, with known
quasars removed and six whole cones reserved.

**Historical release results** (including dedicated subset field fits), measured on 50,746 companions of DESI DR1 quasars that
have their own DESI spectra, scored from Legacy *g, r, z, W1, W2*, in sky
blocks the model never saw (method note §10):

| question | held-out AUC |
|---|---|
| same-*z* vs wrong-*z* quasar, ranked by `log_r_per_unit_z` | **0.828** |
| same, by `p_zmatch_given_qso` alone | 0.865 |
| quasar vs spectroscopic star, by Bayes factor | **0.982**; 0.04 % of stars above the median same-*z* quasar |
| quasar vs spectroscopic galaxy | **0.961**; 0.18 % above |
| same-*z* vs quasars just outside the window (3,000–6,000 km/s) | 0.472 — chance |

Across all 127 survey combinations, reserved quasars are separated from
reserved field sources with median AUC 0.996 (VHS alone is the weakest, 0.662;
[report](docs/MULTISURVEY_VALIDATION.md)). The predicted same-redshift
probability is low by a factor that falls from 9.8 at 3–5″ to 3.3 at 20–30″:
that is quasar clustering, which the scorer deliberately leaves out.

---

## Install

```bash
git clone https://github.com/vasilybelokurov/qso_p_color.git
cd qso_p_color
python3 -m venv .venv                        # Python 3.11 or newer
source .venv/bin/activate
pip install -e ".[dev]"                      # package plus pytest
python -m pytest -q                          # run the regression suite
```

The `wsdb` extra (`sqlutilpy`) is needed only to query the database and
rebuild samples; the `figures` extra only for the example atlas PDF. Scoring
with the saved model needs neither. Run from the repository root so the
`models/` paths resolve; the model files are in Git, not in the wheel.

## Scoring with the earlier multi-survey release

Offline, straight from a clone:

```python
import numpy as np
from qso_pcolor import (BlendPolicy, MultiSurveyModel, MultiSurveyOutlier,
                        Photometry, RedshiftMatch, load_priors)

model  = MultiSurveyModel.load("models/multisurvey.json")
priors = load_priors("models/multisurvey_priors.json", model)
out    = MultiSurveyOutlier.load("models/multisurvey_outlier.json")

# Legacy DR9 south fluxes as observed (nanomaggies, not dereddened) and their variances
phot = Photometry(
    flux=np.array([[1.9, 2.6, 3.1, 11.0, 14.0]]),
    variance=1.0 / np.array([[120.0, 150.0, 60.0, 8.0, 3.0]]),
    bands=("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z",
           "decals_dr9_south:w1", "decals_dr9_south:w2"),
)
s = model.score(
    phot, z_primary=np.array([1.8]),
    l_deg=np.array([276.337]), b_deg=np.array([60.189]),     # (RA,Dec) = (180,0)
    match=RedshiftMatch(half_width_kms=2000.0), min_bands=2,
    priors=priors, outlier=out, ood_flag_sigma=4.0,
    blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
    separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]),
)[0]
print(s.log_bayes_factor_qz_bkg)   # +6.68    evidence: quasar at z0 vs the field
print(s.log_r_per_unit_z)          # -1.24    the ranking statistic
print(s.p_sameq, s.dz_match_eff)   # 1.077e-02 posterior for the ±2000 km/s window, and its width
print(s.p_outlier)                 # 2.3e-03  share taken by "unmodelled"
print(s.qso_ood_sigma_any_z, s.bkg_ood_sigma)   # 0.43 2.33  distance to each model, in sigma
print(s.reference_band, s.status)  # decals_dr9_south:r ok
```

`tests/test_shipped_models.py` is this example, executed; if the numbers drift,
the suite fails.

**Any other survey**: change `bands` and the matching columns; the model and
the call stay the same. `model.transform.bands` lists every accepted label.
Infrared only:

```python
phot = Photometry(flux=np.array([[400.0, 830.0]]),
                  variance=np.array([[40.0**2, 100.0**2]]),
                  bands=("allwise:w1", "allwise:w2"))
row = model.score(phot, z_primary=np.array([1.8]), l_deg=np.array([180.0]),
                  b_deg=np.array([45.0]), match=RedshiftMatch(half_width_kms=2000.0),
                  min_bands=2, priors=priors, outlier=out)[0]
print(row.log_bayes_factor_qz_bkg, row.log_r_per_unit_z)   # +0.98 -3.84
print(row.reference_band, row.bands_used, row.status)      # allwise:w1 ('allwise:w1', 'allwise:w2') ok
```

### Input conventions

- **Native calibrated fluxes**, magnitude 22.5 = flux 1: Legacy and SDSS in
  nanomaggies (AB); AllWISE and VHS keep their Vega zero points (AllWISE DN
  fluxes use zero points 20.5, 19.5, 18.0, 13.0 for *W1–W4*).
  `qso_pcolor.multisurvey_data.catalogue_photometry` converts the WSDB columns
  and applies each survey's quality cuts; use it, or the same cuts, for
  candidates.
- **Observed, not dereddened.** The training and field samples are at
  |*b*| ≥ 25°, and the model's metadata records the convention.
- **Keep negative fluxes.** They are measurements. Mark a band unusable only
  when the survey says so: `flux=np.nan`, `variance=np.inf`.
- **Legacy north and south are different bands** (`decals_dr9_north:*` for
  release 9011, `decals_dr9_south:*` for 9010 and 9012).
- **A real companion needs a `BlendPolicy`** with its separation and
  reference-band `fracflux`; without them it is refused as blended. Mark
  contaminated bands — AllWISE beside a bright primary, typically — unusable
  rather than scoring them.

### How to read the output

- **Rank on `log_r_per_unit_z`.** It is the window-free evidence: two people can
  compare candidates without agreeing on a window. `p_sameq = exp(log_r_per_unit_z) * dz_match_eff`
  exactly; never quote `p_sameq` without `dz_match_eff`.
- **`p_sameq` is small even for a perfect candidate.** A ±2000 km/s window at
  *z* = 1.8 is Δ*z* = 0.037; the photometric redshift width is ~0.6. It is also
  not the probability of a physical pair: multiply the odds by the clustering
  factor for the separation (9.8 at 3–5″, 7.4, 4.9, 3.3 at 20–30″) if you want
  one.
- **The Bayes factor is not an alternative ranking statistic.** It has no
  field-quasar term: AUC 0.720 against 0.828 for same- vs wrong-*z*. Use it to
  reject stars.
- **`outside_both_models`** (set when both distances exceed your
  `ood_flag_sigma`) means every density is an extrapolation. With the outlier
  model such an object goes to `p_outlier` instead of being called a quasar;
  either way it needs a spectrum or a second look, not a number.
- **Without a prior for the reference band** (NSC *u*, SkyMapper *u*, *v*,
  VHS *Y*), `status='no_prior_posterior_unavailable'`: evidence and
  `p_zmatch_given_qso` are returned, `p_sameq` and `log_r_per_unit_z` are NaN —
  by design.

The full output contract is in the method note, §9, and `AGENTS.md` §6.

---

## Reproducing the model

The saved model needs no database. Rebuilding it needs WSDB (`sqlutilpy`,
credentials via `PGUSER` / `PGHOST` / `~/.pgpass`) and the DESI/SDSS parent
caches named in the configuration:

```bash
pip install -e ".[dev,wsdb]"
```
 Queries cache to `data/` (gitignored),
keyed by a hash of the query text, so a rerun reuses the same selection.

```bash
C=configs/multisurvey_lsw.json                      # the training config recorded in the model
python scripts/build_multisurvey_sample.py --config $C          # quasars and 24 field cones
python scripts/build_multisurvey_sample.py --config $C --part validation   # reserved overlap field
python scripts/train_multisurvey_model.py  --config $C          # -> models/multisurvey_lsw.json, ~1 h
python scripts/fit_background_marginal.py  --config configs/background_marginal_lsw_grz.json
python scripts/fit_background_marginal.py  --config configs/background_marginal_lsw_grzw.json
python scripts/build_multisurvey_priors.py --config configs/multisurvey_priors.json
python scripts/fit_multisurvey_outlier.py  --config $C --model models/multisurvey_lsw.json \
       --out models/multisurvey_lsw_outlier.json
python scripts/validate_multisurvey.py     --model models/multisurvey_lsw.json   # 127 subsets
python scripts/validate_pairs_multisurvey.py --max-fracflux 0.2 \
       --ms-model models/multisurvey_lsw.json --ms-outlier models/multisurvey_lsw_outlier.json
```

Training writes a candidate; it becomes `models/multisurvey.json` only after it
passes the validation. Figures and reports, all offline once the samples exist:

```bash
python scripts/make_validation_figures.py       # pair validation and outlier figures
python scripts/plot_colour_redshift.py          # quasar colour density vs redshift
python scripts/make_multisurvey_examples.py     # twelve survey-combination examples + atlas
make -C docs/method                              # the method note
```

## What is in here

| path | what |
|---|---|
| `src/qso_pcolor/` | the package: mixtures, extreme deconvolution, the multi-survey model and priors, the unmodelled term, the scorer |
| `scripts/build_multisurvey_sample.py` | quasar training sample and field cones, seven surveys |
| `scripts/train_multisurvey_model.py` | quasar slices and joint field fit, with spatial component selection |
| `scripts/fit_background_marginal.py` | dedicated field fits for Legacy-only input |
| `scripts/build_multisurvey_priors.py` | surface densities per reference band |
| `scripts/fit_multisurvey_outlier.py` | the unmodelled term: breadth and share, on unused field sources |
| `scripts/validate_pairs_multisurvey.py` | the labelled-pair validation, against the predecessor on the same rows |
| `scripts/validate_multisurvey.py` | all 127 survey subsets on reserved objects |
| `scripts/build_pair_validation.py` | the labelled close-pair sample from DESI DR1 |
| `docs/method/method.pdf` | **the method note** |
| `docs/MULTISURVEY_VALIDATION.md` | the survey-subset results |
| `docs/examples/` | the twelve worked examples, atlas, fixed sample and scores |
| `models/archive/` | retired models (below) |
| `tools/journal.py` | JOURNAL.md updater; `hook-install` journals every commit |

`python scripts/<name>.py --help` for options.

### The archive

`models/archive/` keeps what the model replaced, because its provenance and
some reports depend on it:

- `original_legacy_south/` — the Legacy-only relative-flux model, its field
  model, both surface densities and its outlier term. The model's reserved sky
  blocks and its prior normalisation come from here. Its validated numbers are
  the benchmark in the method note's appendix, pinned by
  `tests/test_archived_original.py`. The scripts that built and studied it
  (`train_qso_model.py`, `build_global_background.py`, `validate_pairs.py`,
  `compare_background_modes.py`, `make_method_figures.py`, …) still run against
  these paths.
- `multisurvey_37band_20260921.json` and its priors — the multi-survey model
  before the Legacy forced *W1, W2* were added.
- `multisurvey_joint_20260921.json` — its initial joint-only version.

Old training configs are in `configs/archive/`.

## Conventions

- Figures are PNG under `plots/`, written with `qso_pcolor.plotting.save_figure`.
- `JOURNAL.md` is local and gitignored; `python tools/journal.py hook-install`
  makes every commit journal itself.
- `AGENTS.md` is the working contract — read it before changing anything.

## Citing

Queries use q3c (Koposov & Bartunov 2006,
<https://ui.adsabs.harvard.edu/abs/2006ASPC..351..735K>). Training data are DESI
DR1 and SDSS DR16Q (Lyke et al. 2020,
<https://arxiv.org/abs/2007.09001>); the method is extreme deconvolution (Bovy,
Hogg & Roweis 2011, <https://arxiv.org/abs/0905.2979>). The method note's
bibliography has the rest.

**No licence file yet** — add one before sharing outside the group.

## Acknowledgements

The code, the method note and the analyses in this repository were written with
**Claude** (Anthropic) in Claude Code, working from the scientific design and
under the review of the author. **OpenAI Codex** was used as an independent
reviewer of the pipeline and the write-up.

Responsibility for the method and for what is claimed of it rests with the
author.
