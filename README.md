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

## The model

There is one model. It accepts **any combination of 41 bands** from seven
surveys, including infrared-only input:

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

**How well it works**, measured on 50,746 companions of DESI DR1 quasars that
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
python -m pytest -q                          # 185 tests, ~1 min
```

The `wsdb` extra (`sqlutilpy`) is needed only to query the database and
rebuild samples; the `figures` extra only for the example atlas PDF. Scoring
with the saved model needs neither. Run from the repository root so the
`models/` paths resolve; the model files are in Git, not in the wheel.

## Scoring a candidate

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
print(s.log_bayes_factor_qz_bkg)   # +3.91    evidence: quasar at z0 vs the field
print(s.log_r_per_unit_z)          # -1.73    the ranking statistic
print(s.p_sameq, s.dz_match_eff)   # 6.6e-03  posterior for the ±2000 km/s window, and its width
print(s.p_outlier)                 # 1.4e-03  share taken by "unmodelled"
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
