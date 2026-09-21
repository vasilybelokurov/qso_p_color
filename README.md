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

## State of play — read this before trusting a number

**What is solid.** The statistical machinery, checked against independent routes
(quadrature, Monte Carlo, closed forms) by 117 tests. The quasar colour model,
trained on 1,106,986 spectroscopic quasars — 917,489 DESI DR1 and 189,497 SDSS
DR16Q, all with `maskbits = 0` — with 20 % of nside=4 sky blocks reserved before
fitting. The
prior-independent Bayes factor.

**What is measured** (`scripts/validate_pairs.py`, 52,099 spectroscopically
labelled companions at 3–30″, held-out and full samples agree to 0.01):

| question | result |
|---|---|
| same-*z* quasar vs wrong-*z* quasar, ranked by `log_r_per_unit_z` | **AUC 0.81** (0.86 by `p_zmatch_given_qso` alone) |
| same, ranked by the Bayes factor alone | AUC 0.76 |
| quasar vs spectroscopic star, by Bayes factor | **AUC 0.98**; 0.4 % of stars above the median same-*z* quasar |
| quasar vs spectroscopic galaxy, by Bayes factor | **AUC 0.96**; 0.3 % above |
| `p_zmatch_given_qso` calibration | right in shape, **low by 3.3× (20–30″) to 9.9× (3–5″)** |

That last row is not a bug: the scorer assumes the companion's redshift is drawn
from the field, and physical pairs cluster. The factor is the measured
clustering excess; multiply the odds by it if you want a probability at a given
separation. Rank on `log_r_per_unit_z`; do not read `p_sameq` as calibrated
without that factor.

`log_bayes_factor_qz_bkg` is **not** an alternative ranking statistic. It
compares "a quasar at *z*₀" against "background" and has no `field_q` term.
Measured: it separates same-*z* from wrong-*z* quasars with AUC 0.76 against
0.81 for `log_r_per_unit_z` — worse, not useless, because p(c | Q, *z*₀) is
itself redshift-dependent. Use it to reject stars, not to order candidates.
Ranking needs a prior; without one the package returns NaN for
`log_r_per_unit_z` rather than substituting something that looks similar.

**Known open issues**, all recorded in `AGENTS.md`:

- `tune_shrinkage` cannot see its own parameter (it holds out at the parent
  HEALPix resolution), so the background pooling constant is fixed, not tuned.
- The selection-bias measurement is unresolved: DESI- and SDSS-selected quasars
  cannot be compared by raw density, because they differ by 0.7 mag in
  brightness, which changes the density mechanically.
- Only the southern photometric system (`release` 9010) is trained. North
  (BASS/MzLS) is a different system and needs its own model.
- Blends are out of scope: `BlendPolicy` refuses companions below a stated
  separation rather than scoring them badly — and only when you pass one.
- `p_zmatch_given_qso` normalises over a redshift grid wider than the model's
  support (0.05–5.0 against 0.15–4.35), where the edge slices repeat. Measured:
  2.25 % of that normalisation is extrapolated for the example below — down
  from 7.1 % before the range was widened, but not zero. The Bayes factor is
  unaffected.
- Only the southern model is validated end to end. `p_zmatch_given_qso` is
  conditional on the companion being a quasar *inside the trained range*; a
  candidate whose colours are best explained beyond z ≈ 4.4 is reported via
  `frac_norm_outside_support`, not scored as if it were inside.

---

## Install

```bash
source ~/Work/venvs/.venv/bin/activate      # or your own environment
pip install -e ".[dev,wsdb]"                 # dev = pytest, wsdb = sqlutilpy
python -m pytest -q                          # 117 tests, ~45 s
```

Python ≥ 3.11 with numpy, scipy, astropy, healpy, matplotlib. `pip install -e .`
alone installs neither pytest nor `sqlutilpy`, so use the extras above: the test
command needs the first and everything that touches data needs the second.

**The worked example below queries WSDB live.** It needs `sqlutilpy` and
credentials (`PGUSER` / `PGHOST` / `~/.pgpass`); without them, only the parts
that use the committed model will run.

---

## Scoring a candidate — offline, straight from a clone

Everything the scorer needs ships in `models/`: the quasar colour model, a
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
- **Ranking on the Bayes factor.** See above: AUC 0.76 against 0.81.
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

Everything else is derived and gitignored. Rebuilding needs **WSDB access**
(`sqlutilpy`, credentials via `PGUSER` / `PGHOST` / `~/.pgpass`):

```bash
python scripts/train_qso_model.py --system south --select-k   # ~4 h
python scripts/make_method_figures.py                          # figs 1-8, ~8 min
python scripts/score_examples.py                               # fig 9, ~15 min
make -C docs/method                                            # rebuild the PDF
```

Queries cache to `data/`, keyed by a hash of the query text, so a rerun costs
nothing and a changed query can never return stale rows. Two caches sit outside
that guarantee: `data/dr16q_ls.npz` (written by `train_qso_model.load_sdss`,
which checks only that the file exists, so changed redshift limits reuse the old
sample) and the fitted `models/method_*.json`, which reload unless `--refit` is
passed.

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
