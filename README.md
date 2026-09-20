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
(quadrature, Monte Carlo, closed forms) by 91 tests. The quasar colour model,
trained on 1.24 M DESI DR1 quasars with a reserved 23.5 % spatial holdout. The
prior-independent Bayes factor.

**What is not.** There is **no calibration**. The labelled close-pair sample
(`scripts/build_pair_validation.py`) has never been run, so no reliability
curve, precision–recall or completeness figure exists. Rank candidates by
`log_r_per_unit_z` or `log_bayes_factor_qz_bkg`; do not read `p_sameq` as a
calibrated probability.

**Known open issues**, all recorded in `AGENTS.md`:

- `tune_shrinkage` cannot see its own parameter (it holds out at the parent
  HEALPix resolution), so the background pooling constant is fixed, not tuned.
- The selection-bias measurement is unresolved: DESI- and SDSS-selected quasars
  cannot be compared by raw density, because they differ by 0.7 mag in
  brightness, which changes the density mechanically.
- Only the southern photometric system (`release` 9010) is trained. North
  (BASS/MzLS) is a different system and needs its own model.
- Blends are out of scope: `BlendPolicy` refuses companions below a stated
  separation rather than scoring them badly.

---

## Install

```bash
source ~/Work/venvs/.venv/bin/activate      # or your own environment
pip install -e .
python -m pytest -q                          # 91 tests, ~40 s
```

Python ≥ 3.11 with numpy, scipy, astropy, healpy, matplotlib. Add the `wsdb`
extra for `sqlutilpy` if you want to pull data yourself.

---

## Scoring a candidate

```python
import numpy as np
from qso_pcolor.background import fit_local_background
from qso_pcolor.features import RelativeFluxTransform, deredden
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
from qso_pcolor.score import BlendPolicy, score_candidates

BANDS = ("g", "r", "z", "w1", "w2")
qso = SlicedColourRedshiftModel.load("models/qso_south_full.json")
tr = RelativeFluxTransform(reference_band="r")

# Legacy Surveys fluxes, inverse variances and transmissions (nanomaggies)
flux  = np.array([[1.9, 2.6, 3.1, 11.0, 14.0]])
ivar  = np.array([[120.0, 150.0, 60.0, 8.0, 3.0]])
trans = np.array([[0.97, 0.98, 0.99, 1.0, 1.0]])
f, v = deredden(flux, ivar, trans)
feat = tr(f, v, BANDS)

# the field population in the candidate's OWN neighbourhood
bkg, dens, info = fit_local_background(
    ra=180.0, dec=0.0, radius_deg=0.5, transform=tr, bands=BANDS,
    mag_edges=np.array([17.0, 19.5, 20.5, 21.5, 22.5]),
    system=qso.system, max_ref_mag=22.5, seed=0, max_iter=200,
)

rows = score_candidates(
    feat, z_primary=np.array([1.8]),
    l_deg=np.array([120.0]), b_deg=np.array([60.0]),
    qso_model=qso, background_model=bkg, background_density=dens,
    match=RedshiftMatch(half_width_kms=2000.0),
    blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
    separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]),
)
s = rows[0]
print(s.log_bayes_factor_qz_bkg)   # +4.91
print(s.status)                    # 'no_prior_posterior_unavailable'
print(s.p_sameq)                   # nan — see below
```

`p_sameq` is `nan` here **by design**: no `qso_prior` was supplied, so the
package returns the prior-independent evidence rather than inventing a
posterior. Pass a `GridQSOPrior` to get one — `scripts/score_examples.py` builds
one the recommended way (global and isotropic, normalised to the observed
coverage plateau).

Every row also carries `log_r_per_unit_z` (the ranking statistic),
`dz_match_eff`, the three log intensities, an out-of-distribution score, quality
flags and a status code. §10 of the method note lists them all.

### Things that will bite you

- **Mixing photometric systems.** North and south Legacy Surveys are different
  systems; the scorer raises rather than silently combining them.
- **Reading `p_sameq` as "probability of a binary".** A ±2000 km/s window is
  Δ*z* = 0.037 against a photometric redshift width of ~0.6, so `p_sameq` stays
  small even for a perfect candidate. Rank on `log_r_per_unit_z`.
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
| `scripts/build_pair_validation.py` | labelled close-pair sample from DESI DR1 — **not yet run** |
| `scripts/make_method_figures.py` | the method note's figures |
| `scripts/check_em_convergence.py` | does the EM iteration cap matter? (measured: no) |
| `scripts/smoke_real_data.py` | end-to-end plumbing check against live WSDB |
| `tools/journal.py` | JOURNAL.md updater; `hook-install` journals every commit |
| `docs/method/method.pdf` | **the method note** — formalism, derivations, 9 figures |
| `docs/REVIEW_OF_PLAN.md` | why the design departs from the original plan |
| `configs/example_ls_dr9.yaml` | every tunable, with nothing defaulted that changes meaning |

`python scripts/<name>.py --help` for options.

## Models and data

`models/qso_south_full.json` is committed: 1.24 M DESI DR1 quasars, Legacy
Surveys DR9 south, 32 redshift slices × 20 components, 23.5 % of sky blocks
reserved before fitting. Enough to score candidates without retraining.

Everything else is derived and gitignored. Rebuilding needs **WSDB access**
(`sqlutilpy`, credentials via `PGUSER` / `PGHOST` / `~/.pgpass`):

```bash
python scripts/train_qso_model.py --system south --select-k   # ~4 h
python scripts/make_method_figures.py                          # ~8 min
make -C docs/method                                            # rebuild the PDF
```

Queries cache to `data/`, keyed by a hash of the query text, so a rerun costs
nothing and a changed query can never return stale rows.

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
