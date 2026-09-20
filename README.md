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
(quadrature, Monte Carlo, closed forms) by 101 tests. The quasar colour model,
trained on 1,116,464 spectroscopic quasars — 931,563 DESI DR1 and 184,901 SDSS
DR16Q — with 20 % of nside=4 sky blocks reserved before fitting. The
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
  separation rather than scoring them badly — and only when you pass one.
- `p_zmatch_given_qso` normalises over a redshift grid wider than the model's
  support (0.05–5.0 against 0.15–4.35), where the edge slices repeat. Measured:
  2.25 % of that normalisation is extrapolated for the example below — down
  from 7.1 % before the range was widened, but not zero. The Bayes factor is
  unaffected.
- The training sample's quality cuts differ by channel: the SDSS third requires
  `maskbits = 0`, the DESI two-thirds does not. And `--select-k` picks K from a
  redshift-pooled *unconditional* mixture, then uses it in all 32 conditional
  slices. Both need a retrain to fix.

---

## Install

```bash
source ~/Work/venvs/.venv/bin/activate      # or your own environment
pip install -e ".[dev,wsdb]"                 # dev = pytest, wsdb = sqlutilpy
python -m pytest -q                          # 101 tests, ~45 s
```

Python ≥ 3.11 with numpy, scipy, astropy, healpy, matplotlib. `pip install -e .`
alone installs neither pytest nor `sqlutilpy`, so use the extras above: the test
command needs the first and everything that touches data needs the second.

**The worked example below queries WSDB live.** It needs `sqlutilpy` and
credentials (`PGUSER` / `PGHOST` / `~/.pgpass`); without them, only the parts
that use the committed model will run.

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
    l_deg=np.array([276.337]), b_deg=np.array([60.189]),   # (RA,Dec)=(180,0)
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

`models/qso_south_full.json` is committed: 1,116,464 training quasars
(931,563 DESI DR1 + 184,901 SDSS DR16Q, de-duplicated at 1″), Legacy Surveys
DR9 south (`release` 9010), 43 redshift slices covering 0.15 < z < 4.35, with K
chosen per slice (20 in the well-populated core, falling to 4 in the sparsest
high-redshift slice). 16 of the 81
populated nside=4 sky blocks were reserved before fitting, holding out 343,704
objects (23.5 % of the sample; the 20 % is a fraction of *blocks*, not of
objects). Those 16 block IDs and the seed are recorded in the file's `meta`, so
downstream code reads the split instead of re-deriving it — re-deriving it is
what produced a figure caption claiming five held-out quasars when three of
them were in the fit. Enough to score candidates without retraining.

The range was widened from the original 0.4–3.6 by *appending* 11 slices
(`scripts/extend_qso_model_redshift.py`), which leaves the original 32 mixtures
bit-identical and reuses the recorded holdout. Measured on 15,552 held-out
quasars outside the old range, log p(c | Q, z_spec) improves by **+1.90 nats**
on average (78 % of objects), rising from +0.1 nats just beyond the old edge to
+5.0 at z ≈ 4.3 — the shape you expect if a real deficiency is being repaired
rather than noise absorbed. `scripts/check_redshift_extension.py` reruns that
comparison.

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

The committed `plots/examples/optical_only_examples.png` predates the holdout
fix, so `score_examples.py` now selects a different five quasars than the ones
in it; §9 of the method note says so explicitly.

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
