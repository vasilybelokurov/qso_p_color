# qso_pcolor

Is the photometric companion of a known quasar itself a quasar at the same
redshift?

Given a quasar with a spectroscopic redshift *z*₀ and a companion a few
arcseconds away with broadband photometry only, this package computes the
colour evidence for three competing hypotheses and turns them into a posterior:

| hypothesis | meaning |
|---|---|
| `same_z` | a quasar whose redshift matches *z*₀ |
| `field_q` | a quasar at some other redshift |
| `bkg` | anything else in the imaging catalogue at that brightness and position |

The third column of that table is the reason the package exists in this shape.
A quasar-versus-star calculation cannot tell a genuine companion from a
foreground quasar at *z* = 2.6, and for a binary-quasar search that is the
failure mode that matters.

## What it computes

For colour/shape features **c**, reference magnitude *m*, and Galactic
position (*l*, *b*):

```
λ_sameQ  = ∫ W(z|z₀) Σ_Q(z,m) p(c | Q,z) dz
λ_fieldQ = ∫ [1−W(z|z₀)] Σ_Q(z,m) p(c | Q,z) dz
λ_bkg    = Σ_B(m,l,b) p(c | B,m,l,b)

p_sameq  = λ_sameQ / (λ_sameQ + λ_fieldQ + λ_bkg)
```

and, separately from any prior, the Bayes factor
`p(c | Q, z₀) / p(c | B, m, l, b)`.

`p(c | Q, z)` is a set of extreme-deconvolution Gaussian mixtures conditional on
redshift; `p(c | B, m, l, b)` is a hierarchy of mixtures over HEALPix cells and
magnitude bins, pooled towards coarser cells where the data are sparse. Both
convolve with each object's own covariance and marginalise exactly over bands
the survey could not measure.

## The honest caveat, up front

A ±2000 km s⁻¹ window at *z* = 1.4 is Δ*z* = 0.016. Broadband colours constrain
a quasar redshift to σ_z ≈ 0.1–0.3. So `p_sameq` under a velocity window is
small even for a candidate sitting exactly on the locus — most of the quasar
intensity falls outside the window, and `field_q` takes it.

The package therefore reports a **ranking statistic and a prior-independent
Bayes factor**, not "the probability this is a binary quasar". Spectroscopy
answers that question; colours decide which candidates are worth the fibre.

## Install and test

```bash
source ~/Work/venvs/.venv/bin/activate
cd ~/Work/Code/qso_p_color
pip install -e . --no-deps
python -m pytest -q
```

## Layout

```
src/qso_pcolor/
  gaussmix.py    Gaussian-mixture algebra: batched log-densities, exact
                 missing-dimension marginalisation, per-object noise, conditioning
  xd.py          extreme deconvolution (Bovy, Hogg & Roweis 2011) in numpy
  features.py    fluxes -> features with a full covariance matrix
  qso_model.py   p(colour | quasar, z); redshift match definitions
  background.py  p(colour | background, m, l, b), hierarchical over the sky
  priors.py      Σ_B(m,l,b) and Σ_Q(z,m)
  score.py       the three-hypothesis scorer
  data.py        WSDB queries, cached to .npz
  plotting.py    save_figure: every figure a PNG under plots/
plots/           all figures, PNG only
scripts/         validation-sample builder, smoke test, method figures
docs/method/     the method note (LaTeX -> PDF)
tools/journal.py JOURNAL.md updater
docs/            review of the original plan
```

## Where to read next

- **`docs/method/method.pdf`** — the method note: the full derivation, the
  three-hypothesis formalism, the window factorisation, and seven figures built
  from real DESI/Legacy data. Build it with `make -C docs/method` after running
  `python scripts/make_method_figures.py`.
- **`AGENTS.md`** — the working contract: rules, milestones, output contract.
  Read this before changing anything.
- **`docs/REVIEW_OF_PLAN.md`** — why the implementation departs from
  `qso_binary_color_probability_plan.md` where it does, with the measurements
  behind each departure.
- **`JOURNAL.md`** — what has been done, with the state of the tests at the
  time. Local only, not in git: run `python tools/journal.py hook-install` in a
  fresh clone and it starts filling itself on every commit.

## Data

Trained on WSDB: DESI DR1 quasars (`desi_dr1.zpix` joined to
`desi_dr1.photometry` on `targetid`, so no positional crossmatch) and Legacy
Surveys imaging (`decals_dr9.main`, `decals_dr11.main`). SDSS DR16Q
(`sdssdr16qso.main`) provides an independently selected check sample.

Queries use q3c (Koposov & Bartunov 2006,
<https://ui.adsabs.harvard.edu/abs/2006ASPC..351..735K>), which should be cited
in any resulting publication.
