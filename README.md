# qso_pcolor

Photometric classification of the companion of a spectroscopic quasar. The primary has a known
redshift *z*₀; for the companion we have only broadband photometry.

- **Resolved companion** (detected and measured as its own source): is it a quasar, and is it a quasar
  at the primary's redshift?
- **Unresolved blend** (primary and companion measured as one source): is the companion a second quasar
  at *z*₀ (QQ), or a star (QS)?

The method is written up in [docs/method_unified/method_unified.pdf](docs/method_unified/method_unified.pdf)
(built with `make -C docs/method_unified`). `AGENTS.md` is the working contract for changing the code.

---

## Which model to use

There are three current models. Two are for resolved companions and share one interface; the third is
for blends and is built on top of either of them.

| Situation | Model | Load with |
|---|---|---|
| Resolved companion, PSF photometry | **A**: magnitude-dependent quasar colours | `UnifiedPSFModel.load("models/multisurvey_psf/current")` |
| Resolved companion, PSF photometry | **B**: magnitude-independent quasar colours (XDQSO assumption) | `UnifiedPSFModel.load("models/multisurvey_psf/current_magindep")` |
| Unresolved blend of the quasar and one companion | **C**: blend mode, QQ (same *z*) vs QS | `BlendModel.load(...)` on the `current` or `current_magindep` pointer |

**Score every object with both A and B** (for C, load it on both pointers). A and B share the
background model, the 41 bands, extinction correction, catch-all, support cut and faint limit; they
differ only in the quasar colour model.

- On our DESI/SDSS-selected test quasars the two rank almost equally well: AUC 0.9939 for A and 0.9934 for B,
  a marginal difference in a paired sky bootstrap. A gives `p_quasar` > 0.5 to 2.7 percentage points more test
  quasars (94.0% against 91.2%), and to 0.11 points more unflagged background objects. B ranks slightly better
  at 18 ≤ *r* < 21 and A at *r* ≥ 21 (method note, Section 6.2).
- B cannot absorb the spectroscopic selection of the training quasars into its colour model. That risk
  is real for quasars unlike the DESI/SDSS targets, and our tests cannot measure it.
- If A and B agree, the result does not depend on that assumption. If they disagree strongly, the
  object needs a closer look, not a choice of whichever number is more convenient.

Pointers in `models/multisurvey_psf/` (small JSON files naming a bundle directory):

| Pointer | Bundle | Notes |
|---|---|---|
| `current` (= `current_magdep`) | `20261005_magdep_qlf` | model A, promoted 5 October 2026 (eBOSS luminosity-function quasar abundance) |
| `current_magindep` | `20261005_magindep_qlf` | model B, same date |
| `previous_20261003_magdep`, `previous_20261003_magindep` | `20261003_magdep`, `20261003_magindep` | rollback: the 3 October bundles (spectroscopic-count abundance) |
| `previous_20261002_magdep`, `previous_20261002_magindep` | `5f4002492dbb849c`, `13866e45ef794059` | rollback |
| `previous` | `630f47f63b6f0694` | 28 September 2026 model, see [docs/EARLIER_MODELS.md](docs/EARLIER_MODELS.md) |

---

## Install

```bash
git clone https://github.com/vasilybelokurov/qso_p_color.git
cd qso_p_color
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11 or newer
pip install -e ".[dev]"
python -m pytest -q
pip install -e ".[dev,wsdb]"                              # only to query WSDB (sqlutilpy)
```

Run from the repository root: the model files are in Git under `models/`, not in the wheel.

---

## Quick start

### Resolved companion (models A and B)

```python
import numpy as np
from qso_pcolor import Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.unified import UnifiedPSFModel

bands = ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z",
         "decals_dr9_south:w1", "decals_dr9_south:w2")
# Catalogue fluxes as published (nanomaggies, NOT extinction-corrected) and their variances.
flux = np.array([[2.6, 3.0, 3.6, 9.0, 11.0],          # a quasar-like companion
                 [1.2, 3.0, 4.6, 2.2, 1.3]])          # a star-like companion
ivar = np.array([[300., 200., 60., 3., 0.6]] * 2)
phot = Photometry(flux, 1 / ivar, bands)

for name in ("current", "current_magindep"):
    model = UnifiedPSFModel.load(f"models/multisurvey_psf/{name}")
    scores, decision = model.score(
        phot, ra_deg=[180., 180.], dec_deg=[30., 30.],    # companion ICRS position
        z_primary=1.8, morphology=["PSF", "PSF"],          # Legacy TYPE of the companion
        match=RedshiftMatch(half_width_kms=2000.),
        blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
        separation_arcsec=[6., 6.], fracflux=[.05, .05],   # from the catalogue
        ood_flag_sigma=4.)
    p = model.quasar_probability(scores)
    for i, s in enumerate(scores):
        print(name, decision["eligible"][i], decision["reason"][i] or s.status,
              round(p[i], 3), s.log_r_per_unit_z, s.p_sameq, s.dz_match_eff)
```

Output: the first object is eligible with `p_quasar` 0.866 (A) and 0.832 (B); the second is refused
with `qso_support_rejected` (its colours are not quasar-like at *z* = 1.8).

### Unresolved blend (model C)

```python
import numpy as np
from qso_pcolor import Photometry
from qso_pcolor.blend import BlendModel

bands = ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z",
         "decals_dr9_south:w1", "decals_dr9_south:w2")
# Whole-blend fluxes as published (nanomaggies, NOT extinction-corrected): the best total flux of the
# combined source (see "Inputs" below), and their variances.
phot = Photometry([[4.1, 5.5, 6.9, 13.0, 14.5]], [[1/300, 1/200, 1/60, 1/3, 1/0.6]], bands)

for name in ("current", "current_magindep"):
    blend = BlendModel.load(f"models/multisurvey_psf/{name}")
    r = blend.score(phot, ra_deg=150.1, dec_deg=2.2, z_qso=1.5, prior_odds_qq=1.0)[0]
    print(name, r.status, r.log_bf_qq_qs, r.p_qq, r.alpha_mean_qq, r.alpha_mean_qs)
```

Output: `ok`, log BF −0.59 (A) and −0.49 (B), P(QQ) 0.36 and 0.38; under QS the star carries about 30% of
the r flux.

---

## Inputs

### Photometry contract (all three models)

- **A `Photometry(flux, variance, bands)`**, arrays of shape (objects, bands). Band labels are
  `survey:band`; `model.base.model.transform.bands` lists all 41. Any subset may be given, but
  **Legacy r must be present** (`decals_dr9_south:r` or `decals_dr9_north:r`): it is the reference band
  and sets the faint limit.
- **Native units**: in every band, magnitude 22.5 corresponds to flux 1, in that survey's own
  photometric system. Legacy, SDSS, PS1, NSC and SkyMapper are AB (Legacy and SDSS fluxes are already
  nanomaggies); AllWISE and VHS are Vega. Surveys are never converted into each other's systems.
- **Not extinction-corrected.** Pass catalogue photometry as published. The models correct Galactic
  extinction internally (SFD98 with each band's coefficient), from the position you give. Do not
  deredden first. (`BlendModel.score` has `dereddened=True` for already-corrected input.)
- **Legacy north and south are different bands.** `decals_dr9_south:*` for `release` 9010 and 9012,
  `decals_dr9_north:*` for 9011. Legacy W1/W2 (forced unWISE photometry at the optical position) and
  AllWISE W1/W2 are also different bands.
- **Keep negative and low-S/N fluxes**: they are measurements. A missing or unusable band is
  `flux = NaN`, `variance = inf`.
- **Magnitudes** can be passed through
  `qso_pcolor.unified.catalogue_photometry(values, errors, bands, ra_deg=, dec_deg=, measurement="abmag")`.
  It converts with flux = 10^((22.5 − m)/2.5), so give magnitudes in each band's native system. For AllWISE
  and VHS pass the Vega magnitudes unchanged: the model's native flux in those bands is on the Vega scale,
  so no Vega-to-AB conversion is wanted (for AllWISE this equals the DN conversion in the table below). The label `legacy:g` (r, z, w1, w2) is assigned north or south from the position
  unless you pass `legacy_hemisphere`. `UnifiedPSFModel.score_catalogue(...)` does the conversion and the
  scoring in one call.

### Which catalogue columns: resolved companions (models A and B)

The models were trained on **PSF photometry** of point sources, so use the PSF columns below. All surveys
are matched to the Legacy DR9 position within 1″ (AllWISE 2″). The training applied the cleaning listed
in the last column; apply the same to candidates. `qso_pcolor.multisurvey_data.catalogue_photometry(name,
rows, clean=True, vhs_bad_bits=2147483392)` converts WSDB rows exactly as in training.

| Survey (WSDB table) | Labels | Flux / error columns | Units | Cleaning (training) |
|---|---|---|---|---|
| Legacy DR9 (`decals_dr9.main`) | `decals_dr9_{south,north}:g r z w1 w2` | `flux_*`, `flux_ivar_*` | nanomaggies (AB) | `maskbits` = 0, `nobs_*` > 0; `type` = PSF for the resolved models |
| SDSS DR14 (`sdssdr14.photoobjall`) | `sdss:u g r i z` | `psfflux_*`, `psffluxivar_*` | nanomaggies | `mode` = 1, `clean` = 1 |
| AllWISE (`allwise.main`) | `allwise:w1 w2 w3 w4` | `w?flux`, `w?sigflux` (DN) | flux = DN × 10^(0.4(22.5 − ZP)), ZP = 20.5, 19.5, 18.0, 13.0 (Vega) | `cc_flags` character for the band = '0' |
| Pan-STARRS1 (`panstarrs_dr1.stackobjectthin`) | `ps1:g r i z y` | `?psfmag`, `?psfmagerr` | AB mag | `primarydetection` = 1, `?nframes` > 0 |
| NSC DR2 (`nsc_dr2.object`) | `nsc:u g r i z y vr` | `?mag`, `?err` | AB mag | `flags` = 0 |
| SkyMapper DR4 (`skymapper_dr4.main`) | `skymapper:u v g r i z` | `?_psf`, `e_?_psf` | AB mag | `?_ngood` > 0, `?_flags` = 0 |
| VHS DR5 (`vhs_dr5.main`) | `vhs:y j h ks` | `?apermag3`, `?apermag3err` | Vega mag | `?pperrbits` ≥ 0 with none of bits 8–30 set |

Magnitudes outside (−50, 90) or with errors ≥ 90 are treated as missing. AllWISE photometry next to a
bright primary is often contaminated: mark such bands unusable rather than scoring them.

### Which catalogue columns: unresolved blends (model C)

Use the **best total flux of the whole blended source**, from any morphology: Legacy DR9 `flux_*`
(Tractor model flux, whatever `type`, including the forced `flux_w1`, `flux_w2`), SDSS `cmodelflux_*`,
PS1 `?kronmag`, and AllWISE `w?flux` (the WISE beam contains both components). This choice is advice,
not enforced by the code: `BlendModel` adds the two components' fluxes in every band it is given, so
each band must measure the total light of both components. **Do not use PSF fluxes of an extended
blend**: they miss part of the light, by an amount that depends on the separation and the seeing.

Caveat: the component models were fitted to PSF photometry of single sources. For a blend smaller than
the seeing, the total flux and the sum of the components' PSF fluxes should agree, but this is untested
on real blends.

---

## Resolved companions: arguments and output

`UnifiedPSFModel.score(photometry, *, ra_deg, dec_deg, z_primary, morphology, match, blend_policy,
ood_flag_sigma, separation_arcsec, fracflux, ...)` returns `(scores, decision)`. Arrays broadcast over
objects.

| Argument | Meaning |
|---|---|
| `ra_deg`, `dec_deg` | companion position (ICRS, degrees); used for extinction, Galactic latitude and sky-dependent background |
| `z_primary` | spectroscopic redshift of the primary quasar (support 0.15–4.35) |
| `morphology` | Legacy `type` of the companion. Only `PSF` is scored; `REX`, `EXP`, `DEV`, `SER` and unknown are refused |
| `match` | `RedshiftMatch(half_width_kms=...)`: the window that counts as "the same redshift" |
| `blend_policy`, `separation_arcsec`, `fracflux` | `BlendPolicy(min_separation_arcsec, max_fracflux)`: companions closer than the separation limit, or with a reference-band Legacy `fracflux` above the limit, are refused. Both limits are required; no defaults. For blends, use model C |
| `ood_calibration` | optional; default `DEFAULT_OOD_CALIBRATION` = `dict(alpha=2/256, draws=256, seed=20261005)`: the calibrated outside-both-models test (below). `None` switches back to the fixed `ood_flag_sigma` cut |
| `ood_flag_sigma` | required by the interface; used for the outside-both-models decision only when `ood_calibration=None` (earlier rule, 4 recommended) |

**`decision`**: `eligible` (bool) and `reason` per object; `qso_support` (the QSO-support percentile).
Only eligible rows are science-grade. Rows refused before scoring have `scores[i] = None`.

| `reason` / `status` | Meaning |
|---|---|
| `non_psf_not_scored` | morphology is not PSF |
| `outside_latitude` | Galactic \|*b*\| < 25° (outside the training footprint) |
| `no_faint_limit_band` | no Legacy r |
| `too_faint` | Legacy r S/N < 10 |
| `blended_not_scored` | fails the `BlendPolicy` |
| `insufficient_photometry` | too few usable bands |
| `primary_z_outside_model_support` | *z*₀ outside the quasar model's range (0.15–4.35) |
| `outside_both_models` | colours unlike both quasars and background: calibrated tail probabilities `qso_ood_p` and `bkg_ood_p` both below 2/256 (see below) |
| `background_out_of_mag_range` | reference magnitude outside the background model's magnitude range |
| `qso_support_rejected` | colours far outside the quasar distribution at *z*₀ (support percentile below the calibrated 2/256) |
| `no_prior_posterior_unavailable` | no population prior for the reference band: evidence is returned, `p_sameq` and `log_r_per_unit_z` are NaN |

**Score fields** (eligible rows):

- `model.quasar_probability(scores)` gives *p*(quasar), the companion's probability of being a quasar at
  any redshift against the background and catch-all, at the population priors for its position and
  magnitude. Neither promoted bundle carries a calibration file, so this is the uncalibrated value.
- `log_r_per_unit_z` is the ranking statistic for "a quasar at *z*₀", per unit redshift, averaged over the
  window. For velocity windows it is independent of the window width to well under 1%, so scores made
  with different velocity windows can be compared. Rank candidates on it.
- `p_sameq` is the posterior probability of a quasar inside the `match` window, with
  `p_sameq = exp(log_r_per_unit_z) × dz_match_eff`. Always quote it together with `dz_match_eff`: it is
  small even for an ideal candidate, because the window (Δ*z* = 0.037 for ±2000 km/s at *z* = 1.8) is
  much narrower than the photometric redshift width.
- `p_sameq` is not the probability of a physical pair. Quasar clustering is deliberately left out of the
  model.
- `log_bayes_factor_qz_bkg` (quasar at *z*₀ against the background)
  is not an alternative ranking statistic: it has no term for quasars at other redshifts, so it cannot order same-*z* against wrong-*z*
  quasars. Use it to reject stars.
- Diagnostics: `p_zmatch_given_qso`, `qso_ood_sigma_any_z`, `bkg_ood_sigma`, `qso_ood_p`, `bkg_ood_p`,
  `reference_band`, `bands_used`, `quality_flags`, and `config_hash` with `model_manifest_id` for
  provenance.

### How many quasars there are: the abundance prior (since 5 October 2026)

`p_quasar` and `log_r_per_unit_z` weigh each hypothesis by how common it is at the object's magnitude:
Σ_Q(z, r), quasars per deg² per mag per unit redshift, against Σ_B, the background objects counted directly from
the imaging catalogue at the object's sky position. Both models share the same Σ_Q.

Σ_Q is the eBOSS quasar luminosity function (Palanque-Delabrouille et al. 2016, A&A 587, A41, Table 7, PLE+LEDE
fit), smoothed onto our redshift and magnitude grid (`scripts/method_unified/build_qlf_prior.py`). It replaced
spectroscopic DESI + SDSS counts multiplied by one completeness constant. Those counts were 1.4–3 times too high
for bright quasars (r < 20), because SDSS is nearly complete there and the constant over-corrected. They were 2
to 50 times too low beyond r ≈ 22.3, where spectroscopic targeting ends. The luminosity function is fitted to
r ≈ 22.3 and 0.68 < z < 4; beyond that it is the published model extrapolated. The method note (Section 9.1)
compares the two (figure `plots/qso_prior/Q3_counts_three_models.png`).

### The outside-both-models test (default since 5 October 2026)

Before ranking, the scorer asks whether the colours resemble *anything* in either model. For each population
(quasars at any redshift; background at the object's sky position) it computes the nearest-component
distance of the object's colours given its Legacy r, with its measurement noise: `qso_ood_sigma_any_z`
and `bkg_ood_sigma`, in σ.

That distance grows with the number of observed bands: a typical member observed in 20 bands lies at
about 4.5σ. The earlier rule (refuse if both distances exceed 4σ) therefore refused ordinary objects with
many bands: 2.5–2.9% of held-out test quasars, rising from 0% at fewer than 12 bands to 7–15% at 20 or
more.

The default test calibrates the same distance for each object. It draws 256 members of each population in
the object's own bands, with its own noise and Legacy r, and computes the same distance for them. The tail
fractions are reported as `qso_ood_p` and `bkg_ood_p`: the probability that a real member lies at least as
far away. The object is outside both models when both are below 2/256. Refusals then no longer depend on the
number of bands, and about 0.15% of test quasars and 0.5% of background objects are refused. It adds about
13 ms per object. The method note (Section 9.2) and `docs/pquasar_diagnostics/pquasar_diagnostics.pdf`
give the derivation and validation. Pass `ood_calibration=None` to reproduce scores made before
5 October 2026.

---

## Unresolved blends: arguments and output

`BlendModel.load(pointer, faint_snr=10, n_alpha=48)` builds the blend scorer on a bundle.
`score(photometry, *, ra_deg, dec_deg, z_qso, prior_odds_qq=1.0, dereddened=False)` returns one
`BlendScore` per object. It does **not** take morphology, separation, `fracflux`, `BlendPolicy`,
`RedshiftMatch` or `ood_flag_sigma`: the source is assumed to be a blend, and nothing is known about
the components separately.

**Hypotheses.**
- QQ: the companion is a quasar at the same redshift *z*₀.
- QS: the companion is a star (the background population at that sky position).
- Not yet implemented: a quasar companion at a different redshift, a negligible companion, and
  companions below the detection limit.

**Base rules.**
- Each component must carry at least `faint_snr` (10) times the blend's Legacy r flux error, so the blend
  needs Legacy r S/N ≥ 20.
- The likelihoods integrate over the companion's share α of the r flux.
- There is no catch-all and no support cut.

**`prior_odds_qq`** is the prior odds P(QQ)/P(QS) that a *detectable* companion of such a quasar is a
quasar rather than a star. It is one scalar per call; for per-object odds, call once per object.
`p_qq` depends on it directly. `log_bf_qq_qs` does not, so report the Bayes factor when the odds are
uncertain.

| `BlendScore` field | Meaning |
|---|---|
| `status` | `ok`, `too_faint_for_two_components` (Legacy r S/N < 20), `no_reference_band`, `redshift_outside_support`, `no_prior_for_reference_band`, `empty_magnitude_prior`, `empty_hypotheses` |
| `log_like_qq`, `log_like_qs`, `log_bf_qq_qs` | log likelihoods and Bayes factor QQ:QS (natural log) |
| `p_qq` | posterior P(QQ) at `prior_odds_qq` |
| `alpha_mean_qq` | posterior mean share of the r flux in the *fainter* quasar under QQ (the hypothesis is symmetric) |
| `alpha_mean_qs` | posterior mean share of the r flux in the star under QS |
| `alpha_range` | the range of shares integrated (set by the detection limit) |
| `reference_band`, `bands_used`, `notes` | provenance; `no_colour_information` if only Legacy r is given |

---

## When not to trust the numbers

Sources: method note Sections 9--11 and `docs/pquasar_diagnostics/pquasar_diagnostics.pdf`.

- **`p_quasar` is approximate, and untested at faint magnitudes.** In the mid range it lies between the
  label bounds or slightly above the upper one. Above 0.97 it exceeds the fraction of *known* quasars
  (0.67–0.97 observed at p > 0.99), but most of the excess objects are flagged likely quasars without
  spectra. At r ≳ 22.5 the labels miss most real quasars (no spectrum, no WISE detection), so the
  calibration of faint `p_quasar` cannot be measured.
- **Quasar abundance beyond r ≈ 22.3 and below z = 0.68** is the eBOSS luminosity function extrapolated,
  and its redshift shape inside each Δz = 1 bin is interpolated. The scorer declines to
  rank 1.0% (B) and 1.2% (A) of known quasars in a natural sample, mostly at the QSO-support cut: treat an
  unranked object as "not a candidate", not as "not a quasar".
- **Variability between surveys** observed years apart is not modelled. A quasar whose SDSS or PS1
  magnitudes disagree with Legacy can look unusual; if it is refused, rescoring with Legacy bands only is a
  valid check.
- **`p_sameq` in narrow windows** has not been tested for reliability on the current models.
- **Selection.** Test quasars share the training selection (DESI, SDSS), so the risk that A has learned
  the selection is untested. This is why B is kept.
- **Blends: share of the light.** The two hypotheses separate when the companion carries ≥ 30% of the
  r flux (AUC 0.88–1.00 by star colour), weakly at 20%, and close to chance at ≤ 10% unless the star is an
  M dwarf (method note Section 11.1).
- **Blends: magnitude.** Separation weakens towards blend r ≈ 21.5 for companions carrying 10–20% of the
  light; true QQ blends still favour QQ there (median log BF 0.7–2.1).
- **Blends: validation.** All blend validation uses synthetic blends of real point sources, not real
  blends.
- **Sparse bands.** NSC VR and VHS Y extinction coefficients are approximate, and SkyMapper u, v and VHS
  have little training support.

## Validation in brief

| Test (held out) | A | B |
|---|---|---|
| Quasars vs background objects, AUC (log R; unranked at the bottom) | 0.9939 | 0.9934 |
| same, *r* < 18 | 0.982 | 0.986 |
| Test quasars with `p_quasar` > 0.5 | 94.0% | 91.2% |
| same, 23 < *r* < 24 | 54% | 30% |
| Background objects not flagged as likely quasars, with `p_quasar` > 0.5 | 0.44% | 0.33% |
| Test quasars not ranked | 0.61% | 0.57% |
| Blend mode, QQ vs QS, AUC (synthetic blends of test objects) | 0.947 | 0.927 |
| Blend mode: fraction of true QQ / true QS blends given P(QQ) > 0.9 (prior odds 1) | 75% / 7.2% | 63% / 6.9% |

Details: method note; `docs/method_unified/*.json` (`model_difference_bootstrap.json`,
`blend_validation.json`, `blend_illustrations.json`).

---

## Repository map (current models)

| Path | What |
|---|---|
| `src/qso_pcolor/unified.py` | `UnifiedPSFModel` (models A, B), `catalogue_photometry`, `quasar_probability` |
| `src/qso_pcolor/blend.py` | `BlendModel` (model C) |
| `src/qso_pcolor/multisurvey_baseline.py`, `multisurvey.py`, `score.py` | the scorer underneath |
| `src/qso_pcolor/multisurvey_data.py` | band schema, WSDB catalogue columns and cleaning |
| `models/multisurvey_psf/` | bundles and pointers; each bundle's `build.json` records how it was built |
| `docs/method_unified/` | method note, validation JSON |
| `scripts/method_unified/` | figures and checks for the note |
| `scripts/validate_blend_mode.py`, `scripts/blend_illustrations.py` | blend validation and maps (the maps take ~25 CPU-min) |
| `docs/EARLIER_MODELS.md` | earlier models, their interfaces and reproduction |
| `tools/journal.py` | JOURNAL.md updater |

`python scripts/<name>.py --help` for options.

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
