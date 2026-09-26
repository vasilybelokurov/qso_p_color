# AGENTS.md — implementation brief for Claude Code / Codex CLI

Read this before touching the repository. It is the working contract for
`qso_pcolor`. `docs/method/method.tex` is the scientific write-up — the
formalism, the derivations and the figures — and is the place to look for *why
the method is what it is*. `docs/REVIEW_OF_PLAN.md` explains why the design
departs from `qso_binary_color_probability_plan.md` where it does. This file
says *what to do*.

**If you change the method, update `docs/method/method.tex` in the same
commit.** A write-up that describes a previous version of the code is worse
than none.

**There is one model** (2026-09-26): `MultiSurveyModel` in
`models/multisurvey.json`, with `models/multisurvey_priors.json` and
`models/multisurvey_outlier.json`. It takes any subset of 41 bands from seven
surveys. Everything else under `models/` is in `models/archive/` and is not a
model to score with (section 10). Milestones M1–M7 below are the history of the
retired Legacy-only model; their rules still hold, their file names are archived.

**Commit and push after every tested step.** Run the suite to a file, commit
only if it passed (8b), then `git push origin main`.

---

## 1. The question

A quasar has a spectroscopic redshift *z*₀. A photometric companion sits a few
arcseconds away. Using broadband photometry only, how strongly does the
companion's colour support the hypothesis that it is itself a quasar at
*z* ≈ *z*₀?

Three hypotheses, never two — plus a fourth that stops the model's tails from deciding:

| symbol | meaning |
|---|---|
| `same_z` | quasar whose redshift matches *z*₀ under a declared window |
| `field_q` | quasar at some other redshift |
| `bkg` | anything else in the imaging catalogue at that brightness and sky position |
| `out` | the broad "unmodelled" share η(u_a) of that same field (`MultiSurveyOutlier`, `models/multisurvey_outlier.json`) |

Without `out`, an object far from both colour models is scored by whichever
Gaussian tail happens to be wider, and is called a quasar with probability one
(measured, 2026-09-26; method note, section "Colour outliers"). Pass
`outlier=MultiSurveyOutlier.load(...)` to `MultiSurveyModel.score` for real
candidates.

Dropping `field_q` is the dominant failure mode: a real quasar at *z* = 2.6
beats the stellar locus easily and would otherwise be scored as evidence for a
same-redshift companion.

**Know this before you start.** A ±2000 km s⁻¹ window at *z* = 1.4 is
Δ*z* = 0.016; colours constrain a quasar redshift to a median σ_z ≈ 0.6
(measured, `docs/REVIEW_OF_PLAN.md` §9a). So
`p_sameq` under a velocity window is small even for a perfect candidate. The
package ranks candidates and reports evidence; it does not deliver "the
probability this is a binary". Do not tune anything to make that number look
larger — and in particular **do not widen the window to raise it**, which
changes the question rather than the answer.

### Rank on `log_r_per_unit_z`, not on `p_sameq`

Because the window is far narrower than the photometric redshift resolution, it
enters as a pure multiplicative constant:

```
R = Sigma_Q(z0, m) p(c | Q, z0) / (Lambda_Q,total + lambda_bkg)     [1/redshift]

p_sameq ≈ R * dz_match_eff                               (narrow-window limit)
```

The denominator carries no window dependence at all — `same_z` and `field_q`
partition the quasar intensity, so their sum is the total whatever window is
declared. `R` is therefore the window-free evidence, and any posterior follows
from one multiplication in the narrow-window limit. The implementation stores
the window-averaged `R`, for which `p_sameq = exp(log_r_per_unit_z) *
dz_match_eff` is exact; wider windows can change that average.

Consequences, all enforced by tests:

- **`R` is the ranking statistic.** Two people can compare candidates without
  agreeing on a window.
- **Never quote `p_sameq` without `dz_match_eff` beside it.** They travel
  together in every output row.
- **Never integrate a velocity window on the redshift grid.** Δ*z* = 0.03 on a
  grid of step 0.01 is two or three points, and the trapezoid rule of that is
  noise. The scorer inserts dense window nodes and exact support boundaries,
  sharing those nodes with the total-intensity integral. It also inserts model
  and prior interpolation knots so a narrow prior peak enters both integrals.

---

## 2. Current state

Working and tested (`pytest -q` → all green; run it before and after any change):

```
models/
  multisurvey.json          THE model: 41-band quasar slices, joint field fit,
                            Legacy-south grz and grzW1W2 field fits, transform
  multisurvey_priors.json   Sigma_Q(z, u_a), Sigma_B(u_a) for 37 reference bands
  multisurvey_outlier.json  the unmodelled term (envelope, eta per band and bin)
  archive/                  retired models; provenance and regression only
src/qso_pcolor/
  gaussmix.py    batched log-densities, missing-dimension marginalisation,
                 per-object noise convolution, joint conditioning,
                 nearest-component distances
  xd.py          extreme deconvolution (Bovy, Hogg & Roweis) in numpy
  multisurvey_data.py  41 survey-labelled bands, native photometry, catalogue
                 adapters and quality cuts, same_training_config
  multisurvey.py MultiSurveyModel (joint mixtures conditioned on a reference
                 band), load_priors, MultiSurveyOutlier
  score.py       the scorer shared by every model, PairScore output record
  outlier.py     envelope covariance, dominance check, ML outlier fraction
  qso_model.py   sliced redshift-conditional mixtures; RedshiftMatch
  priors.py      GridQSOPrior, BackgroundSurfaceDensity
  background.py, features.py   the retired model's field model and flux ratios
  data.py        WSDB queries, cached to .npz
  plotting.py    save_figure: every figure a PNG under plots/
scripts/ (the model)
  build_multisurvey_sample.py   quasar training sample, 24 field cones, overlap field
  train_multisurvey_model.py    slices and joint field fit, spatial K selection
  fit_background_marginal.py    dedicated Legacy-only field fits (append)
  build_multisurvey_priors.py   surface densities per reference band
  fit_multisurvey_outlier.py    kappa and eta on field sources the fit never used
  validate_pairs_multisurvey.py labelled pairs, against the predecessor on the same rows
  validate_multisurvey.py       all 127 survey subsets on reserved objects
  make_validation_figures.py, plot_colour_redshift.py, make_multisurvey_examples.py
scripts/ (the retired model; run against models/archive/)
  train_qso_model.py, build_global_background.py, validate_pairs.py,
  compare_background_modes.py, fit_outlier_model.py, outlier_analysis.py,
  make_method_figures.py, score_examples.py, compare_old_new_models.py, ...
scripts/build_pair_validation.py   labelled close-pair sample from DESI DR1
tools/journal.py          JOURNAL.md updater
docs/method/              method.tex + Makefile -> method.pdf
tests/                    185 tests; see section 7
```

A general CLI and the remaining calibration work are not yet complete.

---

## 3. Rules that are not negotiable

These come from the science, and a change that breaks one is a bug even if the
tests pass.

1. **Never call a likelihood a probability.** `loglike_qso_zprimary` is
   *p(colours | quasar at z₀)*. Field names must not blur this.
2. **No scientific threshold in source code.** Redshift windows, magnitude
   edges, HEALPix resolution, component counts, separation floors: configuration
   or cross-validation. `RedshiftMatch` deliberately has no default and raises
   without one.
3. **Keep everything in log space.** Use `logsumexp`. Never form a covariance
   inverse; use Cholesky factors and triangular solves.
4. **Colour errors are correlated.** Any transform must return a full matrix.
5. **A negative flux is a measurement.** Never clip, floor, or drop a band
   because its flux is negative or low signal-to-noise. A band is unusable only
   when the survey says so (`flux_ivar <= 0`, `nobs = 0`, mask bit), and that
   goes into the `observed` mask, which is marginalised exactly.
6. **Never mix photometric systems.** SDSS *ugriz*, LS DR9 *grz*, LS DR11 north
   and LS DR11 south are four different systems. Models carry a system string
   and the scorer raises on a mismatch.
7. **No prior, no posterior.** If a defensible Σ_Q is unavailable, return the
   Bayes factor and leave the posterior fields NaN with a status code. Never
   substitute a spectroscopic class fraction.
8. **Report evidence and posterior separately**, plus the out-of-distribution
   score and the quality flags. A high posterior from contaminated photometry
   must be distinguishable from a high posterior from clean photometry.
9. **De-duplicate by sky position, not by identifier.** `zcat_primary` still
   leaves 9,577 repeated quasars in DESI DR1 (measured).
10. **Every random procedure takes an explicit seed** from the run config.
11. **Scope: cleanly deblended companions only, for now.** Every candidate
    scored for science passes a `BlendPolicy` (separation floor, `fracflux`
    limit). Blends are deferred to M7 and need image-level forced photometry,
    not a looser threshold. A missing separation or `fracflux` counts as
    blended: an object cannot be certified clean without the numbers that
    would show it.
12. **Figures are PNG and live in `plots/`.** Write them with
    `qso_pcolor.plotting.save_figure(fig, "name")`, which forces the format,
    creates subdirectories, and returns the path — do not call `savefig`
    directly with an ad-hoc path.

---

## 4. Environment and house style

```bash
source ~/Work/venvs/.venv/bin/activate
cd ~/Work/Code/qso_p_color          # symlink to the Dropbox copy; same directory
pip install -e . --no-deps
python -m pytest -q
```

- Python, numpy 2.5 / scipy 1.16 / astropy 8. `healpy` for sky cells.
- NumPy-style docstrings that state **units** and **what the density is
  normalised over**.
- Type annotations on public functions. Vectorise; loops only where the
  per-object covariance genuinely forces them.
- WSDB access through `sqlutilpy` (`import sqlutilpy as sqlutil`), not psql, when
  the result feeds numpy. Read `~/.claude/skills/wsdb/references/sqlutilpy.md`
  first. Column names come **first** in `q3c_radial_query` and `q3c_join`, the
  local list goes first in a join and the survey table second, and every pull is
  cached to `.npz` via `qso_pcolor.data.cached_query`.
- **`JOURNAL.md` is local and gitignored.** A post-commit hook appends an entry
  for every commit (subject, body, diffstat, test state), so the mechanical
  cadence is automatic. Install it once in a fresh clone:

```bash
python tools/journal.py hook-install
```

  The hook does *not* capture findings. After any substantive step — a fit that
  produced a number, a choice settled by a measurement, a hypothesis abandoned —
  add one explicitly:

```bash
python tools/journal.py add --what "..." --found "..." --next "..."
```

  Bypass with `QSO_JOURNAL_SKIP=1 git commit ...`, or skip the test run with
  `QSO_JOURNAL_NO_TESTS=1`.

---

## 5. Work order

Each milestone states what "done" means. Do not start one before the previous
acceptance criteria pass.

### M1 — labelled validation sample  ← **start here**

The plan put this last. It goes first, because nothing downstream can be
calibrated without it, and because it is cheap.

```bash
python scripts/build_pair_validation.py --out data/pairs_desi_dr1.npz
```

Expected yield at 3–20″ separation (measured 2026-09-18): ≈ 1,600 `same_z`,
≈ 8,400 `field_q`, plus every confirmed star/galaxy companion.

**Done when**: the file exists; the de-duplication report is in `JOURNAL.md`;
counts are broken down by separation bin and by `fracflux_r`.

### M2 — real quasar colour model, LS DR9 *grz* (+W1, W2)

Training set from `qso_pcolor.data.fetch_desi_qso_training` — joined on
`targetid`, so no crossmatch. Split by `release` (9010 south / 9011 north) and
fit each separately.

- Exclude the validation pairs from training (spatially blocked folds by
  HEALPix group, `select_n_components(..., groups=...)`).
- Choose the number of components and the redshift-slice width by held-out
  predictive density, not by assertion.
- Fit optical-only *and* optical+WISE variants; keep both.

**Done when**: held-out log density is reported as a function of redshift and
magnitude; a coverage test shows the fraction of held-out quasars whose
spectroscopic redshift falls inside the nominal credible interval of
`redshift_posterior`, and it is close to nominal.

### M3 — background colour model and surface density

Same footprint, same quality cuts, same photometric system as the candidates.
`fetch_ls_background` with `maskbits = 0`; apply the identical cut to
candidates, and use the **masked** area for Σ_B, not the nominal cone area.

Tune the pooling constant `n0` with `tune_shrinkage` on held-out sky blocks.

**Done when**: predicted versus observed counts agree by sky cell and magnitude;
the local/parent/global fallback fraction is reported; held-out log density is
plotted against |*b*| and magnitude.

### M4 — Σ_Q and the posterior

`EmpiricalQSOPrior.build` over a stated area. Default completeness 1 — a
declared, reproducible choice, **not** a conservative one. It makes `p_sameq` a
lower bound only if completeness is redshift-independent; if completeness is
worse away from `z0` than at it, correcting it raises the field term more and
the posterior falls. Say that in the report rather than inventing a luminosity
function.

**Done when**: `test_bayes_factor_is_invariant_under_a_prior_shift` still
passes on the real models, and every scored row carries both evidence and
posterior fields.

### M5 — calibration on the M1 sample

This is the milestone that decides whether the method works.

- Reliability curve, Brier score, log loss, precision–recall.
- Stratified by separation, `fracflux_r`, reference magnitude, |*b*|, and
  redshift — because prevalence changes strongly along all of them.
- Compare `SlicedColourRedshiftModel` against `JointColourRedshiftModel` on
  held-out likelihood, and settle that choice with the number.
- Answer explicitly: does magnitude conditioning of the quasar model improve
  held-out density? If not, leave it at one bin.

**Done when**: a calibration report exists with those curves, and the answers to
the two model-choice questions are recorded in `JOURNAL.md` with the numbers
that settled them.

### M6 — candidate scoring and reports

CLI over `score_candidates`, Parquet output matching the contract in section 6,
plus a one-page diagnostic per interesting candidate: observed colours with
errors, the quasar locus at *z*₀, the local background density, the quasar-only
redshift PDF, the Bayes factor, both posteriors, and every quality flag.

### M3a — two background modes, and comparing them

`fit_background_model` (HEALPix hierarchy, amortised over a footprint) and
`fit_local_background` (one cone around a single candidate) now both exist, and
`compare_backgrounds` scores the same candidates under both and reports how much
the choice moved each quantity. Use the hierarchy for a survey-wide scan and the
local fit for specific candidates; disagreement between them is a diagnostic
that the answer depends on the background assumption, not a failure.

Both now **remove known quasars** (`drop_known_quasars`, DESI DR1 + SDSS DR16Q
within 1 arcsec). This is not optional: measured on a 1-degree cone, known
quasars are 1.0% of the background sample overall but **72% of it where
log BF > 5** — the colour region a genuine companion occupies. Leaving them in
made the background model learn the quasar locus and compete against itself,
costing roughly 1.3 in log BF, comparable to the entire same-z versus
wrong-z gap.

Removal handles only *observed* quasars, so the unobserved ones stay in the
background. That is a residual of the same bias in the same direction, and worth
measuring — but it is **not** a reason to block on obtaining a completeness, and
an earlier version of this note said it was.

A uniform completeness correction multiplies **both** `lambda_sameq` and
`lambda_fieldq`, so it cancels in `p_zmatch_given_qso`. It does not cancel
against the background: if the multiplier is `a`, `p_sameq` and `R` change by
`a*(Q+B)/(a*Q+B)`, where `Q` is total quasar intensity and `B` is background
intensity. This factor depends on colours, so ranking is not guaranteed to
remain unchanged. The previously quoted 13% shift was an example, not a
candidate-independent bound.

What does *not* cancel is **redshift-dependent** incompleteness, which changes
the shape of `Sigma_Q` between inside and outside the window: intensities
(1, 1, 1) give p = 1/3, while a correction multiplying only the field term by
100 gives 1/102. Thus the shape affects the quasar-only redshift probability,
while the level affects competition against the background.

### M2a — the classifier review (2026-09-19)

An independent review of the classifier found five defects, all reproduced:

- **The training script's "held-out" evaluation was not held out.** It fitted
  every usable object and then sampled those same objects, so the
  selection-channel comparison would have measured training density. Fixed:
  whole nside=4 sky blocks are now reserved *before* fitting (`--holdout-frac`,
  default 0.2) and the per-channel numbers come only from the reserved blocks.
- **The saved models could not be combined**: the quasar model declared
  `ls_dr9_south_grzw` and the background `ls_dr9_grzw`, which the scorer's own
  system check rejects. Names are now release-aware everywhere.
- **The local background divided masked counts by unmasked area** — the same
  class of error as the 17x `Sigma_B` bug. The cone query now returns
  `maskbits` unfiltered, the caller applies the cut, and the masked fraction
  comes out of the area. Estimating it from source counts is biased (masked
  regions sit around bright stars, where detections are denser), so the value
  and the fact that it was estimated are both recorded.
- **`--select-k` sampled only z ≈ 1.8**, capped at 40k objects, and so could not
  choose K across 0.4 < z < 3.6. It now samples uniformly across the range.
- **The figure script fitted its background without removing quasars.** Fixed.

Design points taken from the same review, not yet acted on:

- Make the *decision* binary — target-redshift quasar versus everything else —
  while keeping all three components internally. Folding `field_q` into the
  alternative is equivalent when the weights are right; dropping it is not.
- The cheapest real test is not the close-pair sample: reserve spatial blocks,
  then score withheld quasars inside and outside the window, withheld stars and
  galaxies, **including quasars just outside the window**, stratified by
  magnitude and error. Fig. 7 omits that hard-negative regime entirely.
- Compare r-reference, z-reference and asinh on identical withheld objects by
  classification loss versus reference S/N — not by raw predictive density,
  which is not comparable across coordinate systems without the Jacobian.

### M6a — close out the external review

`docs/reviews/2026-09-19-*` (local, gitignored) holds an independent Codex
review of the pipeline and the note. Every checkable finding was reproduced and
all were fixed except the two below, which are recorded rather than done:

- **The background sample still contains quasars**, so `field_q` and `bkg` are
  not strictly disjoint. Needs a crossmatch, and the size of the effect
  measured *in quasar-like colour space* — "quasars are rare" does not bound it.
- **`tune_shrinkage` cannot see `n0`.** It holds out whole nside=2 cells, which
  is also the default parent resolution, so every validation object falls back
  to the global model and carries no information about the pooling constant.
  Hold out at a finer resolution than the parent, or tune on cells rather than
  objects.

`tests/test_review_regressions.py` pins every defect that was fixed.

### M6b — close out the package-state review (2026-09-20)

A second Codex pass assessed the repository as a handover artefact. Every
checkable finding was reproduced; `/tmp/codex_state.md` (local) has the
original. **Batch 1, done:**

- **The holdout split was re-derived, not recorded.** `train_qso_model.py`
  stored `holdout_frac` and `holdout_nside` but neither the seed nor the block
  IDs, so `score_examples.py` re-derived the draw from a different sample
  (DESI-only, no de-duplication, no quality cut): 78 candidate blocks against
  training's 81, 7 of 16 blocks shared, 52 % of "reserved" objects actually
  fitted, and 3 of the 5 quasars in fig. 9 trained on. Measured cost: **+0.018
  nats** in-sample versus held-out, matched in (z, r-mag) over 45 cells,
  against log B ~ 5-15 — negligible, so the numbers stand and only the claim
  was wrong. `holdout_seed` + `holdout_blocks` are now written at training
  time; `scripts/recover_holdout_blocks.py` recovered them for the shipped
  model and verified the replay against all four recorded counts; consumers
  read the list and refuse rather than guess.
- README's model description (said 1.24 M DESI; a third of the sample is SDSS),
  install extras, test count, and example (l, b).
- Local background caches keyed on (ra, dec, radius, system, bins), not on the
  loop index — `--seed` used to silently reuse the previous run's cones.
- `fit_local_background` now checks the returned `release` against the `system`
  label it is asked to stamp, instead of certifying whatever it was handed.

**Batch 2, done (2026-09-20):**

- **The scorer no longer integrates outside model support.** `DEFAULT_Z_GRID`
  spans 0.05-5.0; the model now 0.15-4.35. `log_p_colour_given_z` clamps
  outside its range, so a wider grid replays the edge slice rather than
  extending the model, and `p_zmatch_given_qso` then depended on where the grid
  happened to stop. The grid is masked to `in_support` and the discarded share
  is reported as `frac_norm_outside_support` — a large value now means the
  candidate's colours are best explained at a redshift the model has never
  seen. Measured for the README candidate: 7.08 % before the range was widened,
  2.25 % after, 0 % integrated either way now.
- **`log_bayes_factor_qz_bkg` is no longer offered as a ranking statistic.** It
  contains no `field_q` term, so a foreground quasar at z = 2.6 scores superbly
  on it — the exact failure the three-hypothesis framing exists to catch.
- **`load_sdss` is keyed on its query.** The positions query was hash-keyed but
  the match result it fed was not, so the hash protected the wrong half and
  changed `--zmin/--zmax` silently reused the old sample.
- **Caches write atomically** (`os.replace`) and are loaded without
  `allow_pickle` unless the file genuinely needs it, with a warning when it
  does. A truncated cache is worse than none: the next run loads it and
  proceeds.
- **The Makefile depends on figure 9**, which lives under `plots/examples/` and
  was missed by the `plots/method/*.png` wildcard.

**Retrain done (2026-09-20, 4.3 h):** `maskbits = 0` on both channels
(69,987 DESI quasars, 5.41 %, removed), native 0.1–4.4 in 43 slices, K = 20
where n ≥ 10,000 and selected per slice below (12/8/4 at high z), the 16
holdout blocks READ from the previous model, output to a dated file and
promoted only after `validate_pairs.py` passed. Result: on 40,000 identical
mask-clean held-out quasars the new and old models agree to **+0.0008 nats**;
every validation AUC moved by ≤ 0.006; calibration ratios identical to two
decimals. Masked quasars really are different — 0.85 nats below the clean
locus under the old model — but at 5.4 % of the sample they did not move the
fit. **The retrain bought consistency and provenance, not accuracy.** Recorded
so nobody expects otherwise from a rerun.

**Found while promoting (2026-09-20):** Legacy Surveys release **9012** is
DECam photometry too — reprocessed southern bricks (e.g. `3273p080`, 577
sources in one 0.5° cone beside 69,165 at 9010). `fit_local_background` now
admits {9010, 9012} for the south. The DESI training cache holds **9,628
release-9012 quasars (0.7 %)** that `train_qso_model.py` excludes with
`release == 9010`; consistent treatment would include them. Not worth a
retrain on its own — fold into the next one.

**Historical, now resolved by the retrain:**

- **Training quality cuts differ by channel.** The SDSS match requires
  `maskbits = 0`; the DESI branch returns `maskbits` and never applies it. One
  third of the training set is mask-clean, two thirds is not. Needs a retrain.
- **K is selected on the wrong model.** `--select-k` pools 0.4 < z < 3.6 into
  one sample, fits a single *unconditional* mixture, and uses that K in all 32
  conditional slices. Not held-out selection of the deployed model. Needs a
  retrain; do it with the `maskbits` fix.
- `allow_pickle=True` and non-atomic writes in `data.cached_query`; the README
  offering `log_bayes_factor_qz_bkg` as an alternative ranking statistic when it
  contains no `field_q` term. Both cheap; folded into batch 2.

### M6c — redshift range extended (2026-09-20)

The trained range 0.4 < z < 3.6 was a default, not a data limit. Outside it
`log_p_colour_given_z` clamps and replays the edge slice, so 7.1 % of the
no-prior redshift normalisation for the README candidate was fiction.

`scripts/extend_qso_model_redshift.py` widens a trained model by **appending**
slices: the existing mixtures are untouched, the recorded holdout is reused, and
the run takes ~35 min instead of a 4 h retrain. 11 slices added (3 below, 8
above) → 43 slices, support **0.15–4.35**.

Two constraints, both load-bearing:

- **Homogeneity.** New slices use the same transform, S/N floor, release,
  de-duplication, quality cuts, EM settings and seed. A selection change at the
  join (e.g. applying the `maskbits` fix to only the new slices) would later
  read as a feature of quasar colour. Selection changes belong to a retrain.
- **K per slice.** The core slices have ~35,000 objects for K=20 (~117 per free
  parameter); the sparsest new slice has 893 (~3 per parameter). K is chosen per
  appended slice by spatially blocked held-out density and falls 20 → 12 → 8 → 4.
  `min_per_slice` is a don't-crash fallback, not a quality criterion — do not
  use it as one.

Measured, not assumed (`scripts/check_redshift_extension.py`): on 15,552
quasars in the reserved blocks, mean gain **+1.90 nats**, 77.9 % improved,
rising monotonically from +0.1 nats just outside the old edge to +5.0 at
z ≈ 4.3. That shape is the signature of a repaired deficiency — extra
parameters absorbing noise would not know where the old boundary was.
Out-of-support normalisation 7.1 % → 2.25 %, so the grid clip in batch 2 is
still wanted as a safety net.

### M1 — done: the pair validation (2026-09-20)

`build_pair_validation.py` rewritten around an index-driven search (the
original pulled 20.4 M spectra; a joined query planned as two Seq Scans; split
into pair search → spectra by targetid it runs in 4 min). `validate_pairs.py`
scores every companion and reports on all pairs and on the reserved blocks;
they agree to 0.01 everywhere.

Findings, in order of consequence:

- **The colours DO carry redshift information at ±3000 km/s:** AUC 0.84
  (same_z vs field_q, by log R). The note had argued from σ_z ≈ 0.6 that it
  would be weak. Measured, it is useful.
- **Calibration is off by a separation-dependent factor: 9.8× at 3–5″ falling
  to 3.3× at 20–30″.** Shape is right (AUC 0.86 by p_zmatch alone), only the
  normalisation is short, and a scorer bug would not know about separation.
  This IS the physical-pair clustering excess that M7 deliberately leaves out
  of the scorer — first measurement of it. Not yet known whether it reaches 1
  beyond 30″; extending to 60–120″ is the next run.
- **~~Galaxies are the contaminant~~ — RETRACTED 2026-09-20 evening.** That
  result (AUC 0.81, 10.5 % above the same_z median) came from validating
  against the examples figure's eight-field background, which was fitted to
  `type = 'PSF'` sources only: a background with no galaxies in it cannot
  recognise a galaxy as background. Against the shipped all-source background
  (`scripts/build_global_background.py`) the same galaxies give AUC **0.96**
  and **0.3 %** above the median; stars 0.98 / 0.4 %. The quasar-vs-quasar
  AUC by log R moved 0.84 → 0.81 (denser background in R's denominator);
  by p_zmatch, which no background enters, unchanged at 0.86; calibration
  table unchanged. Morphology stays useful as a *flag*, not a gate. The
  general lesson is the design principle in the note: the background must be
  built from everything a chance neighbour can be, with no selection the
  candidates do not share.
- **Bayes factor vs log R: 0.76 vs 0.84.** Earlier text said the BF "cannot"
  separate the classes; it can, worse. Wording corrected in README and note.
- **Bug found and fixed:** `GridQSOPrior.__call__` tested magnitude against
  bin *centres* not *edges*, refusing 17 ≤ r < 18.25 and 22 < r < 22.5 — 20 %
  of the sample — as "prior empty". Edges now stored and used. Test pinned.

Not done, and why: per-candidate local backgrounds (days of cones for 5e4
pairs, and the quasar-vs-quasar comparisons do not use the background);
blends < 3″; north.

**Consequence for the retrain:** it can go ahead on its own merits (maskbits),
but nothing here says it is urgent. The larger scientific gains are the
morphology gate and the clustering factor, neither of which needs a retrain.

### M3b — global vs local background, measured (2026-09-20)

30 fully covered 0.5° cones, |b| 20–78°, both hemispheres, each split 80/20;
local model fitted on the 80 %. Held-out field log density local − global:
+0.08 nats at |b| 20–45° (max +0.21, in the densest cones), +0.01 at 45–60°,
0.00 above 60°. False-positive rate (log BF > 0 at z0 = 1.8): identical under
both models at every latitude — and it RISES with |b| (0.015 → 0.03) because
the high-latitude faint field is galaxies. Quasar log R shifts +0.1 nat under
local at all latitudes: small-sample mixtures put less mass in the tails where
quasars sit, so they underestimate p(c|B) there. **Rule: shipped global model
by default; local for |b| ≲ 25°, densities ≳ 3e4/deg², or odd neighbourhoods.**
The footprint cannot test the plane. Figure: plots/validation/background_modes.png;
per-cone table: data/background_modes.json (local).

### Maintenance validation (2026-09-21)

The current trained models are retained unchanged. Shared supported integration
nodes repair scoring on coarse grids; an empty prior retains colour evidence;
row subsets retain their flags; future redshift extensions inherit the saved
maskbits policy. Validation now applies `maskbits = 0` and an explicit
`fracflux_r` limit, derives labels from the requested velocity window, and saves
the full score contract. Run with `--max-fracflux 0.2
--hard-negative-max-kms 6000 10000`.

The clean rerun scores 50,746 companions, 14,221 held out. Held-out AUCs are
0.815 by log R, 0.759 by log BF and 0.855 by p_zmatch. Against quasars only
3,000–6,000 km/s from the primary they are 0.488 by log R and 0.508 by p_zmatch;
the broad comparison does not establish velocity resolution. See
`docs/MAINTENANCE_2026-09-21.md` for the changes and verification. Clustering
remains outside this maintenance scope; no retraining is needed or performed.

### Colour outliers: the unmodelled hypothesis (2026-09-26)

Far from both colour models, log p(c|Q)/p(c|B) is a difference of Gaussian
quadratic forms and runs away; with three hypotheses such objects were called
quasars with P = 1.000 (README candidate with g x 8: 6.6 sigma from every
quasar component, 14.8 from the background, ln BF +24.7). In the pair
validation 7 of the 11 non-quasars > 3 sigma from both models were called
quasars (0.9 % near the loci). Remedy, in `outlier.py`:

- **Two distances, always reported:** `qso_ood_sigma_any_z` (all slices;
  `qso_ood_sigma` is z0-only and is large for every wrong-z quasar) and
  `bkg_ood_sigma`. `outside_both_models` needs an explicit `ood_flag_sigma`.
- **A fourth hypothesis U**, eta(m) of Sigma_B, density N(mean_B, kappa^2 E).
  E is an envelope >= every component (Gershgorin bound in the whitened,
  rotated frame); an isotropically scaled field covariance would need kappa =
  67, because a few IR-bright quasar components are 67x wider than the field
  in w1/r, w2/r. kappa > kappa_min is enforced on construction.
- **Calibrated on cones 08-32 of `compare_background_modes.py`** (00-07 are
  the background's own fields), split A/B: eta = (6.2, 8.6, 4.3, 2.4)e-4,
  kappa = 1.016 (the dominance limit binds; likelihood wants narrower), gain on
  B +0.0090 nats/object.
- **Measured effect:** tail non-quasars called quasars 7/11 -> 0/11; every
  held-out AUC unchanged to three decimals; 99.9 % of |d ln R| < 0.004. Cost:
  quasars far from the background but near a quasar component lose Bayes
  factor (median 26.9 -> 12.4 for 84 objects; 3 fall below 0).
- **Not for the multi-survey scorer**, whose densities are conditional; it
  raises if given one, and reports conditional distances.

### M7 — blends, and other extensions

Everything above assumes a cleanly deblended companion, enforced by
`BlendPolicy`. Blended pairs are a separate problem and are addressed here, in
this order:

image-level forced photometry below the separation floor; Gaia parallax/proper
motion as a separate likelihood factor; the physical-pair clustering prior
(externally supplied model, switchable off, reported separately from
`p_sameq`).

---

## 6. Output contract

Every scored row carries all of this. `PairScore` in `score.py` is the
definition; do not return a bare probability.

```
candidate_id, primary_id, z_primary, ref_mag, photometric_system
loglike_qso_zprimary        p(colours | Q, z0)          evidence, not probability
loglike_bkg                 p(colours | background)
log_bayes_factor_qz_bkg     the most robust number here
p_zmatch_given_qso          conditional on being a quasar at all
z_phot_mode
log_lambda_sameq, log_lambda_fieldq, log_lambda_bkg
log_r_per_unit_z            window-free evidence; RANK ON THIS
reference_band, bands_used, surveys_used    what the densities were conditioned on
dz_match_eff                the window area that turns R into p_sameq
p_sameq_vs_bkg              same_z vs every non-quasar (bkg + out); ignores field quasars
p_sameq                     = exp(log_r_per_unit_z) * dz_match_eff
qso_ood_sigma               distance to the nearest component of the z0 slice
qso_ood_sigma_any_z         distance to the nearest quasar component at any z
bkg_ood_sigma               distance to the nearest background component used
loglike_outlier, outlier_fraction, log_lambda_out, p_outlier
                            the unmodelled hypothesis; NaN without an outlier model.
                            With one, loglike_bkg is the whole field density
background_local_weight     0 => the score came from the pooled model
background_density_level    0 local, 1 parent, 2 global
n_bands_used, status, quality_flags, model_manifest_id
```

`status` values in use: `ok`, `insufficient_photometry`,
`no_prior_posterior_unavailable`, `qso_prior_empty_at_this_magnitude`,
`primary_z_outside_model_support`, `window_outside_model_support`,
`blended_not_scored`. Add to this list
rather than returning a silent number. Quality flag `outside_both_models` is
set only when the caller passes `ood_flag_sigma` (no default: rule 2).

---

## 7. Testing

`pytest -q` runs the suite. New numerical code needs a test that checks it
against an *independent* route — quadrature, Monte Carlo, or a closed form —
not against its own output.

Existing patterns to follow:

| file | what it pins down |
|---|---|
| `test_gaussmix.py` | analytic identities vs `scipy.integrate.quad` and `multivariate_normal` |
| `test_xd.py` | deconvolution recovers the *intrinsic* width where a plain GMM recovers the broadened one |
| `test_features.py` | Jacobian covariance vs Monte Carlo flux realisations |
| `test_score.py` | posterior normalisation, prior-shift invariance, system-mismatch refusal, the velocity-window limitation |
| `test_multisurvey.py` | conditioning against closed-form Gaussians, luptitude covariance vs Monte Carlo, all 127 survey subsets, band layout |
| `test_multisurvey_priors.py` | priors vs stored counts, vs the archived prior, and across surveys' r bands; vectorised distances vs direct solves |
| `test_multisurvey_outlier.py`, `test_outlier.py` | the unmodelled term vs scipy, Schur-complement dominance, eta = 0 identity, far outliers |
| `test_shipped_models.py` | the README example on the one model; the three files are consistent |
| `test_archived_original.py` | the retired model's benchmark numbers, from `models/archive/` |

`filterwarnings = ["error::RuntimeWarning"]` is set on purpose: an overflow in
an exponential is a bug, and it caught one during development. Do not relax it.

Two tests document scientific facts rather than code behaviour, and must not be
weakened to make a change pass:

- `test_a_quasar_at_the_wrong_redshift_is_not_evidence_for_a_pair`
- `test_a_velocity_window_is_far_narrower_than_any_colour_redshift`

---

## 8. Data on WSDB (verified 2026-09-18)

| Purpose | Table | Note |
|---|---|---|
| Quasar training | `desi_dr1.zpix` ⋈ `desi_dr1.photometry` on `targetid` | 1,645,842 with `spectype='QSO'`, `zwarn=0`, `zcat_primary`; LS DR9 *grz*+W |
| Independent quasars | `sdssdr16qso.main` | 750,414; `psfflux`, `psfflux_ivar`, `extinction` are *ugriz* arrays |
| Background, candidates | `decals_dr9.main` | `release` 9010 south, 9011 north |
| Deeper, adds *i* | `decals_dr11.main` | `release` 11010 south, 11011 north |

DESI quasar targeting uses a random forest on these same Legacy Surveys colours,
so the training set is *p(colours | Q, z, selected by DESI)*. Report held-out
likelihood separately for the SDSS-selected sample, whose channels differ, and
treat a large gap as a selection-bias warning rather than a curiosity.

---

## 8a. Before reporting anything done

Open the artifact and check it against the request, verbatim. Scripts that
*select* things -- which objects, which fields, which models -- are science
code, not presentation: verify the selection is what was asked for, not merely
that the code ran without error.

Failures from 2026-09-19 that this rule would have caught: quasars scored
against a stellar model fitted 159 degrees away; five "random" field objects
all drawn from one 0.5 degree cone; axis limits that silently dropped 29% of
the plotted population.

Cache fitted models to disk (`--refit` to rebuild). Re-running minutes of EM to
move a text label wastes time and invites stale-cache errors when the
underlying selection changes.

### 8b. Never commit behind a pipe

`python -m pytest -q 2>&1 | tail -1 && git commit ...` commits when **tail**
succeeds, which is always. On 2026-09-20 a failing test went into `495731b`
this way. Run the suite to a file or capture `$?` / `${PIPESTATUS[0]}` and
branch on that; the commit step must see pytest's exit status, not the
pager's.

## 9. If you get stuck

- A number that looks too good is usually a leak: the same object in train and
  validation, or a positional duplicate.
- A likelihood that is enormous and negative is usually a feature-order
  mismatch between the model's `labels` and the candidate's.
- A posterior that will not move when the prior moves is a bug; a Bayes factor
  that *does* move when only the prior moves is a worse one.
- When the answer depends on a choice nobody has measured, measure it and put
  the number in `JOURNAL.md`. Do not pick a default and move on.

## 10. The model (2026-09-26)

`models/multisurvey.json` (run `22820c906d75`; quasar and joint field fits from
`d1f7f0f7cc78`, training config `configs/multisurvey_lsw.json`) is the only
model. It was promoted after it matched or beat the retired Legacy-only model
on that model's own validation, on identical rows (method note §10.3):
held-out same/wrong-z AUC by log R 0.828 vs 0.815, quasar/star 0.982 vs 0.988.

- **41 bands**, each in its own system: SDSS ugriz; Legacy DR9 south and north
  g, r, z, W1, W2 (the forced unWISE fluxes; release 9010/9012 south, 9011
  north); AllWISE W1–W4; PS1 grizy; NSC ugrizy+VR; SkyMapper uvgriz; VHS YJHKs.
  **Legacy forced W1/W2 and AllWISE W1/W2 are different bands; never relabel
  one as the other.** Leaving the Legacy ones out cost 0.11–0.13 in AUC.
- Native luptitudes of observed (not dereddened) photometry; AllWISE/VHS keep
  Vega. Do not apply a dereddening transform to this model's inputs.
- Scoring conditions the joint noisy distribution on the first observed band
  in the saved priority, then marginalises missing bands. Two measurements
  supply one colour. Do not multiply separate survey Bayes factors.
- **Priors are per reference band**, per unit native luptitude of that band,
  counted inside that survey's footprint, for objects with that band measured.
  One completeness constant (C = 2.30) ties them to the archived original
  prior. A prior must carry the reference label and `transform_id`; a new
  transform needs new priors.
- **Dedicated field fits** apply only when every observed band is inside their
  band set (smallest first): Legacy-south g, r, z and g, r, z, W1, W2. Refit
  them, and then the outlier term, whenever the joint model is retrained.
- **The unmodelled term** must dominate every component of every density the
  scorer uses, including the dedicated fits (embedded). `kappa > kappa_min` is
  enforced on construction; `fit_multisurvey_outlier.py` rebuilds the envelope.
- Preserve the quasar holdout blocks. Field validation reserves whole cones;
  component selection uses further blocks/cones inside training. Rare field
  measurement patterns are retained with abundance-restoring weights.
- Training writes a candidate (`models/multisurvey_lsw.json`); promote it to
  `models/multisurvey.json` only after `validate_pairs_multisurvey.py` and
  `validate_multisurvey.py` pass, and regenerate priors, outlier, figures,
  examples and the method note from the promoted file.
- Fits stop at a 150-iteration cap (1 of 43 quasar slices and the joint field
  fit do not reach 1e-5). Validation reports the saved fits at that point.
- Clean-companion requirements still apply. A low-resolution AllWISE source
  beside a bright primary does not certify a separated measurement.

**The archive.** `models/archive/original_legacy_south/` (the retired model),
`multisurvey_37band_20260921.json` and its priors (before Legacy W1/W2),
`multisurvey_joint_20260921.json` (initial joint-only). Never score science
candidates with them, never move them back to top level, never delete them:
the model's holdout blocks and prior normalisation come from the original
bundle, and `tests/test_archived_original.py` pins its benchmark numbers. A
moved input file is accepted by `same_training_config` only if its content, or
where the old path is gone its name, is unchanged.
