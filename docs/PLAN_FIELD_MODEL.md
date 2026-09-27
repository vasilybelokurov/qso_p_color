# Plan: morphology selection, field model, sky dependence, survey depth (2026-09-27)

Follows `docs/FIELD_MODEL_FINDINGS.md`. Drafted by Claude, critiqued by Codex
(high effort, read-only); Codex's code citations were checked against the
source (table at the end). Clustering is parked and not part of this plan.

## Summary

- **Morphology** becomes a declared training option, `selection = "point" | "all"`,
  stored in the model and applied identically to the quasar sample, Σ_Q, the
  field sample, Σ_B and the candidates. v1 policy: Legacy DR9 `TYPE = PSF`.
  A candidate without Legacy morphology is not eligible for the point model.
- **Architecture is as in the retired model:** one global HEALPix-based field
  model is the default; a locally tuned refit around one candidate is an
  option. The global model is built from a new area-based field sample; per
  cell it adapts the counts and the mixture weights, with shared component
  shapes and shrinkage towards parent and global.
- **Field fit:** enough components (chosen by held-out likelihood *and*
  count accuracy per band), several starts, converged.
- **Depth:** the reference band is the deepest available r-like band, by a fixed
  survey precedence (Legacy first). This removes most of the SDSS-reference
  offset, but the real fix is the refit, and reference invariance becomes a
  regression test with a tolerance. Catalogue non-detections stay
  uninformative in v1. The "0 ± σ_lim" pseudo-measurement is dropped: it is
  statistically wrong (details in §4).

## Decisions (2026-09-27, after discussion)

1. **The baseline comes first:** a Legacy-PSF-only model in Legacy DR9
   g, r, z, W1, W2, built and tested thoroughly before any other infrastructure.
   It includes PSF quasars with their Σ_Q, PSF field sources with their Σ_B,
   component number chosen on held-out data, fits run to convergence, and a
   refitted Student-t term. It is validated with the pair head-to-head, star
   and galaxy rejection, counts per magnitude, tail counts, and quasar
   retention after the PSF cut. Everything after the baseline must beat it on
   held-out data.
2. **Morphology rule v1:** Legacy DR9 `TYPE = PSF`. `unknown` never counts as
   point.
3. **HEALPix.** The earlier design used local cells at nside=8 with parents at
   nside=2 and n₀ tuned over 100–3000 (`configs/example_ls_dr9.yaml:56–60`);
   what shipped was nside=1, a single footprint average
   (`background_south_global.json`), plus a local refit in a 0.5° cone. The new
   plan uses nside=4 cells (192 in the whole sky, 214.9 deg² each; 112 with
   centre at |b| > 20°; about 60 inside Legacy) under nside=2 parents, and
   moves to nside=8 if held-out cells show structure that nside=4 misses.
4. **Cones and counts.** A cone is a small circle (0.3°, 0.28 deg²) in which
   every catalogue source is downloaded, with its full cross-matched photometry
   and its usable area. There are 2–3 separated cones per cell. The cones
   supply the magnitude and colour distribution, which sets the per-cell
   component weights. **Counts per (cell, magnitude bin) come from a
   server-side aggregate query over the whole cell where possible**, which
   gives exact counts instead of cone-sampled ones. Its speed on Legacy DR9 is
   not yet measured, so one cell is timed first; the cone-sampled count is the
   fallback. Counts and area must use the same selection and the same mask
   (Legacy randoms).
5. **Non-detections.** A survey reports no object for one of two reasons:
   (a) the area was not covered (outside the footprint, a gap, or masked),
   which is common and uninformative; or (b) the area was covered and the
   object is too faint, which is informative. Using case (b) requires a
   coverage map per survey, to tell (a) from (b), and the local depth, to give
   P(not detected | f) = Φ((L − f)/σ). Until a survey has both, every absence
   is treated as (a): this loses information but adds no bias. The baseline is
   unaffected: Legacy forced photometry gives every source g, r, z, W1, W2
   fluxes. AllWISE is the first candidate for a detection model, because it
   covers the whole sky and almost every absence is case (b).

## 1. Morphology selection

What the user asks for: `train(..., selection="point")`, which gives a model
for PSF-only sources.

For a declared selection S the intensities become
λ_H = Σ_H(u_a | S) · p(u_rest | u_a, H, S), for H = quasar and field. Both
hypotheses must be selected. If only the field is filtered, the alternatives
change while the quasar hypothesis does not.

- **Policy v1: Legacy DR9 `TYPE = PSF`** (south and north), where a Legacy
  measurement exists. Each object gets one of `point`, `extended`, `unknown`;
  `unknown` never counts as point. Other surveys come later as separate,
  named policies. Each would use a fixed precedence, with a quality criterion
  per classifier: SDSS `type`, PS1 PSF − Kron, NSC/SkyMapper `class_star`,
  VHS `mergedclass`. The earlier rule "the deepest resolved survey decides"
  is dropped because it mixes depth, seeing and classifier reliability.
- **Quasar side:** refit the quasar mixtures on PSF quasars, or show that the
  colours at fixed (z, u_ref) do not depend on the cut. Σ_Q is counted on the
  selected quasars. The completeness constant C stays tied to the all-source
  reference. The point Σ_Q must **not** be renormalised back to the all-source
  total, because that would undo the measured 6–9 % loss.
- **Candidates:** the scorer refuses a candidate whose selection flag does not
  match the model's selection, or whose flag is `unknown`.
- **Data:** a WSDB pull that adds `type` (and `shape_r`) from the Legacy DR9
  tables. `Photometry` gets a per-object morphology field. The current queries
  and data class carry no morphology.
- **Tests:** the same policy function gives identical accepted IDs in training,
  in the counts and for candidates. Measure quasar retention by z, magnitude
  and separation from the primary. Quasar/galaxy AUC for point vs all.
  Mismatched and unknown selections are refused.

## 2. Field model quality

- **Fit domain.** Keep a broad parent cache. Restrict the fit by a cut on the
  reference magnitude itself, for example 16 < u_ref < 23.5 for Legacy r. A
  cut on the conditioning variable leaves p(u_rest | u_a) unchanged inside the
  cut. A cut on a different band would bias it.
- **Capacity.** K in {8, 16, 32, 64}, three or more starts each, run to
  convergence. Choose K on held-out fields using two criteria: the conditional
  colour likelihood (the only criterion today) and the predicted-vs-counted
  magnitude distribution in each reference band. The second criterion catches
  the SDSS r error, which the first misses. A full 41-band covariance has about
  900 parameters per component, so we check the effective co-observed counts
  for each band pair before trusting a large K. If support is thin, we compare
  against a structured or lower-dimensional model.
- **Dedicated band-set fits** (Legacy grz, grzW1W2) stay for now. Two problems
  need checking: they are trained only on rows with every band measured but are
  applied to rows with fewer bands, and one extra observed band sends a row back
  to the joint fit.
- **Student-t term** is refitted on a separate calibration partition after the
  new field model is fitted.
- **Tests:** counts by magnitude and band pattern (predicted vs counted), tail
  counts with intervals (§2.2 of the findings), repeat-start stability, and a
  noisy extra band that should not change the score.

## 3. Sky dependence: one global HEALPix model, plus an optional local refit

**Default: the global model.**

λ_B(u, s) = A(s) Σ_k w_k(s) p_k(u)

- The components p_k (in magnitude and colours) are fitted globally.
- The amplitude A(s) and the weights w_k(s) are set per HEALPix cell, with
  shrinkage cell → parent → global.
- Because magnitude is a coordinate of the components, the cell weights also
  set the local magnitude distribution.
- Pooling strengths are tuned separately for the counts and for the weights,
  on held-out sub-regions *inside* populated cells (does the model adapt
  locally?) and on held-out whole cells (does the fallback work?). The old
  tuner used nside=2 folds, which removed the parent and child fits it was
  meant to tune.
- Full per-cell mixtures are a later comparison, adopted only if the data
  support them.

**Option: the local refit.** For one candidate, refit A and w_k in its own
cone (candidate excluded), shrunk towards its cell. Adopt it only if it
improves the score on separate neighbouring data.

**New field sample.** The present 24 cones are centred on quasars and cannot
support per-cell estimates. The new sample:

- HEALPix nside=4 (215 deg² per cell) as sampling strata. They are not a
  claim about the scale of real sky structure.
- 2–3 separated cones of radius 0.3° (0.28 deg² each) per cell, over the Legacy
  footprint at |b| > 20°. Rough count: about 60–70 cells × 3 ≈ 200 cones ≈
  60 deg² in total.
- The sampling additionally stratified by latitude, extinction and depth.
- Usable area measured from Legacy randoms, masks included. Today a cone
  counts its full nominal area if the survey has even one detection in it.
- Held-out cells chosen before anything is fitted.

**Tests:** synthetic fields (homogeneous, and with a known gradient) recovered;
independent count prediction per cell; validation loss that actually varies
with the pooling strength; continuity at cell boundaries; an empty cell falls
back correctly.

## 4. Surveys of different depth

**The reference is the deepest available band.** Fixed precedence:
Legacy r (south, then north) > PS1 r ≈ NSC r > SDSS r > SkyMapper r > VHS J >
AllWISE W1. It is chosen from survey coverage, never from the object's own
S/N, because choosing by S/N makes the choice depend on the noise. It is a
fixed ordering and does not measure local depth. Why it helps:

- Legacy measures more of the field (18,945 vs 12,303 deg⁻² at 17 ≤ r < 22.5),
  so Σ_B,a is most complete in that band.
- Legacy has forced photometry in g, r, z, W1 and W2. Faint objects get
  low-S/N fluxes, including negative ones, instead of going missing, and the
  model handles those correctly as they are.

**What it does not fix.** The ~0.9 nat SDSS offset comes from Σ_B,a / m_a
differing between bands: the counts and the model's marginal disagree.
Changing the reference hides this for objects with Legacy data; it does not
remove it. The refit of §2 (and, later, a common intensity whose marginals
match the counts in every band) removes it. The invariance check becomes a
regression test. It compares ln R for the same objects with the anchor
changed, requires |median Δ ln R| < 0.2 for field sources, and does not
require the Bayes factors themselves to be invariant.

**Missing bands** are still marginalised. This is exact only if a band is
missing for reasons unrelated to the source (it lies outside the footprint).
It is not exact when the band is missing because the source is too faint.

**Non-detections in catalogue surveys** (AllWISE, VHS, SkyMapper, SDSS):

- v1 keeps treating them as uninformative. This is conservative: it loses
  information but does not bias anything.
- "0 ± σ_lim" is dropped. For a limit L and Gaussian errors,
  P(not detected | f) = Φ((L − f)/σ), whereas a fake zero flux gives a
  likelihood of exp(−f²/2σ²). Example: at L = 5σ, a source of true flux
  f = 2.5σ goes undetected with probability Φ(2.5) = 0.994, but the fake zero
  gives it exp(−3.125) = 0.044 of the peak likelihood. Using the fake zero in
  both training and scoring does not make it correct.
- Later, a proper detection model, one survey at a time. AllWISE first, only
  if it adds information beyond Legacy's forced W1/W2. It needs a probit
  selection term, a coverage product, and training that accounts for the
  selection (so the term is not counted twice). Outside-footprint, masked and
  truly undetected cases are kept distinct; unknown coverage stays unknown.

## 5. Order of work

| step | deliverable | tests | rough cost |
|---|---|---|---|
| 0 | Legacy r first in the priority; Σ_B without reserved cones; model/prior/outlier files carry selection and sample IDs and refuse mismatches; bundle published together | 127-subset and pair validation unchanged or better; invariance rerun; mismatch refusal | ½ day |
| 1 | WSDB pull with Legacy morphology; `selection` option end to end (quasars, Σ_Q, field, Σ_B, candidates) | identical IDs across paths; quasar retention; quasar/galaxy AUC point vs all | 2–4 days |
| 2 | New area-based field sample (nside=4 strata, randoms for area); converged global refit with K selection; Student-t refit | counts per band, tails, repeat starts, invariance tolerance, full pair validation | 3–5 days + WSDB time |
| 3 | Per-cell A(s), w_k(s) with tuned shrinkage; optional local refit | synthetic gradients, per-cell count prediction, held-out cells and sub-cells, low-latitude check | 3–5 days |
| 4 | Method note: all figures regenerated from the new model | — | 1 day |
| later | Detection model for one catalogue survey; measured completeness for Σ_Q | synthetic censoring recovery; calibration by depth | days per survey |

A cheaper alternative to test against before steps 2–3 grow in scope:
a Legacy-PSF-only model with a converged global fit and correct counts, and no
per-cell adaptation. If the per-cell model does not beat it on held-out cells,
this simpler model ships.

## 6. Codex's points, checked

| Codex point | verdict | evidence |
|---|---|---|
| Retired Σ_B used local → parent → global fallback with no shrinkage; only the colour densities were shrunk | confirmed | `src/qso_pcolor/priors.py:113–126` (first level with data wins); `background.py:208–230` (n/(n+n₀) mixing) |
| Selection must be applied to the quasar side too; do not renormalise the point Σ_Q back to the all-source total | confirmed | `scripts/build_multisurvey_priors.py:170–183` normalises to an archived all-source prior |
| Current data carry no morphology | confirmed | `multisurvey_data.py` queries only flux/ivar; `Photometry` = flux, variance, bands |
| K is chosen by conditional colour likelihood only, which cannot see marginal errors | confirmed | `scripts/train_multisurvey_model.py:40–47` |
| Dedicated fits are trained on fully observed rows but routed to partially observed ones; one extra band sends a row back to the joint fit | confirmed | `fit_background_marginal.py:104`; `multisurvey.py:171–181` |
| Field cones are centred on quasars and stratified by declination, not drawn by area | confirmed | `scripts/build_multisurvey_sample.py:157–163` |
| Area counts a cone's full nominal area if the survey has any detection in it | confirmed | `build_multisurvey_priors.py:145–147` |
| The old n₀ tuner's nside=2 folds cannot identify the pooling strength | confirmed in code (the effect was noted earlier) | `background.py:613` |
| "0 ± σ_lim" is wrong | agreed; the numbers checked (Φ(2.5) = 0.994, e^{−3.125} = 0.044) | §4 |
| "The deepest survey decides morphology" is underspecified | agreed | fixed precedence with a quality criterion per classifier |
| The 127-subset validation runs without priors or the outlier term | confirmed | `scripts/validate_multisurvey.py:134` calls `model.score` with neither; to be fixed in step 0 |
| Cost estimates in days | accepted as rough | not measured |
