# Plan: morphology selection, field model, sky dependence, survey depth (2026-09-27)

> **Superseded for the baseline by `docs/BASELINE_PLAN.md`** (return to the original XDQSO-style design, dereddened). Kept as the record of the first attempt; §8 lists what went wrong.

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
   retention after the PSF cut. Later spatial models must improve a
   predeclared held-out objective while still passing the baseline gates (§7).
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
   P(not detected | f) = Φ((L − f)/σ). Coverage and depth are necessary but not
   sufficient: an absence can also come from confusion, blending, a quality
   rejection, a deblending failure, association policy (nearest match within
   a radius for quasars, greedy one-to-one for the field; competing matches
   are left unassigned) or variability. The states to keep apart are: outside
   coverage, masked/unusable, valid forced measurement (negative fluxes
   kept), covered non-detection, ambiguous association, and unknown. Until a
   survey has this, every absence is treated as missing; this is an
   approximation whose effect is measured, not a guarantee of no bias. The baseline is
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

- v1 keeps treating them as missing (marginalised). This is an
  approximation: when a band is missing *because* the source is faint, the
  result can be biased, and neither the size nor the sign of that bias is
  guaranteed. It has to be measured, for example by deleting bands from
  objects that have them. The baseline does not depend on it.
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

**This table is superseded by §7**, which puts the baseline first and makes
it a gated deliverable. Steps 2–3 above (new cell-spread field sample,
per-cell adaptation) start only after the baseline passes its gates. A later
spatial model must improve a predeclared held-out objective, with
uncertainty, while still passing the baseline gates. It does not have to
improve every metric.

## 7. The baseline: specification and gates

The immediate deliverable is a global Legacy-PSF model. It has no sky
dependence, uses the single reference band Legacy r, and uses the five
bands g, r, z, W1 and W2. The items below were added after Codex's second
review; the numerical tolerances are **judgement calls**, frozen before the
final test set is looked at.

**Population and domain**
- Legacy DR9, `TYPE = PSF`, where the r measurement is usable (valid ivar,
  nobs > 0, maskbits = 0). The other four bands may be missing or low-S/N;
  forced fluxes, including negative ones, are kept.
- Candidate domain 17 ≤ r < 22.5 (Legacy r, native luptitude) and quasar
  support z 0.15–4.35. Outside the domain the scorer refuses; it does not
  extrapolate.
- The fits use a margin (16–23.5). Truncation is checked, not assumed:
  compare the conditional densities near 17 and 22.5 against a fit to a wider
  cache, and on synthetic truncated data.
- **South and north are separate five-band fits** (they are different
  photometric systems), with south and north ownership as in the adapter
  (releases 9010/9012 south, 9011 north). The pair validator is extended to
  north, since today it builds south-band inputs only.

**Normalisation (C)**
- C comes from the all-source quasar population (reference and matched
  counts before the morphology cut), is frozen with its provenance, and is
  then applied unchanged to the PSF counts. The builder must not recompute
  reference / PSF-total, because that would undo the retention loss.
- Sampling weights are set on the draw before the morphology cut.
- Applying the south C to north is an assumption; a sensitivity test goes
  with it.
- `test_legacy_south_r_prior_reproduces_the_validated_original` (1 %
  against the all-source total) is replaced by a retention-aware identity:
  the PSF prior's integral equals C × the weighted PSF count / area, to a
  relative 10⁻³.

**Selection, partitions and provenance**
- One selection function, used everywhere. The test is identical decisions
  on shared fixtures, not identical catalogue memberships.
- The fit, selection, calibration and test partitions are frozen as saved
  ID lists and sky blocks before anything is fitted. The Student-t
  calibration set is reserved up front; today it is rebuilt from a seed and a
  budget argument (`fit_multisurvey_outlier.py:70–71`). The test partition is
  kept out of C, priors, K and start selection, and Student-t tuning.
- Every artifact (model, prior, outlier) records the selection, domain,
  sample and partition IDs. Loading refuses mixed bundles even when the
  transform matches. Bundles are versioned, and a manifest is switched only
  after validation passes.

**Counts and area**
- Per-magnitude-bin counts come from a server-side aggregate over the
  sampled area with exactly the downloaded sample's cuts (PSF, release, mask,
  usable r, known quasars removed). Aggregate counts must equal the
  downloaded counts on test regions, bin edges and duplicates included.
- Area comes from Legacy randoms with the same mask.
- The baseline uses λ_B = Σ_B(m) p_B(colours | m): the counts give Σ_B and
  the mixture gives the conditional colours, so there is no second implied
  magnitude distribution competing with the counts. (For the later per-cell
  model this factorisation, or fitting the weights to the counts, is chosen
  explicitly.)

**Fitting**
- K ∈ {8, 16, 32, 64}, three starts each. The rule is the best conditional
  held-out likelihood among fits that pass the convergence rule. If the
  largest K wins, the grid is extended.
- A fit that does not converge cannot be published. Today `converged=False`
  is recorded and ignored, and there is one start per K.

**Gates (all must pass)**

| gate | criterion |
|---|---|
| selection/provenance | 0 selection disagreements on fixtures; 0 partition overlaps; mixed bundles and unknown morphology refused |
| normalisation | C unchanged by the PSF cut; prior integral = C × weighted PSF count / area to 10⁻³ |
| counts | aggregate = downloaded counts exactly on test regions; areas agree within randoms noise |
| convergence | continuing a fit changes held-out mean log density < 0.01 nat/object and median \|Δ ln R\| < 0.02; best starts agree similarly |
| count prediction | on held-out regions: ≤ 5 % pooled, ≤ 10 % per well-populated bin, errors from spatial blocks |
| tails | predeclared model-density regions; combined field + Student-t; fail if observed/expected differs by > ×2 **and** > 3σ where ≥ 20 are expected |
| ranking | on identical eligible PSF rows vs the current model: lower 95 % spatial-bootstrap bound on ΔAUC > −0.01 for same-z/wrong-z, stars, PSF galaxies; contamination at a fixed quasar retention; north/south and magnitude strata reported |
| morphology loss | quasar retention after the cut by z, magnitude, separation, with intervals; galaxy-rejection results reported on PSF galaxies only |
| numerics | negative fluxes kept; missing-band patterns exercised; p_sameq = e^{ln R} Δz_eff identity; a band with vanishing information changes ln R by < 10⁻³ (synthetic) |

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

## 8. Baseline progress (2026-09-27)

**Built and tested.**
- **Selection:** `src/qso_pcolor/legacy.py`, one selection used for training, counts and candidates.
  - Hemisphere is taken from the position (DESI rule). WSDB holds both north and south rows in the overlap stripe, and north-release rows with `nobs_r = 0` at dec ≈ 2°.
  - Morphology comes from Tractor `TYPE`.
  - Usable area comes from the brick MASKBITS and nexp-r images. On a test cone the pixel condition reproduces the catalogue's `maskbits == 0 & nobs_r > 0` for all 28,117 sources.
- **Bundles:** `src/qso_pcolor/baseline.py`, with a manifest of hashes and bundle and selection ids. Mixed or tampered bundles are refused.
- **Sample:** `scripts/build_legacy_baseline_sample.py`.
  - Field: 304 cones of 0.3°. The footprint at |b| ≥ 25° covers 124 nside-4 cells (18,288 deg²), not the ~60 estimated in §3.
  - Quasars: 82,489 targets, 82,333 matched to a DR9 row of their own hemisphere.
  - Companions: 568,287 matched.
- **Fitter and validation:** `scripts/fit_legacy_baseline.py` and `scripts/validate_legacy_baseline.py`.

**First bundle (`e71820f57f83`, test cells, against the current model on identical PSF rows).**

| gate | result | pass? |
|---|---|---|
| same-z vs wrong-z, ln R AUC | 0.849 vs 0.822 | ✓ |
| quasar vs star | 0.986 vs 0.990 | ✓ |
| quasar vs PSF galaxy | 0.664 vs 0.817 | ✗ |
| uninformative band | moves ln R by up to 0.19 | ✗ |

Three causes were found:
1. **Unrecognised quasars in the PSF field sample.** The recognised fraction per cone has a median of 0.37 and is zero in 36 % of south cones. Refitting on covered cones only raises quasar ln BF by +0.9 and leaves galaxies (+0.01) and stars (+0.10) unchanged. Fix: each field source is weighted by 1 − P(unrecognised quasar), and Σ_B subtracts (1 − κ) Σ_Q. The prior-based and model-based expectations of the unrecognised quasars agree to 13 %.
2. **The fixed covariance floor 10⁻³ mag².** It makes every colour direction at least 0.045 mag wide. The held-out score at K = 32 is −1.978, −1.786 and −1.759 for floors of 10⁻³, 10⁻⁴ and 10⁻⁵. Fix: the floor is chosen on the select cells.
3. **Student-t with noise added to its scale.** Fix: the exact scale-mixture convolution, which reproduces the closed form to 3 × 10⁻¹⁴. Opt-in, so the shipped model is unchanged.

**Other measurements.**
- **PSF retention** among quasars that pass every other condition: 0.18 at z < 0.5, 0.73 at 0.5–1, and 0.98–1.00 above z = 1.
- **Completeness constant** from the all-morphology south quasars: C = 2.256. The current model's 2.37 came from a different matching and mask setup.

**Open.**
- The north has no held-out field cones, because the archived test blocks are all in the south. It has only 39 same-z companions in test cells.
- Field counts in the 15 south test cells are 1.25 times the prediction, 95 % interval 0.83–1.84. This spatial scatter is expected for a global model.

## 9. Ship criterion (after Codex's pragmatic review, 2026-09-27)

Goal: a sound, tested baseline, without over-optimising.

**Hard requirements**, checked by `scripts/validate_legacy_baseline.py`, which exits non-zero if any fails:
- selection, partition isolation and bundle integrity (unit tests);
- the normalisation identity;
- the uninformative-band numerics (< 10⁻³);
- all fits converged, and continuing the final field fit under its own non-quasar weights for 200 iterations changes the select score by < 0.01 nat/object and the test companions' median |Δ ln R| by < 0.02;
- ranking on identical test-cell PSF companions against the current model: the lower 95 % block-bootstrap bound of ΔAUC is above −0.01 for same-z vs wrong-z, quasar vs star, and quasar vs PSF galaxy;
- a coarse tails check against draws of the full observable model (non-quasars, unmodelled term, unrecognised quasars): no bin with ≥ 20 expected sources differs by more than a factor of 2 **and** more than 3σ.

**Reported, not gated:**
- field counts against Σ_B: the spatial scatter of a global model over 15 test cells is not a pass/fail;
- agreement between starts;
- whether the largest K won: K is capped at 128;
- north ranking: only 39 held-out same-z companions.

**North** ships marked `provisional` in the manifest. A validated north needs whole north cells reserved and the north refitted without them. The target draw and the parent counts use the archived held-out blocks, so relabelling cells is not enough. This is a separate, later step.

**Assumptions stated, not tested here:**
- the completeness constant C is scientifically accurate;
- recognition of quasars does not depend on magnitude (κ per cone). Uncapped κ is recorded.

The prior-based and model-based counts of unrecognised quasars both use Σ_Q and κ, so their agreement is not an independent check.
