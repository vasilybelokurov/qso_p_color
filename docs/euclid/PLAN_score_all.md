# Scoring every Euclid point source — plan (v2, 2026-11-19)

Status: REVIEWED PLAN, nothing fitted or implemented. Two rounds with two independent reviewers (Codex via the
ChatGPT CLI, and a Fable agent; both read the repository). v1 -> v2 changes are listed in section 9. Every factual
claim below was checked in the code or in the stage-1/2 outputs unless marked ASSUMED or UNVERIFIED.

## 0. Goal and target population

Give every source in the Euclid point-source deployment selection one record: `p_quasar` summed over redshift,
`p_outlier`, sensitivity scenarios, and flags. "Every" means a record for every row; a finite probability for every
row is the goal of the last phase, not of the first. A score that is mostly prior must say so. Promoted bundles,
existing roles and existing validation outputs are not modified.

Target population (PI decision D1): MER sources passing the existing point-source deployment selection
(`scripts/euclid/cones.py`, `src/qso_pcolor/euclid_bundle.py`), any VIS, any S/N, any subset of Y/J/H. Rows removed by
the cleaning cuts (flag bits, isolation, VIS < 18, |b| < 25) stay out and are counted separately.

## 1. Where refusals come from (stage 1, model A, Euclid test cones; B similar)

24,000 selected background point sources, median VIS 24.3, 66% at VIS >= 23. Legacy r: absent for 9,555 (40%),
present at S/N < 10 for 5,491 (23%), usable for 37%.

| VIS | n | scored, Euclid-only | main refusal |
|---|---|---|---|
| 18-22 | 5,551 | 60% | `qso_support_rejected` 2,004 |
| 22-23 | 2,558 | 35% | `legacy_r_outside_training` |
| 23-24 | 2,959 | 0.6% | `legacy_r_outside_training` |
| 24-27 | 12,931 | 0% | `legacy_r_outside_training`, `fallback_too_faint` |

2,246 spectroscopic test QSOs: 7.5% unranked, 136 by `legacy_r_outside_training` (median VIS 22.8).

| Refusal | Cause (verified) | Treatment |
|---|---|---|
| `qso_support_rejected`, `outside_both_models` | wrappers clear pair probabilities and `p_outlier` but keep the four intensities (`unified.py:206, 253-260`); support is tested at the supplied z0 only | catalogue validity separated from pair-ranking eligibility (Phase 2) |
| `primary_z_outside_model_support`, `window_outside_model_support` | the total integral uses window nodes set by z0 (`score.py:448`) | z-free entry point (Phase 2) |
| `too_faint` with a weak measured Legacy r | fallback needs Legacy r absent (`fallback_reference.py:79`, gate `:162`), and it overwrites South-r flux and variance with the synthetic latent (`:198-200`); also the first available fallback band is chosen before its S/N is checked | new integral that keeps the weak-r likelihood (Phase 4) |
| `insufficient_nisp`, `no_fallback_colour` | the integration window is built from VIS-Y | VIS-referenced model (Phase 4) |
| `fallback_too_faint` | S/N policy; luptitude covariance is first-order (`multisurvey.py:50`) | after the low-S/N benchmark (Phase 3) |
| `legacy_r_outside_training` | no population model beyond Legacy r ~ 23.5: Legacy Sigma_B(r) turns over at r ~ 25 (incompleteness); runtime QSO prior is zero outside its table (`priors.py:310`) though the table builder extrapolates a slope (`build_qlf_prior.py:45`); thinning f_k(r) extrapolated; catch-all fraction clamped (`multisurvey.py:675-676`) | new population model (Phase 4) |

## 2. Why the faint model is referenced in VIS (design 2B; was 2A in v1)

Keeping latent Legacy r and fitting the background on all MER sources (2A) is not identifiable: Legacy r is missing
because the source is faint in r (censoring, not missing at random), and for sources without informative Legacy r,
any change of the latent-r distribution that keeps the Euclid marginal leaves the likelihood unchanged
(`projected_xd.py:63`). VIS counts fix the marginal, not Sigma_B(r) and the r-Euclid relation separately. 2A would
also need a new trainer: projected XD fits a normalised density (`projected_xd.py:231`) with bins assigned by measured
Legacy r (`fit_stage2.py:155`). 2A stays an option only with an identifying constraint (forced Legacy r photometry, or
an explicit Legacy detection model) and a demonstrated insensitivity to it.

2B: in the Euclid-only path the reference magnitude is VIS.
- Sigma_B(VIS, l, b): counted from the selected population per unit usable area.
- Background colour model: XD on Euclid colours (VIS-Y, VIS-J, VIS-H) in VIS bins, fitted on the total selected
  population, with the QSO intensity included and held fixed, so unknown QSOs are not absorbed into the background.
  The label is "not a quasar" (faint compact galaxies dominate beyond VIS ~ 23.5), not "star".
- Sigma_Q(z, VIS): the QLF mapped through each redshift slice's p(VIS | r, z) from the stage-1 joint model (not a
  fixed colour shift), times c_Q(z, VIS). c_Q beyond VIS 23 is unmeasured and is carried as a sensitivity scenario,
  UNTESTED.
- QSO colours: model B continues by construction; model A beyond its training range is flagged
  `qso_colour_extrapolated` (a clamp in the 36-d slice models is ASSUMED, to verify). A and B reported separately.
- Selection: the population is fitted directly, so the existing thinning factors (`multisurvey.py:200`,
  `euclid_bundle.py:185`) and the separate c_Q on the QSO prior (`score.py:541`) are not applied again; the catch-all
  gets a declared normalisation for this population.
- The existing Legacy-r path stays unchanged for sources with usable Legacy r; the overlap is tested (T9).

## 3. Phases (in order)

**Phase A — data and roles.**
- An immutable experiment manifest: cells used so far for fitting or decisions (incl. the current Euclid test
  cells, spent by the stage-2 diagnosis), and newly allocated untouched evaluation cells (D3). Caches keyed by
  coordinates and manifest hash; `fetch_cones.py:49` currently skips any existing filename without checking
  coordinates, so this needs a fix.
- Download fit-role cones: none exist locally (40 test, 12 select, 12 calib; `sky_roles.py` creates only those
  roles). Size from Phase B's requirement; target >= 30 deg^2 of fit cones (ASSUMED adequate; to be set by B).
- Check whether Euclid deep fields and faint spectroscopic or X-ray AGN samples are available (UNVERIFIED).

**Phase B — census** (select cones until fit cones exist): area-normalised counts by VIS (0.25 mag), S/N per band,
Legacy detection fraction and Legacy r S/N by VIS, `point_like_prob`, usable area and the VIS where counts turn
over. Paired standard errors of each planned metric for the exact cohorts, used to set the margins in section 4.

**Phase C — low-S/N likelihood benchmark.** Labelled bright QSOs and stars degraded with the measured cone
error-magnitude relation, S/N 1-20. Luptitude-Gaussian vs flux-space likelihood; PIT against the true
(pre-degradation) flux. Where they disagree, the flux-space likelihood is used. Must finish before Phase D's fit.

**Phase D — catalogue scoring of bright rows, no refit.**
1. Catalogue validity separate from pair-ranking eligibility; support and outside-both become flags.
2. `p_quasar` and `p_outlier` recomputed from the four intensities; [p_Q, p_Q + p_outlier] labelled as an
   allocation bound conditional on the fitted intensities.
3. z-free entry point integrating over all slices with no z0-dependent window; log R only when z0 is supplied.
4. Sensitivity scenarios Sigma_Q x 0.5, x 2 (a uniform +-0.69 shift in logit p, described as such, not an interval).
Released only as a diagnostic export: about 80% of the target population is still unscored after it.

**Phase E — faint model (2B) and the weak-r integral.**
- Fit as in section 2 on fit cones, after Phase C.
- Euclid-only path: VIS-referenced, no VIS-Y window, no centre or tail-mass cut; numerical convergence (T3).
- Weak Legacy r: integral that keeps the measured South/North r likelihood; select the first fallback band that
  passes S/N; opt-in until T6 passes.
- Information measure I = logit p - logit p_prior(VIS, l, b), with the prior from the same selection and spatial
  model; flag `prior_dominated` when |I| < 1; `extrapolated` when > 1% posterior mass lies beyond trained ranges.
- Throughput and peak-memory benchmark on 10^6 rows before deployment.
Optional: pilot 2A with forced-photometry constraint against 2B on synthetic truth, if forced r is available.

## 4. Tests and acceptance (thresholds fixed before fitting; margins below 2 paired SE, measured in Phase B, are
declared unresolvable and widened before any fit)

| ID | Test | Pass |
|---|---|---|
| T1 | Records: one per input ID, order/batch invariant | 100% |
| T1b | Finite p per phase on the target population, denominators stated (D: VIS < 23 rows; E: all) | D >= 99.5%; E >= 99.9% |
| T2 | Promoted and stage-1 bundles: every numeric field, flag and eligibility unchanged on the parent panels | max abs diff < 1e-9 |
| T3 | Numerics: doubled grid, widened bounds; each integrated intensity, boundary mass and p | log-intensity diff < 0.01 and |dp| < 0.005 for 99%; every failure listed and flagged |
| T4 | z-free entry point: p_quasar identical across supplied z0 | < 1e-9 |
| T5 | Phase D on new cells, VIS < 23: old vs new on the identical all-rows cohort, old refusals at p = 0; paired cell bootstrap | AUC lower 95% bound > -margin; recall > -1 pp |
| T6 | Background p > 0.5 rate, unflagged rows (known/likely-QSO exclusions as `validate_stage1.py:129`), combined and Euclid-only modes | upper 95% bound of the change < +0.1 pp |
| T7 | Weak-r integral: paired masking (drop Legacy r from rows that have good r, then degrade it) | |dp| < 0.02 for 95%; AUC change > -margin |
| T8 | Phase C: PIT and 68/95% coverage of degraded vs true flux, S/N 1-20 | coverage within +-3 pp per S/N bin |
| T9 | Overlap VIS 21-23, rows with usable Legacy r: Phase E Euclid-only vs stage-1 Legacy path, identical cohort | AUC > -margin; recall > -1 pp; bkg < +0.1 pp |
| T10 | Background colour model: per VIS bin, predictive distribution of the Mahalanobis distance and 68/95% coverage of the 3-colour regions, select cones, all S/N, minimum 500 rows per bin | within +-3 pp |
| T11 | Counts per VIS bin on untouched cells vs model, cone-to-cone scatter and area uncertainty as the error | within 99% predictive interval |
| T12 | Sum p per deg^2 vs stellar density across cones, after depth/extinction adjustment; minimum detectable slope stated first | slope bounded by the pre-stated value |
| T13 | Degraded labelled sample at deployment prevalence (QSO:background mix fixed to the prior ratio per VIS): reliability, Brier, log loss; p -> prior as S/N -> 0 | reliability within binomial 99% intervals |

Not tests of truth: T11-T13 at VIS > 24 check self-consistency and the measurement model, not faint QSO colours or
prevalence; sum p vs Sigma_Q is tautological where |I| < 1. c_Q and the censoring of Legacy r are untested.

## 5. Risks
1. Faint truth: at VIS > 24 prior odds are ~0.4% (stage-1 A priors.json at r ~ 24.25) and three colours at S/N ~ 5
   rarely overturn them; most faint scores will be prior.
2. QLF and c_Q at VIS 24-26 are extrapolations; faint AGN may be host-dominated and not point-like.
3. Star/galaxy separation defines the population beyond VIS ~ 23.5.
4. Spectroscopic selection bias in the faint training QSOs; degradation tests share model B's assumption that faint
   QSOs look like dimmed bright ones.
5. Data: fit cones need a new WSDB download; cache collisions if roles change.

## 6. Against this plan
- At VIS > 24 the product may be "prior with flags" and not justify a new model and new cells; D4 decides.
- Flagging the support cut re-admits high-p points placed outside the dense core (method note); only the flag
  protects users.
- 2B adds a reference system; a constrained 2A that predicts the Euclid marginal well would avoid it.

## 7. PI decisions
- D1 target population (cleaning cuts in or out).
- D2 2B as the faint design (reviewers agree), or fund a 2A pilot with forced Legacy photometry.
- D3 new untouched evaluation cells and fit-cone download (size from Phase B).
- D4 at VIS > 24: publish prior-dominated p_quasar with flags, or a likelihood ratio only.

## 8. First steps
Phase A manifest + availability checks; Phase B census on select cones; Phase C benchmark. No fitting before D2 and D3.

## 9. Changes v1 -> v2 (review round 2)
- 2A -> 2B as the default (non-identifiable under censoring; no trainer for it). Both reviewers.
- Added Phase A: no fit-role cones exist locally; manifest and cache keying. Both.
- Routing fix moved from Phase 1 to Phase E; it is a likelihood change (South-r overwrite). Both. Only 73 of 5,491
  weak-r rows would have become scorable under the current rules (Fable).
- Low-S/N benchmark moved before the faint fit. Both.
- Phase 1 (now D) no longer "releasable alone"; diagnostic export only; catalogue validity separated from pair
  eligibility; z-free entry point implemented directly, not by an arbitrary z0. Both.
- Tests: paired identical cohorts; margins from measured SE; T9/T11 errors from cone-to-cone scatter; T13 at
  deployment prevalence; T12 with stated detectable slope; T4 z0-invariance added. Both.
- Corrected: `multisurvey.py:676` clamps the catch-all fraction, not a background colour bin (Codex);
  `need = ~legacy` is at `fallback_reference.py:79` (Fable).
