# Scoring every Euclid point source — plan (v2, 2026-11-19)

Status: REVIEWED PLAN; Phase D-main done (19 Nov 2026), nothing fitted. D5 decided 2026-11-19 (Phase D-main, section 3). Two rounds with two independent reviewers (Codex via the
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
**DONE 2026-11-19 except the fit-cone download (blocked on D3):** cache check, manifest, availability; results in
`docs/score_all/PHASE_AB.md`.
- An immutable experiment manifest: cells used so far for fitting or decisions (incl. the current Euclid test
  cells, spent by the stage-2 diagnosis), and newly allocated untouched evaluation cells (D3). Caches keyed by
  coordinates and manifest hash; `fetch_cones.py:49` currently skips any existing filename without checking
  coordinates, so this needs a fix.
- Download fit-role cones: none exist locally (40 test, 12 select, 12 calib; `sky_roles.py` creates only those
  roles). Size from Phase B's requirement; target >= 30 deg^2 of fit cones (ASSUMED adequate; to be set by B).
- Check whether Euclid deep fields and faint spectroscopic or X-ray AGN samples are available (UNVERIFIED).

**Phase B — census** (DONE 2026-11-19 on select cones, `docs/score_all/PHASE_AB.md`; repeat on fit cones) (select cones until fit cones exist): area-normalised counts by VIS (0.25 mag), S/N per band,
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

**Phase D-main — the same three fixes in the main (Legacy) scorer. PI decision D5, 2026-11-19: test and switch on.**
**DONE 2026-11-19: M1-M4 PASS for both bundles; defaults switched on** (results `docs/score_all/`, method note
Section "Flags instead of refusals"). M4 first crashed the laptop (memory); fixed in commit 521fd0d.
The three no-refit fixes act in code shared with the main scorer, so they are implemented once and switched on for
both scorers:
1. support and outside-both refusals become flags (`unified.py:253-260`, `multisurvey_baseline.py:125`);
2. the z-free `p_quasar` entry point;
3. weak measured Legacy r no longer blocks the fallback (PS1/NSC/SkyMapper r for the main scorer, VIS for Euclid):
   the integral keeps the measured weak-r likelihood instead of overwriting the South-r column
   (`fallback_reference.py:79, 162, 198-200`), and the first fallback band that passes S/N is used.
Tests (in addition to T1-T13):
- M1: rows eligible before the change: every numeric field and flag unchanged, max abs diff < 1e-9 (fixes 1 and 3
  only touch previously refused rows; fix 2 adds fields). This replaces T2's "parent unchanged".
- M2: parent held-out main panels (`performance_qlf`, S/N >= 10), identical all-rows cohort with old refusals at
  p = 0, paired bootstrap by sky cell: AUC lower 95% bound > -margin, recall > -1 pp, unflagged background p > 0.5
  upper bound < +0.1 pp; margins set from the measured paired SE first. Reported separately for each newly scored
  stratum (support-flagged, outside-both, weak-r). These panels served as the stage-2 comparator but were not used to
  design these fixes.
- M3: z-free `p_quasar` (no z0) against the pair scorer's total p at z0 = 0.6, 1.5, 3.0, 4.2: max |dp| < 1e-4
  (the pair scorer adds window nodes around z0 to the same quadrature; 1e-9 was not attainable — changed
  19 Nov 2026 before any panel run; unit test `tests/test_score_all.py`).
- M4 (made precise 19 Nov 2026, before running; `scripts/score_all/weak_r_masking.py`): held-out main-panel rows
  with South r S/N > 30, no North r and a fallback band at S/N >= 10; Legacy r degraded to S/N 5 (noise added).
  Scored with the weak r kept (weak), with r removed (none), and as measured (good). Pass: lower 95% bound of
  AUC(weak) - AUC(none) > -0.002 (paired cell bootstrap) and median |logit p - logit p_good| of weak <= that of
  none. Quantiles of |p_weak - p_good| reported. (The draft's |dp| < 0.02 vs good-r was not a valid criterion:
  degrading r to S/N 5 legitimately changes p.)
- Unit tests (`tests/test_score_all.py`): an uninformative weak r reproduces the no-r fallback exactly; fallback
  band = first passing S/N.
- Method note and README updated in the same commit as the behaviour change (AGENTS.md); old behaviour kept behind
  a flag (`apply_support=True`, `weak_reference_fallback=False`) for reproduction of earlier results.
Phase D-main needs no fitting and does not depend on D1-D4; it can start before Phases A-C.

**Phase E — faint model (2B) and the weak-r integral.**
- Fit as in section 2 on fit cones, after Phase C.
- Euclid-only path: VIS-referenced, no VIS-Y window, no centre or tail-mass cut; numerical convergence (T3).
- Weak Legacy r for Euclid rows: the Phase D-main integral, with VIS as the fallback reference.
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
| T5 | Phase D on new cells, VIS < 23: old vs new on the identical all-rows cohort, old refusals at p = 0; paired cell bootstrap | AUC lower 95% bound > -margin; recall > -2 pp (D3b) |
| T6 | Background p > 0.5 rate, unflagged rows (known/likely-QSO exclusions as `validate_stage1.py:129`), combined and Euclid-only modes | upper 95% bound of the change < +0.1 pp |
| T7 | Weak-r integral: paired masking (drop Legacy r from rows that have good r, then degrade it) | |dp| < 0.02 for 95%; AUC change > -margin |
| T8 | Phase C: PIT and 68/95% coverage of degraded vs true flux, S/N 1-20 | coverage within +-3 pp per S/N bin |
| T9 | Overlap VIS 21-23, rows with usable Legacy r: Phase E Euclid-only vs stage-1 Legacy path, identical cohort | AUC > -margin; recall > -2 pp (D3b); bkg < +0.1 pp |
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
- D1 DECIDED 2026-11-19: keep the cleaning cuts (target = the cones.py deployment selection).
- D2 DECIDED 2026-11-19: 2B, the VIS-referenced faint model.
- D3 DECIDED 2026-11-19: background evaluated on new cones in the 67 unused well-covered cells (manifest); quasars
  evaluated on EDF-North deep-MER DESI QSOs degraded to wide depth; fit cones ~30 deg^2 in other fit-role cells.
- D4 DECIDED 2026-11-19: publish both p_quasar (flagged prior_dominated where colours carry little information)
  and the likelihood ratio.
- D6 DECIDED 2026-11-19 (after Phase E step 0, docs/score_all/PHASE_AB.md): option (a). Faint-galaxy colours are fitted on all of EDF-North deep MER, with every known QSO (DESI DR1, DR16Q; same removal rule in every training field) removed. Tests unchanged: QSOs = EDF-N DESI QSOs degraded to wide depth (never in training); background = the 40 wide eval cones. Added: EDF-South at VIS < 25 as an independent check of the galaxy model (never fitted), which also bounds the same-field optimism of the EDF-N QSO test; faint checks with Chandra X-ray point sources and the Euclid slitless QSO class (checks, not labels).
- D3b (from Phase B, 19 Nov 2026): the Euclid-only recall margin (1 pp) is below 2 paired SE (1.2-1.4 pp) at the
  present cohort size. DECIDED 2026-11-19: recall margin 2 pp for all score-all tests (T5, T9), fixed now.
- D5 DECIDED 2026-11-19: the three no-refit fixes are tested and switched on in the main scorer too (Phase D-main).

## 8. First steps
Phase D-main (no fitting, independent of D1-D4); Phase A manifest + availability checks; Phase B census on select
cones; Phase C benchmark. No fitting before D2 and D3. Every step journalled, committed and pushed.

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
