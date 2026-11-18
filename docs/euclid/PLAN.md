# Resolved mode from Euclid photometry only: plan and tests

Version 2 (18 November 2026): final after adversarial review (2 reviewers x 2 rounds; method reviewer M, data and
engineering reviewer D; round-2 findings M2.n, D2.n). Version history and the reviews' finding lists:
`docs/euclid/REVIEW_LOG.md`.

## 0. Task and decisions

**Task.** Resolved mode (`UnifiedPSFModel`): score an individual point source, measured as its own source, with
**Euclid photometry only** (VIS, Y, J, H; Euclid DR1 MER) for `p_quasar` and `log_r_per_unit_z` at z0.
Hypotheses (same_q, field_q, bkg, out), \|b\| >= 25, support and outside-both rules carry over.

**PI decisions (18 November 2026).** (1) Euclid photometry only. (2) Stage 1 frozen extension of both promoted
bundles (A = `current`, magnitude-dependent; B = `current_magindep`), then stage 2 full joint refit. (3) Adversarial
review before implementation.

**Technical decisions taken here (flagged for PI review in the report).**
- T1. Euclid sky roles: HEALPix nside-16 cells inside DR1 wide at \|b\| >= 25, assigned fit/select/calib/test =
  55/15/10/20% by a seeded assignment **stratified** by matched-QSO count, coverage and \|b\|; cells < 50% covered go
  to fit (D2.5, M2.1). Frozen (seed + table) before any fit. QSOs in non-fit cells are excluded from extension fits.
  Background cones of the parent in non-fit Euclid cells are excluded from extension fits. New background cones
  (r = 0.3 deg, >= 1 deg from parent cones and each other) in test (>= 25), select (>= 8) and calib (>= 8) cells.
  Residual leakage: QSOs in Euclid test cells may be parent fit rows (one in ~10^6, marginalised out); recorded.
- T2. Deployment selection is an argument of scoring: `legacy_psf` (unchanged) or `euclid_point` (new). Priors,
  thinning, catch-all fraction and support threshold are keyed by selection (M7).
- T3. Scope: pairs closer than 1.5" are out of scope for Euclid-only scoring (isolation cut); stated in README.

### Knobs and the role that tunes each (M2.8)
| Knob | Tuned on | Never touches |
|---|---|---|
| Extension a, B, W; z-kernel; kappa; candidate choice | fit (parameters), select (hyperparameters, choice) | calib, test |
| s_b (Euclid error scale) | fit-cell stars, W fixed from high-S/N rows | calib, test |
| p_min (point rule), bright limit, flag bits, isolation rule | select cells (QSOs + cones) | calib, test |
| r_hat(VIS, VIS-Y), window, VIS_max | fit + select | calib, test |
| f_k(r), c_Q(z, r) | fit | calib, test |
| catch-all eta, support threshold, ood alpha check | calib | test |
| reported numbers | test only | - |

## 1. Measured facts (18 November 2026)
| Fact | Value | Source |
|---|---|---|
| MER tables | `wide_survey` (888M) + `wide_mode` (58M), disjoint, similar depth; deep tables excluded | WSDB |
| Eligible QSO targets matched within 1" | **17,616 of 1,657,444** (wide_survey 11,319; wide_mode 6,297) | `scripts/euclid/fetch_qso_euclid.py` |
| VIS columns on these QSOs | `VIS_TO_Y_TEMPLFIT` - Kron = -0.013..-0.020 mag flat over VIS 17-24; PSF - Kron = 0.10-0.14 varying; VIS_TO_Y finite 99.9% of vis_det=1 | this work, D2.7 |
| QSO `point_like_prob >= 0.1` (VIS < 23) | 0.55 (z<0.5), 0.81 (0.5-0.8), 0.94 (0.8-1.5), 0.98 (1.5-2.5), 0.985 (z>2.5); >= 0.5: 0.49/0.68/0.88/0.97/0.98 | this work |
| QSO flags | det_quality_flag 0: 60%, 2: 35%, 386: 2.5% (bright-star mask bits 128+256 set for 38.5% at VIS 15-19); spurious_flag=1: 1.5%; vis_det=0: 0.7% | this work, D2.3 |
| QSO neighbours | 15% have any MER source within 1.5"; 6% one with vis_det=1, non-spurious | D2.1 |
| Euclid R = A/E(B-V)_SFD (SF11 recipe, scale 0.8787 refitted: SF11 Table 6 rms 0.98%) | VIS 1.917 (7000 K; QSO 1.894; 3000 K 1.702), Y 0.921, J 0.629, H 0.420 | `src/qso_pcolor/band_extinction.py` |
| Background cones overlapping DR1 wide | 33 of 274 | this work |

## 2. Design

### 2.1 Bands, units, cleaning
- `euclid:vis` = `FLUX_VIS_TO_Y_TEMPLFIT`; `euclid:y|j|h` = `FLUX_{Y,J,H}_TEMPLFIT` (uJy x 0.275423 -> native).
- Versioned Euclid band registry separate from `SURVEYS`; per-bundle fallback band list and faint-limit bands
  (D7, D2.9). Parent bundles unchanged.
- Cleaning (training rows and deployment identical): `vis_det = 1`, `spurious_flag = 0`; no isolation-neighbour
  within 1.5" (neighbour = MER source with vis_det = 1, spurious_flag = 0, VIS within 2.5 mag) (D2.1);
  `det_quality_flag` bits and |VIS_TO_Y - Kron| outliers decided by T-D3 per bit, with QSO retention (D2.3, D2.7);
  bright limit from T-D5. Sentinels 32767, -9999, 9223372036854775807, NaN handled explicitly (D11).
- Point rule: `point_like_prob >= p_min` (T-D1); never `point_like_flag`.
- Extinction: SFD98 with the R values of section 1 (7000 K source, as for every existing band); the SED spread
  (VIS 1.70-1.92) is a systematic of <= 0.011 mag at E(B-V) = 0.05. T-D7 tests that MER fluxes are not already
  dereddened.
- Error model: sigma' = s_b sigma per Euclid band; s_b fitted on fit-cell stars with W fixed from rows with all
  Euclid S/N > 50, then frozen (M2.5).
- Training rows get **no** observed-data faint-limit cuts (M2.7); only the cleaning and point rule above.

### 2.2 Stage 1: frozen extension
Latent 36 -> 40 (Euclid VIS, Y, J, H). Per component k the old block is bit-identical; Euclid colours
c_E = x_E - x_r 1 are linear-Gaussian in d = (g-r, z-r, w1-r, w2-r [, r for model A, clamped to the training range]):
c_E | d, k ~ N(a_k + B_k (d - mu_k,d), W_k). Implemented and unit-tested in `src/qso_pcolor/euclid_extension.py`
(exact EM, old block frozen, errors in regressors through posterior second moments).
- E-step observes Legacy grzW1W2 (native north/south through the parent operator) + Euclid bands; other parent
  bands are marginalised (exact under the model, less informative, much cheaper).
- QSOs: B, W shared within a Gaussian z-kernel over slices (width h_z chosen on select; <= 0.2 at z > 3); per-
  component intercepts shrunk to the kernel regression (kappa_a); slices with < 30 kernel-weighted QSOs: kappa_a = inf.
- Background: per component a_k, B_k, W_k shrunk to the r-bin regression (kappa_b, kappa_w).
- Model B excludes x_r from d.
- Candidates: (i) B = 0 with per-component W, (ii) full, (iii) group-only. Chosen on select rows by Euclid-only
  marginal predictive density and Euclid-only AUC (M5).

### 2.3 Selection `euclid_point`: thinning
- Background: lambda_bkg,E(y, r) = Sigma_B(r) sum_k w~_k(r, l, b) f_k(r) p_k(y | r), with w~_k the parent conditional
  weights and f_k(r) = P(Euclid point and clean | component k, r), a logistic in r per component shrunk to the bin
  fit, estimated from **old-block** responsibilities of parent background rows in covered fit-cell area (M2.4).
  Coverage from MER tile footprints (tile_index extents) and bright-star masks; test: unmatched Legacy-PSF rows do
  not cluster at bright stars (D2.6). Data test: predicted vs observed Euclid-point counts per r bin, select cones.
- QSOs: Sigma_Q,E(z, r) = Sigma_Q(z, r) c_Q(z, r), c_Q = P(Euclid point and clean | Legacy PSF, z, r) measured on
  Legacy-PSF spectroscopic QSOs in fit cells; Sigma_Q keeps the parent convention "QSOs are Legacy PSF" (M2.2).
  z < 0.8: flag `low_z_morphology_uncalibrated` (c_Q depends on targeting there).
- Euclid-point sources that are not Legacy PSF are counted per r (reported; stage-2 item).
- Catch-all: old-block marginal unchanged; eta refitted for `euclid_point` on calib rows.
- Known stellar systems (dwarf galaxies, globular clusters) masked from cones, Sigma_B and T-D6 (D2.4).

### 2.4 Euclid-only scoring
- Legacy r integrated over given Euclid (generalised `fallback_reference.py`; fallback list per bundle).
- Window: centred on r_hat(VIS, VIS - Y) fitted on fit+select rows of both populations; asymmetric half-widths from
  the residual quantiles (0.05%, 99.95%) where >= 2,000 rows, else a Student-t fit (M2.9); + 5 sigma.
- Eligibility: VIS S/N >= 10, >= 2 NISP bands at S/N >= 5, VIS <= VIS_max; and the flat-prior likelihood
  p(y | r) evaluated to r = 26 has < 1e-3 of its mass above r = 23.5 (the training range); otherwise
  `legacy_r_outside_training` (M1, M2.6). VIS_max set from T-V3 so that this holds for 99% of select rows.
- Support and outside-both: same code; `euclid_only` mask threshold calibrated on calib cells, target 0.995.
- Blend policy for Euclid-only rows: the isolation cut of 2.1 replaces Legacy `fracflux`.

### 2.5 Stage 2: full joint refit
- Parent training rows with Euclid columns attached where matched (Legacy-matched rows only; roles unchanged;
  Euclid non-fit cells excluded from fitting); 40-d latent, same slices/bins/K; warm start = stage 1; existing
  trainer + MAP update; stopping panels plus a Euclid select panel; thinning, eta, support recalibrated.
- Acceptance, decided before fitting: paired bootstrap over cells, **pass if the lower 95% bound of
  (stage 2 - comparator) > -margin**; margins: AUC 0.002, recall at p > 0.5 1 pp, unflagged-background p > 0.5
  rate 0.1 pp (upper bound < +margin); comparators: parent bundles on the 41-band test panels, stage 1 on Euclid
  test cells; blend mode (`log_bf_qq_qs` AUC and P(QQ) > 0.9 rates on the synthetic blends) with the same margins
  (M2.3). Before fitting, the stage-1-vs-itself bootstrap SE of each statistic is reported; a margin below 2 SE is
  declared unresolvable. Failing stage 2 leaves stage 1 as the deliverable.

## 3. Data acquisition
1. Done: QSO targets. Re-query matched QSOs for 1.5" isolation-neighbour counts (definition 2.1). Provenance:
   SQL text, target-file hash, `object_id` alignment, `statement_timeout` (D2.8).
2. Background rows of the 33 overlapping cones: DR3 ids from the parent preparation (already DR3) -> MER `gaia_id`
   (duplicates resolved by `gaia_match_quality`, then separation); others by position (1.5") (D2.2).
3. Euclid sky roles (T1); new cones; per cone: all MER sources passing cleaning, Legacy DR9 (type, fluxes),
   Quaia, DESI/SDSS labels; Legacy W1-W2 likely-QSO flags.
4. No other downloads.

## 4. Tests and acceptance

### Unit
| ID | Test | Pass |
|---|---|---|
| T-U1 | `legacy_psf`: density/intensity fields identical for 41-band input | < 1e-9 (A and B, 2,000 rows); seed-dependent percentiles: KS p > 0.01 |
| T-U2 | Old block of every extended component == parent | exact (implemented) |
| T-U3 | Extension EM recovers synthetic truth; monotone | implemented, passes |
| T-U4 | Euclid conversion, sentinels (32767, -9999, 2^63-1, NaN), cleaning | exact |
| T-U5 | Extinction recipe vs SF11 Table 6 | rms <= 1% (measured 0.98%) |
| T-U6 | Legacy r posterior coverage, Euclid-only draws | 68/95% within +-3% |
| T-U7 | Support/ood p uniform, Euclid-only band set | KS p > 0.01 |
| T-U8 | Thinning algebra; `legacy_psf` path unaffected | exact |

### Data
| ID | Test | Pass / decision |
|---|---|---|
| T-D1 | Point rule: QSO completeness by (z, VIS); contamination by DESI spectroscopic galaxies | p_min: QSO completeness >= 0.9 at z > 1, VIS < 23, galaxy contamination <= 5% |
| T-D2 | VIS column | done: VIS_TO_Y_TEMPLFIT (flat, -0.013..-0.020 mag vs Kron) |
| T-D3 | Per bit (det_quality_flag, flag_band), blended_prob, VIS_TO_Y-Kron outliers: stellar-locus residuals **and** QSO retention by VIS | keep a bit unless residual scatter > 1.2x clean |
| T-D4 | Match paths (position / gaia_id), rates, separations, duplicates | QSO > 99% within 0.3"; rates reported |
| T-D5 | Bright limit | VIS where locus residual median > 0.02 mag |
| T-D6 | Per-tile offsets perpendicular to the stellar locus (stellar systems masked) | rms <= 0.02 mag, else per-tile correction |
| T-D7 | Locus colour vs E(B-V) slope | consistent with uncorrected MER fluxes |
| T-D8 | Thinning: predicted vs observed Euclid-point counts per r bin, select cones | within Poisson 99% per bin |

### Validation (both models; bootstrap over cells; binomial intervals)
| ID | Test | Pass / report |
|---|---|---|
| T-V1 | Candidates on select rows | choice recorded |
| T-V2 | Posterior predictive VIS-Y, Y-J, J-H; QSOs by z, background by r; PIT vs S/N | coverage within binomial 99% intervals |
| T-V3 | r - r_hat residuals by population/colour; VIS_max | recorded |
| T-V4 | Window +-50%: all, red stars, z > 4 QSOs | median \|dp\| < 0.01, 90th pct < 0.02 |
| T-V5 | **Main**, Euclid test cells: AUC (log R; unranked at bottom), recall at p > 0.5, recall at the background rate of the Legacy-baseline p > 0.5, unflagged-bkg p > 0.5 rate (upper bound), unranked; by VIS, z, \|b\|; complete-DESI-coverage cones reported separately | reported **relative to** the same objects scored with Legacy g,r,z,W1,W2 only (baseline) and Legacy + Euclid (M2.10) |
| T-V6 | Support/ood retention, calib cells | >= 0.995 |
| T-V7 | p_quasar calibration, complete-coverage cones | within label bounds |
| T-V8 | Redshift: PIT and 68/95% coverage of the Euclid-only z posterior for z0 slices with >= 100 test QSOs; p_sameq vs \|z - z0\| | PIT KS p > 0.01, coverage in binomial intervals (M2.M8) |
| T-V9 | Model B: Euclid colour residuals vs r on test QSOs | slope consistent with 0 (M13) |
| T-V10 | Stage 2 acceptance (2.5) | as stated |

## 5. Order of work
1. Data (QSO re-query, background matches, sky roles, cones). 2. T-D1, T-D3-T-D8. 3. Code + T-U1-T-U8.
4. Stage 1 (A, B) + T-V1. 5. T-V2-T-V9; method note; README. 6. Stage 2 + T-V10. 7. Report.

## 6. Residual risks
- ~17.6k QSOs, DESI/SDSS-selected; fit cells hold ~10k; z > 3.5 thin.
- Three colours; weak separation expected at some z.
- Parent background lacks Euclid-point sources that are not Legacy PSF (stage-1 bias, measured).
- VIS -> Legacy r link includes quasar variability; MER DR1 calibration preliminary.
