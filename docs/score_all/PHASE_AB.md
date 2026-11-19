# Score-all Phases A, B and C: results (19 November 2026)

Plan: `docs/euclid/PLAN_score_all.md`. Nothing fitted, nothing downloaded, no promoted bundle changed. Decisions
that block the next steps are listed at the end; nothing here pre-empts them.

## Phase A — data and roles

**Cache fix (done).** `scripts/euclid/fetch_cones.py` trusted any existing `<role>_<id>.npz`. It now checks the
stored centre and radius against the requested cone and raises `CacheMismatch` on a difference
(`cached_matches`; tests `tests/test_fetch_cones_cache.py`, 5 pass). All 64 local cone files (40 test, 12 select,
12 calib) match `docs/euclid/sky_roles.json`, so no earlier result used a wrong cone.

**Experiment manifest (done).** `scripts/score_all/manifest.py` -> `docs/score_all/manifest.json` (content sha256
recorded). Per nside-16 cell: role, cones, clean matched QSOs, parent-background rows, uses.

| role | cells | cones | clean matched QSOs | cells with parent-background rows |
|---|---|---|---|---|
| fit | 175 | 0 | 8,350 | 24 |
| select | 22 | 12 | 3,623 | 6 |
| calib | 16 | 12 | 620 | 4 |
| test | 30 | 40 | 2,246 | 3 |

- Test cells are spent (stage-2 diagnosis looked at their results); select and calib cells chose models and
  thresholds; fit cells trained the QSO extension.
- 13 test/select/calib cells also hold parent-background training rows. Their cones are >= 1 deg from those parent
  cones (sky_roles.py), so cone objects were never training rows, but the cells are not untouched.
- **109 cells have no recorded use** (no cone, no clean matched QSO, no parent-background row): 97 fit-role,
  4 test, 4 calib, 4 select. **67 of them have coverage >= 0.5** (55 fit-role, 12 others). This is the pool for new
  untouched evaluation cells (D3). Nothing is allocated.

**Availability of faint labels and deep fields (checked in WSDB, 19 Nov 2026).**

| source | what | count | use |
|---|---|---|---|
| `euclid_dr1.mer_catalogue_deep_survey` | Euclid Deep Field MER catalogue | 51.8 M rows (pg_class estimate) | never used by this project: an untouched sky set; deeper than wide, so evaluation needs degrading to wide depth |
| DESI DR1 QSOs (zwarn = 0) x deep MER (0.5") | spectroscopic QSOs in EDF-North | 2,661, of which 574 at Legacy r > 22.5 | faint labelled QSOs; EDF-Fornax and EDF-South have none in DESI DR1 |
| `chandra_sc2.main` (conf_flag = 0, extent_flag = 0) x deep MER (1") | X-ray point sources | 1,266 | small faint-AGN check sample |
| `euclid_dr1.spectro_z_spe_classification_deep` | NISP slitless classification | 60,829 `spe_class = qso` of 7.4 M | UNVERIFIED purity; not a label until checked against DESI |
| `xmm3dr4` | 3XMM-DR4 (old); 4XMM and eROSITA are not in WSDB | — | not checked further |

Deep-field centres used for the DESI count (EDF-N 269.73, +66.02; EDF-F 52.93, -28.09; EDF-S 61.24, -48.42, 4 deg
radius) are from the Euclid mission description, UNVERIFIED to the arcminute; the count is robust to that.

**Not done (blocked):** downloading fit-role cones (size depends on D3 and the Phase-B requirement below).

## Phase B — census of the deployment selection (select cones)

`scripts/score_all/census.py` -> `docs/score_all/census_select.json`, `plots/score_all/census_select.png`.
12 select cones, 3.39 deg^2 (full cone area assumed, see script), 60,701 selected point sources.

| VIS range | selected / deg^2 |
|---|---|
| 18-22 | 4,211 |
| 22-23 | 1,901 |
| 23-24 | 2,176 |
| 24-25 | 3,384 |
| 25-27.5 | 6,219 |

- Selected counts peak at **VIS = 25.4** (0.25-mag bins) and fall steeply after; all MER rows peak near the same VIS.
- **Legacy r at S/N >= 10 falls to half the selected sources at VIS = 23.6** (10% at VIS 24.4, 2% at 24.6); any
  Legacy match falls to half at VIS = 24.9. This is the census version of the stage-1 refusal table.
- Median NISP S/N reaches 10 at VIS = 24.7 (Y), 24.85 (J), 24.9 (H); VIS S/N reaches 10 at VIS = 25.8.
- So a VIS-referenced model (design 2B) has three informative colours to VIS ~ 24.8 and essentially VIS alone beyond.
  (Crossing points: linear interpolation between 0.25-mag bin centres.)

**Margins against the measured paired SE** (plan section 4: margins below 2 SE are unresolvable). The nearest
measured analogue is the stage-1 vs stage-2 paired comparison on the 40 test cones (2,246 QSOs, 24,000 background;
cell/cone bootstrap; `models/multisurvey_psf/work/euclid/compare_stages_*.json`):

| metric | margin | paired SE, combined mode | paired SE, Euclid-only | resolvable? |
|---|---|---|---|---|
| AUC | 0.002 | 0.0005 | 0.0007-0.0011 | yes; Euclid-only borderline (2 SE up to 0.0022) |
| recall p > 0.5 | 1 pp | 0.2 pp | 0.6-0.7 pp | combined yes; **Euclid-only no (2 SE = 1.2-1.4 pp)** |
| background p > 0.5 | 0.1 pp | 0.01-0.03 pp | 0.01-0.03 pp | yes |

SE scales as 1/sqrt(n): resolving the 1 pp recall margin in Euclid-only mode needs about twice the present test
cohort (>= 2 x 2,246 QSOs), or the margin must be widened before any fit. The T5 and T9 cohorts (VIS < 23,
VIS 21-23) are subsets and have larger SE; their SE will be measured on the new cells once D3 is decided.

**Fit-cone area (ASSUMED estimate, to be confirmed in Phase E).** At the census densities a 0.5-mag VIS bin holds
430-520 selected sources per deg^2 at VIS 19-20 and 1,060-2,030 at VIS 23-25 (VIS < 18.5 is cut). A 3-colour background mixture per bin
fitted on ~5,000 rows needs ~12 deg^2 at the bright end and ~5 deg^2 at the faint end; covering the latitude and
extinction range argues for more, spread over many cells. The plan's >= 30 deg^2 (~106 cones of r = 0.3 deg, ~1.5 GB
at the present ~14 MB per cone) is therefore ample; ~15 deg^2 is a defensible minimum.

## Phase C — is the low-S/N measurement model calibrated? (T8)

`scripts/score_all/phase_c_lowsnr.py` -> `docs/score_all/phase_c_lowsnr.json`, `plots/score_all/phase_c_lowsnr.png`.
The scorer treats each band's luptitude as Gaussian with first-order variance. True fluxes are set to S/N 1-20
against the measured faint-end flux error of the deployment selection (median error at VIS 24-25: 0.0061 nmgy in
VIS, 0.026-0.027 in Y, J, H; softening/error = 1.2-1.3), Gaussian flux noise is added (20,000 draws per band and
S/N), and the coverage of the true luptitude by the scorer's interval is measured.

| S/N | 68% coverage (all bands) | 95% coverage | T8 (+-3 pp) |
|---|---|---|---|
| 1 | 0.667-0.669 | 0.921-0.923 | fails (95%: -3.2 pp) |
| 2 | 0.673-0.680 | 0.934-0.937 | pass |
| 3-20 | 0.676-0.691 | 0.942-0.955 | pass |

No bias (median standardised residual within +-0.01); PIT KS distance 0.025-0.035 at S/N <= 5, 0.01 at S/N 20.
**Result: the Gaussian-luptitude likelihood is adequate for S/N >= 2; at S/N ~ 1 its 95% tails are 3 pp too narrow.**
Following the plan (flux-space where they disagree), bands at S/N < 2 should use the flux-space likelihood or be
treated as non-detections in the faint model; by the census that concerns NISP beyond VIS ~ 26.5.
Limitation: the test assumes the catalogue's quoted flux errors are right. Checking them needs repeat measurements
(`euclid_dr1.mer_catalogue_deep_visits` holds per-visit photometry in the deep fields; not yet used).

## Decisions (PI, 19 November 2026)

All five are decided (PROJECT_STATE.md, PLAN_score_all.md): D1 keep the cleaning cuts; D2 VIS-referenced faint model
(2B); D3 evaluation background from 40 new cones in unused cells + quasars from EDF-North, fit cones ~30 deg^2
(allocated: `scripts/score_all/allocate_cones.py` -> `docs/score_all/new_cones.json`, 40 eval cones in 32 cells,
106 fit cones in 54 cells, 30.0 deg^2, no shared cell); D3b recall margin 2 pp; D4 publish p and the likelihood ratio.

## Earlier list of decisions (before 19 November 2026 decisions)

- **D1** target population: keep the cleaning cuts as in `cones.py`, or include removed rows.
- **D2** 2B (VIS-referenced faint model; the census supports it: usable Legacy r is gone by VIS ~ 24.6) or a 2A pilot.
- **D3** new evaluation cells: from the 67 well-covered unused cells, and/or the Euclid Deep Fields (untouched, with
  574 DESI QSOs at r > 22.5 in EDF-N, but deeper photometry). Also: fit-cone area (>= 15 deg^2; plan says 30).
- **D3b** (new) the Euclid-only recall margin of 1 pp is unresolvable at the present cohort size: double the
  evaluation cohort or widen the margin, decided before any fit.
- **D4** what to publish at VIS > 24.
