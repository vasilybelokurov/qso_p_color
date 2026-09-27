# Legacy-PSF baseline: what we are building and why (2026-09-27)

This replaces the design parts of `docs/PLAN_FIELD_MODEL.md` (§§1–9). That file
is kept as a record of how we got here. Its §8 lists what went wrong on the
first attempt.

## Goal

A model trained on, and applied only to, point sources (Legacy DR9
`TYPE = PSF`) in Legacy g, r, z, W1 and W2. For a PSF companion of a quasar at
z₀ it returns ln R (the ranking statistic) and p_sameq. It is validated on
held-out sky, compared with the current model and with the original archived
Legacy model. After this, it is the baseline that any further infrastructure
has to beat.

## Design: the original XDQSO-style model

Bovy et al. 2011, [arXiv:1011.6392](https://arxiv.org/abs/1011.6392); 2012, [arXiv:1105.3975](https://arxiv.org/abs/1105.3975).

| part | what |
|---|---|
| photometry | Legacy DR9 fluxes **dereddened**: flux / `mw_transmission`, variance / `mw_transmission`² |
| features | relative fluxes g/r, z/r, W1/r, W2/r; ≥ 3 usable; r S/N ≥ 5 (flag otherwise); 17 ≤ r < 22.5 (dereddened) |
| quasar model | 43 redshift slices, z 0.1–4.4, one XD mixture each in the relative fluxes. **Colours do not depend on apparent magnitude.** Trained on DESI DR1 + DR16Q PSF quasars with the original trainer (`scripts/train_qso_model.py --morphology point`): K = 20 where a slice is large, chosen on held-out blocks where it is small; floor 10⁻⁶ |
| field model | same features; one XD mixture per r bin (17, 19.5, 20.5, 21.5, 22.5), so it **does** depend on magnitude (disc/halo mix, galaxies). K ∈ {4, 8, 12, 16} and floor ∈ {10⁻⁶, 10⁻⁵, 10⁻⁴} per bin, chosen on the select cells |
| priors | Σ_Q(z, r) from the weighted DESI + SDSS draw over the parent footprint; one completeness constant C from the **all-morphology** south quasars, applied unchanged to the PSF prior. Σ_B(r) from the PSF field counts |
| unmodelled term | the original broad Gaussian (`OutlierModel`); κ and η(r) on the calib cells |
| scorer | unchanged: `score_candidates`, four hypotheses, supported-window integration |

## What is new relative to the original model, and why

1. **PSF selection**, one rule used for training, counting and candidates
   (`qso_pcolor.legacy.LegacySelection`):
   - hemisphere from position (DESI rule), and the release must match it;
   - `maskbits = 0`; usable r;
   - `unknown` morphology is never counted as point.
2. **Field sample**: 304 cones of 0.3°, spread over the 124 nside-4 cells of
   the footprint at |b| ≥ 25°. Each cone's usable area comes from the brick
   mask images. The original model used 8 cones in the south.
3. **Unrecognised quasars.** Only quasars with a spectrum can be removed from
   the field sample: per cone the recognised fraction κ has a median of 0.37.
   In a PSF sample the rest sit on the quasar locus.
   - Without correction they lowered real quasars' ln BF by ~1.8 nats in the first attempt.
   - Fix: each field source is weighted by 1 − P(unrecognised quasar), and Σ_B subtracts (1 − κ) Σ_Q.
   - Compare with the simpler option of fitting only cones with high spectroscopic coverage.
4. **Fixed sky partition.** Cells are assigned to fit / select / calib / test before anything is fitted.
   - test = the archived held-out blocks. They are all in the south, so the north ships as *provisional*.
5. **Parallel fits**, and a **local data store** in `~/data/qso_p_color`, so nothing is re-queried.

## Deliberately not done (and why)

- **Joint magnitude–colour quasar model:** it made quasar colours inherit the DESI/eBOSS WISE selection (W1 − r shifted ~0.9 mag between r = 18 and 22).
- **Joint 5-D field mixture:** it needed K = 64–128.
- **Student-t term:** it goes unless a tail test shows the broad Gaussian fails.
- **Per-cell (HEALPix) adaptation:** deferred until the baseline passes.
- **Legacy DR10:** local DR10 matches exist, but everything here is DR9. Switching is a separate decision.

## Known limitations (stated, not fixed here)

- The quasar colour distribution is still the *selected* one (DESI/eBOSS); only its magnitude trend is removed.
- The PSF cut removes most low-z quasars: PSF fraction 0.18 at z < 0.5, 0.73 at 0.5–1, ≥ 0.98 above z = 1.
- C and a magnitude-independent recognised fraction κ are assumptions.
- The held-out companions are DESI-selected, like the training quasars.
- **The shipped multi-survey model uses native, not dereddened, fluxes.** It needs the same correction. Not changed yet: that is a decision for the PI.

## Steps

| # | step | test | status |
|---|---|---|---|
| 1 | Local store: DESI DR1 QSO, DR16Q × DR9, pairs, quasar draw, cones | row counts, match rates, `nobs`/extinction columns present | copies done; DESI and DR16Q fetching |
| 2 | Quasar slices, south then north, from the store | all slices converged; K per slice; held-out density by channel (DESI vs SDSS) | code ready |
| 3 | Field mixtures, Σ_Q, Σ_B, contamination weighting | converged; K and floor per bin; the correction's iteration converges; the 5 % Σ_B floor does not activate; compare with covered cones only | code ready |
| 4 | Broad Gaussian term on the calib cells | extreme colours, missing and useless bands | code ready |
| 5 | Validate once on the test cells | ΔAUC vs original and current (same-z/wrong-z, stars, PSF galaxies) with block bootstrap; convergence; normalisation; coarse tails | validator to be pointed at the new bundle |
| 6 | Publish the bundle if it passes; update the method note | load/score round trip; mixed-bundle refusal | — |
