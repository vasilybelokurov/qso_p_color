# Euclid data tests (docs/euclid/PLAN.md section 4), 18 November 2026

Numbers: `docs/euclid/data_tests.json` (`scripts/euclid/data_tests.py`). Background = parent background rows of the
33 overlapping cones with a MER match (403,658); QSOs = 17,616 matched eligible targets. Stellar locus: cubic in
Legacy g-r, r-z, z-W1 per Euclid colour, fitted to 2,922 clean isolated point sources with all bands S/N >= 30;
locus NMAD 0.044 / 0.048 / 0.044 mag (VIS-Y, Y-J, J-H).

| Test | Result | Decision |
|---|---|---|
| T-D1 point rule (QSO side) | completeness at p >= 0.1: z 0.8-1.5: 0.99 (VIS < 21), 0.90 (21-23); z 1.5-2.5: 1.00 / 0.98; z > 2.5: 1.00 / 0.99; z < 0.5: 0.54; at p >= 0.3 z 0.8-1.5 VIS 21-23 drops to 0.83 | p_min = 0.1 (QSO completeness >= 0.9 at z > 1, VIS < 23); galaxy contamination measured in the select cones (T-D1 second half) |
| T-D2 VIS column | VIS_TO_Y_TEMPLFIT - Kron flat (-0.013..-0.020 mag) over VIS 17-24 | VIS = FLUX_VIS_TO_Y_TEMPLFIT |
| T-D3 flag bits | scatter ratio vs clean (max over colours and S/N bins): bit 2: 1.38; 4: 1.62 (51 rows); 128: 3.34; 256: 4.92; 512: 1.39. QSOs kept if cut: bit 2: 7-49% at VIS < 21; 128/256: 14-66% at VIS < 19, >= 99.3% fainter; 4: >= 99.6%; 512: 98.9% | cut 4, 128, 256, 512 (bad_bits = 900). **Bit 2 kept against the 1.2x rule**: it is set for most bright QSOs (MER deblending flag); isolation (no neighbour within 1.5", Delta VIS < 2.5) already removes close pairs. blended_prob never > 0.1: unused |
| T-D4 matching | QSOs: 99.86% within 0.3" (99.9th pct 0.40"); background positional median 0.099", Gaia-pm-propagated median 0.011" (25,514 rows) | pass |
| T-D5 bright limit | median locus residual VIS-Y: -0.071 (VIS 16.5), -0.046 (17.0), -0.029 (17.5), -0.016 (18.0), < 0.01 at 18.5-20 | vis_bright = 18.0. Noted: VIS-Y residual drifts to +0.019/+0.023 at VIS 21/21.5 (population change or calibration; not corrected) |
| T-D6 tiles | 71 tiles with >= 200 locus stars: rms 0.014 / 0.009 / 0.008 mag; max |offset| 0.054 (VIS-Y, one tile) | pass (rms <= 0.02); no per-tile correction |
| T-D7 dereddening | slope of locus residual vs E(B-V) (our correction applied): -0.20, +0.26, +0.25 per unit E; a double correction would give -1.00, -0.29, -0.21 (E range 0.007-0.068) | MER fluxes are not dereddened; our SFD correction stands |
| Isolation | QSOs isolated 96.1% (loss 2.6% at z < 0.8 to 4.3% at 1.5-2.5); background 93.9% | rule kept; loss enters c_Q |
