# Adversarial review of docs/euclid/PLAN.md (18 November 2026)

Two independent reviewers (Claude Opus subagents, read-only; short WSDB queries allowed), two rounds each.
M = statistics and method; D = data and engineering. Version 0 -> 1 after round 1, 1 -> 2 after round 2.
Each line: finding (severity) -> disposition in version 2.

## Round 1, method (M)
1. VIS S/N 10 reaches VIS ~25.5, beyond every prior (background bins end r = 24.5; Euclid QSO r 99th pct 23.5); out-of-prior grid points silently truncated (BLOCKER) -> eligibility rule on flat-prior likelihood mass above r = 23.5, VIS_max, NISP S/N.
2. Morphology selection applied to the background only; Sigma_Q counts all QSOs (BLOCKER) -> c_Q(z, r) and per-component thinning f_k(r).
3. Fallback window centred on VIS; VIS - r up to -2 for M dwarfs (BLOCKER) -> asymmetric window on r_hat(VIS, VIS-Y).
4. 21/43 slices < 300 Euclid QSOs; pooling erases Ly-alpha signal (MAJOR) -> z-kernel sharing, kernel width capped at z > 3.
5. Wrong selection metric (conditional density); shared B for a whole background bin (MAJOR) -> Euclid-only marginal density + AUC; per-component background extension.
6. Constant extra variance unidentifiable (MAJOR) -> multiplicative s_b fitted on stars with W fixed.
7. T-U1 contradicts refitting eta/catch-all (MAJOR) -> selection-keyed priors/eta/support; T-U1 under legacy_psf.
8. Redshift output untested (MAJOR) -> T-V8.
9. Deployment environment (blends, proper motions) (MAJOR) -> isolation cut, Gaia-id matching, T-D3.
10. Labels incomplete (MAJOR) -> complete-coverage cones reported separately; false-positive rate as upper bound.
11. Stage 2 ill-defined (MAJOR) -> Legacy-matched rows only, Euclid stopping panel, confidence-bound acceptance.
12-17 (MINOR): EM second moments (done), T-U7 tautology (replaced by T-V9), magnitude clamp (done), binomial intervals (done), support target 0.995 (done), point-rule truth from DESI galaxies (done).

## Round 1, data/engineering (D)
1. FLUX_VIS_PSF biased by 0.05-0.18 mag, magnitude- and type-dependent (Redmine #34505, #33210; MER Photometry Cookbook) (BLOCKER) -> VIS = FLUX_VIS_TO_Y_TEMPLFIT (verified flat on 17,616 QSOs).
2. Point rule removes low-z QSOs; point_like_flag is a shear-star selection, 32767 unset (BLOCKER) -> c_Q, point_like_prob only.
3. Test cones as specified infeasible (82 deg^2 of parent test cells) (BLOCKER) -> Euclid-specific stratified sky roles (T1).
4. r - VIS up to 1-2 mag for red stars (MAJOR) -> window redesign.
5. Flag semantics: bright-star-mask bits (386) on bright stars, saturation bit never set, point_like_prob NaN at VIS 16-17 (MAJOR) -> per-bit T-D3 with QSO retention, bright limit T-D5.
6. Epoch differences up to 1" for high-pm stars; 4.6% doubles (MAJOR) -> gaia_id path; isolation cut.
7. Code integration: registry, seeds, priors anchors, reference_priority, blend policy, hard-coded 41/36 in tests (MAJOR) -> separate versioned registry, per-bundle fallback/faint-limit bands, T-U1 redesign.
8. Tile-to-tile calibration (Redmine #33419) (MAJOR) -> T-D6.
9. Partial cone coverage biases f_pt (MAJOR) -> coverage from tile footprints and masks.
10-12 (MINOR): gal_ebv = 1.21-1.23 x SFD (not used); sentinels; wide_mode undocumented, deep tables excluded.

## Round 2, method (M2)
Status: M5, M7, M12, M14-M17 resolved; M1-M4, M6, M8-M11 partial; M13 not resolved -> all addressed in v2.
1. No calib role (BLOCKER) -> 55/15/10/20 fit/select/calib/test; knob/role table.
2. c_Q assumes morphology independent of spectroscopic selection; DESI targets point-like at z > 1 (BLOCKER) -> c_Q conditional on Legacy PSF; z < 0.8 flagged.
3. Stage 2 misses blend mode; decision rule undefined (MAJOR) -> lower 95% bound vs margin, SE pre-check, blend checks.
4. f_k constant in r; responsibilities mixed old/new blocks (MAJOR) -> logistic f_k(r), old-block responsibilities, T-D8.
5. s_b barely identifiable from QSOs (MAJOR) -> stars with W fixed from high-S/N rows.
6. Out-of-prior mass not computable (MAJOR) -> flat-prior likelihood to r = 26.
7-10 (MINOR): no faint-limit cuts on training rows; role leakage table; parametric window tails for sparse populations; T-V5 relative to the Legacy baseline.

## Round 2, data/engineering (D2)
Status: D1, D2, D4, D10, D12 resolved; D3, D5-D9, D11 partial -> addressed in v2.
1. 1.5" isolation cut removes 15% of QSOs, mostly chance neighbours (MAJOR) -> neighbour = vis_det = 1, non-spurious, within 2.5 mag; re-query counts.
2. gaia_id route: DR2 ids, duplicates (MAJOR) -> parent preparation already holds DR3 ids; duplicates by match quality.
3. Bright-star-mask bits on 38.5% of VIS 15-19 QSOs (MAJOR) -> QSO retention per bit in T-D3.
4. Tile medians confounded by stellar systems (Draco) (MAJOR) -> locus-perpendicular offsets, stellar systems masked.
5. Test yield seed-dependent (2.4k-6.1k QSOs) (MAJOR) -> stratified assignment.
6-9 (MINOR): coverage from footprints; VIS_TO_Y checked (99.94% finite, S/N 248 vs 197 Kron); provenance of the match cache; per-bundle fallback lists.
