# Complete-score replacement checks

The user authorized release checks and activation if satisfactory on 30 September.
Use only saved data and the frozen full-data bundle. No density fitting, new
acquisition or broad capacity/convergence work is part of this step.

Configuration is frozen in `configs/full_sample_release.json` before scoring:
128 northern and 128 southern test-role rows per population, paired across
models; QSOs also lie in the original held-out cells. Six practical band
selections use the complete public scorer, including priors, catch-all and hard
support guard. The stellar primary-redshift distribution matches the sampled
QSO redshifts. Wrong-primary-redshift QSO checks explicitly retain field_q.
Clean separation/fracflux values are test fixtures to isolate photometry, not
measured close-pair labels. Samples are star-dominated, not certified stars.

All eligible scores must be finite with bounded probabilities and the exact
p_same = exp(log_r) * dz identity. Outside-both rows must be excluded with NaN
posterior/rank. Review ranking discrimination with exclusions retained at the
bottom of the ranking; an AUC loss over 0.03 is a declared review trigger,
not a threshold tuned after seeing results. Revisit the saved bright cases
through the complete scoring interface at five fixed primary redshifts.

Both Legacy hemispheres are tested on identical 15x15 colour grids at three
magnitudes. Compare tails on points low in both models' QSO and stellar
intensities; a guarded high-QSO rate increase over two percentage points
triggers review. Record all exceptions. A synthetic grid is a stress test, not
a prevalence sample or a completeness calibration.

If the checks pass and the targeted cases disclose no blocking defect, copy the
hash-verified bundle to an immutable release directory, save the old pointer for
rollback, and atomically update current. Preserve probability-calibration and
external-area limitations. Update state, journal and method status; run the
suite, commit and push. No claim of comprehensive 41-band subset calibration.
