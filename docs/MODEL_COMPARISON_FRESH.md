# Original versus seven-survey model: identical Legacy DR9 data

Identical southern Legacy DR9 grz measurements; original dereddened relative fluxes versus new observed luptitudes. WISE excluded because Legacy forced WISE and ALLWISE are different measurements. Spectroscopically selected individual objects, not random imaging sources or a close-pair calibration. No matched new population prior; compare dimensionless Bayes factors and flat-prior redshift summaries only.

Evaluated 2,000 quasars, 530 stars, and 1,470 galaxies in 14 common reserved sky blocks. Only blocks recorded as held out by both quasar models are admitted; every background fitting/selection cone is excluded. Objects are positionally de-duplicated; the full JSON report records all cuts, seeds, and model paths.

| Metric | Original | Seven-survey |
|---|---:|---:|
| Quasar/non-quasar AUC | 0.9617 | 0.9590 |
| True/other redshift AUC | 0.7901 | 0.7912 |
| Central 68% redshift interval coverage | 0.7945 | 0.7875 |
| Median absolute dz/(1+z) | 0.1975 | 0.1263 |
| Quasar/star AUC | 0.9695 | 0.9530 |
| Quasar/galaxy AUC | 0.9589 | 0.9612 |

Paired 95% intervals resample whole sky blocks, retaining each object’s old/new scores together. The configured equivalence margin is ±0.02 AUC, declared before scoring; the entire interval must lie inside it.

- Quasar/non-quasar AUC: new minus old -0.0027, interval [-0.0063, +0.0009]; equivalence check passes.
- True/other redshift AUC: new minus old +0.0011, interval [-0.0044, +0.0055]; equivalence check passes.

Individual score agreement:

- log_bf: Spearman 0.941; median new minus old -0.443 nats; median absolute difference 1.115 nats.
- log_pz: Spearman 0.872; median new minus old +0.001 nats; median absolute difference 0.169 nats.

Near-equal AUC, when present, does not establish interchangeable individual scores. This comparison does not refit either model. The original model remains available.

The old model uses g/r and z/r after marginalising its missing WISE dimensions; the new model conditions the three observed luptitudes on r. Raw likelihood densities have different units and are not compared. The same true and assigned primary redshifts enter both calculations. QSO/non-QSO AUC scores quasars at their true redshifts and non-quasars at redshifts drawn from the held-out quasar population; true/other-z AUC uses each quasar at its true and one drawn redshift outside ±2000 km/s.

All g,r,z measurements must be usable, with maskbits=0, r S/N≥5, 17≤dereddened r<22.5, |b|≥25°, release=9010, 3–30 arcsec catalogue separation, and finite fracflux_r≤0.2. Negative non-reference fluxes are retained. This defines a matched diagnostic sample, not the prevalence of classes in the sky.

Reproduce with `python scripts/compare_old_new_models.py --config CONFIG`, using the configuration recorded in the [full results and hashes](MODEL_COMPARISON_FRESH.json). The source catalogue is local. The hash-keyed score cache records source rows, target IDs, coordinates, assigned redshifts, and both model outputs. [Other survey combinations](MULTISURVEY_VALIDATION.md).

This confirmation sample excludes every target in `data/multisurvey/old_new_comparison.npz` before the random draw. It uses previously unscored objects in the common spatial holdout; the sky blocks are shared with the diagnostic sample, so this is an object-level confirmation, not a new-footprint test.
