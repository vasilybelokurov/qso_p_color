# Review of `qso_binary_color_probability_plan.md`

Written 2026-09-18, after implementing the core of the plan and testing it.
Every numerical claim below was measured, either against WSDB or against the
synthetic universe in `tests/test_score.py`; nothing here is asserted from
memory.

## Summary

- **The statistics are right and I kept them.** The three-hypothesis structure
  (`same_z` / `field_q` / background), the likelihood-times-surface-density
  factorisation, always reporting a prior-independent Bayes factor, exact
  missing-band marginalisation, and the refusal to call a likelihood a
  probability — all correct, all implemented as specified.
- **The plan omits its own binding limitation.** A ±2000 km/s window at
  *z* = 1.4 is Δ*z* = 0.016, while broadband colours pin a quasar redshift to
  σ_z ≈ 0.1–0.3. So `P_sameQ` under a velocity window is ≲ 0.3 *even for an
  object sitting exactly on the locus* — verified in
  `test_a_velocity_window_is_far_narrower_than_any_colour_redshift`. Colours
  rank candidates; they cannot establish a velocity-scale redshift match. This
  must be stated in the output contract, not discovered later.
- **Blending, not statistics, is the binding constraint**, and the plan treats
  it as a flag (§4.6). Measured: of 10,636 DESI DR1 quasar "pairs" closer than
  3″, 9,577 sit at < 0.5″ with identical redshifts — they are the *same object
  entered twice*. The survey pipeline does not deliver two independent
  measurements at those separations, so a separation floor is a scientific
  parameter, not a warning.
- **A labelled validation set exists and should be built first, not last.**
  DESI DR1 yields ≈ 1,600 spectroscopically confirmed same-redshift quasar
  pairs and ≈ 8,400 projected quasar–quasar pairs at 3–20″. The plan puts
  calibration in Milestone 7; without it the whole package returns an
  unvalidated Bayes factor.
- **Four scope changes**: redshift-conditional slices instead of a joint (*c*,
  *z*) mixture for v1; no magnitude conditioning of the quasar colour model in
  v1; the empirical background rather than an explicit star model as the
  default denominator; and our own ~150-line extreme deconvolution instead of
  `astroML` / `extreme_deconvolution`.

---

## 1. What I would not change

The following are correct as written and are now implemented and tested:

| Plan section | What it gets right | Test |
|---|---|---|
| §2, §8 | Separating the colour likelihood from the surface density, so a changed prior moves the posterior and not the evidence | `test_bayes_factor_is_invariant_under_a_prior_shift` |
| §1.2, §8 | Including `H_fieldQ`. Without it a genuine quasar at the wrong redshift scores as evidence *for* a pair | `test_a_quasar_at_the_wrong_redshift_is_not_evidence_for_a_pair` |
| §4.4 | Never assuming independent colour errors | `test_asinh_colours_share_bands_so_are_anticorrelated` |
| §4.5 | Exact marginalisation over missing bands, not median imputation | `test_missing_dimension_marginalises_exactly` |
| §4.2 | Preserving negative fluxes as real measurements | `test_negative_flux_is_preserved_not_clipped` |
| §21.17 | Returning `null` posteriors rather than a fabricated prior | `test_no_prior_means_no_posterior_not_a_made_up_one` |
| §16.9 | Refusing to score SDSS features with a Legacy model | `test_photometric_system_mismatch_raises` |

The instruction in §21 not to hard-code scientific thresholds is followed
throughout: `RedshiftMatch` has no default window at all and raises unless the
user declares one.

---

## 2. The limitation the plan does not state

The plan asks for `P(Q, z ~ z0 | colours, m, l, b)` and specifies a velocity
window (§1.3). Those two things are in tension.

At *z*₀ = 1.4, a ±2000 km s⁻¹ window is

    Δz = (1 + z0) Δv / c = 2.4 × 2000 / 299792.458 = 0.016

Broadband quasar photometric redshifts are, at best, σ_z of order 0.1 and
frequently multi-modal — that is the central result of the XDQSOz paper
(Bovy et al. 2012, <https://arxiv.org/abs/1105.3975>), whose whole purpose was
to produce a quasar redshift *PDF* rather than a point estimate. The window is
therefore ~10× narrower than the measurement. The consequence is arithmetic:
even a perfect on-locus candidate puts only a small fraction of its quasar
intensity inside the window, so

    λ_fieldQ ≫ λ_sameQ  ⟹  P_sameQ ≲ 0.1–0.3.

This is not a defect in the method; it is the honest answer. But it means:

1. **`P_sameQ` under a velocity window is a ranking statistic, not a
   probability anyone should quote as "the chance this is a binary".** The
   package reports it, and reports what window produced it, in every row.
2. **Report the posterior at two window widths.** The physical one the user
   cares about, and a photometric one (Δ*z* ≈ σ_z,phot) answering the question
   colours can actually answer: "is this consistent with *z*₀ at the precision
   the photometry allows?"
3. **The prior-independent `log_bayes_factor_qz_bkg` is the most robust number
   the package produces** and should lead the candidate report.

Implemented: `RedshiftMatch` takes either a velocity or a Δ*z* window, and the
scorer records the definition in every row.

---

## 3. Model form: slices beat a joint mixture for v1

The plan's §3.1 baseline is a single extreme-deconvolution mixture over the
joint vector `[colours, z]`, conditioned analytically on *z*. I implemented
both that (`JointColourRedshiftModel`) and a set of redshift-conditional
mixtures (`SlicedColourRedshiftModel`), and made the sliced one the default,
for two reasons.

**Linearity.** Inside a joint Gaussian component the colour–redshift relation
is a straight line. The quasar track is strongly non-monotonic — Lyman-α and
the major broad lines crossing band edges — so the conditional at a given *z*
is built partly from components extrapolating along straight lines away from
where they were fitted. Enough components fix this, but "enough" is set by the
curvature, not by the data volume, and it is not diagnosable from the fit.

**Declared versus implicit redshift prior.** Conditioning the joint mixture on
*z* reweights components by their *z*-marginal, which is the training sample's
redshift distribution — i.e. the *spectroscopic targeting selection*, not the
quasar population. The conditional density is still mathematically correct, but
the redshift prior enters silently. With slices, each mixture is normalised over
colour space at fixed *z*, and the redshift prior is supplied explicitly as
Σ_Q(*z*, *m*). The plan's §3.4 formula inherits this and does not say so.

This is a preference, not a proof. Both backends share an interface, and which
predicts held-out quasar colours better is the empirical question the validation
step should settle.

## 4. Drop magnitude conditioning of the quasar model in v1

§3.1 and §13.2 condition the quasar colour model on reference magnitude via
overlapping bins. I would not do this first, for a specific reason: **the
dominant magnitude dependence of observed colours is photometric noise, and
that is already handled exactly** by convolving each component with the
candidate's own covariance. What remains is the intrinsic colour–magnitude
dependence at fixed redshift, which is weak (host contribution at low *z*, some
luminosity dependence of line equivalent widths). Against that, magnitude
binning multiplies the number of fitted models by ~5 and dilutes the statistics
of every one.

Keep the interface magnitude-aware, default to one bin, and let the validation
report decide — "does held-out log density improve with magnitude bins?" is a
one-line experiment. The *background* model does need magnitude conditioning,
and has it, because which populations are reachable genuinely changes with depth.

## 5. Use the empirical background, not a star model, as the default denominator

§5.3 offers an explicit star sample (Strategy A) and an empirical point-source
background (Strategy B) as equals. They are not equals for this problem.

The competing hypothesis is *"a chance projection near the primary"*. What
chance projection draws from is **everything in the imaging catalogue** at that
brightness and position — stars, compact galaxies, unclassified blends. That
population needs no selection function, because it *is* the population being
counted; the number density and the colour distribution come from the same
catalogue with the same quality cuts as the candidate. A Gaia-selected stellar
sample is purer but is not representative at *r* ≳ 20, which is where the
candidates are.

Strategy A remains implemented (`mode='star'`) and is the right thing for the
two-class quasar-versus-star number the brief explicitly asks for. It should not
be the default for ranking real candidates.

## 6. Blending is the binding constraint

§4.6 is correct and too brief. The evidence:

| separation | DESI QSO–QSO pairs | of which Δv < 3000 km/s |
|---|---|---|
| 0–0.5″ | 9,577 | 9,549 |
| 0.5–3″ | 1,059 | 958 |
| 3–5″ | 545 | 212 |
| 5–10″ | 2,052 | 499 |
| 10–20″ | 7,423 | 900 |

The 0–0.5″ row is not a population of ultra-close binaries. It is the same
object entered twice under two `targetid`s with the same redshift — the survey's
own catalogue does not resolve them as two things. Two consequences:

1. **A separation floor is a scientific configuration parameter.** Below
   roughly 3″ the Legacy Surveys model fit assigns flux between overlapping
   sources in a way that makes the two colour vectors neither independent nor
   individually trustworthy. Catalogue-based scoring below that floor should
   return a status code, not a number. Image-level forced photometry is the
   only honest route there, and that is a separate project.
2. **Validation must be done at matched separation and matched `fracflux`.** A
   model validated on isolated quasars says nothing about performance on
   companions 4″ from a bright primary, which is the only case we care about.
   The pair sample above supports exactly this stratification.

**Also: de-duplicate by sky position, not by identifier.** `zcat_primary`
de-duplicates per `targetid` and still leaves 9,577 repeated objects (0.6 % of
the quasar sample). Training on those silently up-weights them.
`scripts/build_pair_validation.py::deduplicate` does the positional version and
reports what it removed.

## 7. Build the validation set first

The plan puts calibration in Milestone 7 and the physical-pair prior in
Milestone 8. I would move labelled-sample construction to Milestone 1, because
it is the thing that determines whether any of the rest is worth doing, and
because the sample turns out to be easy to build and well populated:

- **positives**: ~1,600 confirmed same-redshift quasar pairs at 3–20″;
- **hard negatives**: ~8,400 projected quasar–quasar pairs — the `field_q`
  class, which a two-class star-versus-quasar score cannot distinguish from a
  positive even in principle;
- **easy negatives**: every spectroscopically confirmed star or galaxy within
  20″ of a quasar.

Both objects carry spectra, so the labels are independent of the colours being
scored. This supports reliability curves, precision–recall, and the
magnitude/latitude/separation stratification the plan asks for in §9.3 — on real
data rather than in principle.

## 8. Dependencies: implement extreme deconvolution rather than depend on it

§10 lists `astroML` for the baseline XD. Neither `astroML` nor
`extreme_deconvolution` is installed in `~/Work/venvs/.venv`, the latter needs
GSL built against it, and neither handles per-object *missing dimensions*, which
this problem has constantly (a band with `flux_ivar = 0`). The EM update of
Bovy, Hogg & Roweis (<https://arxiv.org/abs/0905.2979>) is about 150 lines of
numpy. `src/qso_pcolor/xd.py` implements it, and
`test_xd_recovers_intrinsic_width` asserts the property that actually matters:
fitting noisy draws recovers the *intrinsic* scatter, where a noise-blind GMM
recovers the broadened one.

One bug found this way and worth recording: the missing-dimension branch of the
M step had a transposed array in the `B_ij` cross term. Diagonal variances came
out right and only the off-diagonals were wrong, so it would have passed any
test that did not check a correlated covariance with bands missing.

## 9. Smaller corrections

1. **§3.3 conditional weights** should be `α_k N(z0 | μ_z,k, V_zz,k + σ²_z0)`
   when the primary redshift has an uncertainty. The plan mentions kernels but
   not this convolution. Implemented in `condition_joint(..., y_var=...)` and
   tested against a numerically convolved reference.
2. **§7 Σ_Q modes** are all heavy. A defensible v1: histogram a spectroscopic
   quasar catalogue over a stated area with completeness *C* = 1 by default.
   Since *C* ≤ 1 understates the quasar density, `p_sameq` is then a **lower
   bound** — a clean, honest statement, and better than inventing a luminosity
   function. `EmpiricalQSOPrior` records whether a completeness was supplied.
3. **§10 package layout** is over-scoped: 30-odd files and 8 milestones for one
   research question. The implementation here is 7 modules. Extension points are
   preserved; speculative files are not.
4. **§12** is right that covariances must never be inverted, but note the
   consequence: with per-object noise *and* per-object missing dimensions,
   Cholesky factors cannot be precomputed once per component. Cost is
   *N* × *K* × *d*³/3, which is negligible at *d* ≈ 5 and matters only if the
   background model is ever scored over millions of rows.

## 10. Worth adding

- **Gaia astrometry as a separate likelihood factor** (§18 lists it as
  optional). For *G* < 20 a significant parallax or proper motion settles the
  question outright, and `decals_dr9.main` already carries `parallax`,
  `parallax_ivar`, `pmra`, `pmdec` — no extra join. Keep it factorised so its
  contribution stays inspectable.
- **Run the optical-only and optical+WISE models side by side.** W1 − W2 is the
  strongest single star/quasar discriminant in this data, but the WISE PSF is
  ~6″, so for a close pair it is also the most blend-contaminated. Disagreement
  between the two models is itself a blending diagnostic.
- **Quasar variability** is not mentioned anywhere in the plan. It is not
  uniformly available over the footprint, so it is correctly out of scope for
  v1, but it is the obvious next axis of evidence.

## 11. Data available on WSDB (verified 2026-09-18)

| Need | Table | Count / note |
|---|---|---|
| Spectroscopic quasars | `desi_dr1.zpix` + `desi_dr1.photometry` | 1,645,842 with `spectype='QSO'`, `zwarn=0`, `zcat_primary` |
| — joined to photometry | by `targetid` | **no positional crossmatch needed**; LS DR9 *grz*+W1–W4 |
| Independent quasar sample | `sdssdr16qso.main` | 750,414; `psfflux`/`psfflux_ivar` arrays (*ugriz*), different targeting channels |
| Background / candidates | `decals_dr9.main` | *grz*+WISE; `release` 9010 south / 9011 north |
| Deeper, +*i* band | `decals_dr11.main` | `release` 11010 south / 11011 north — **different photometric systems, model separately** |

The `targetid` join for the quasar training set is worth emphasising: it removes
crossmatch ambiguity entirely, in exactly the crowded configurations where a
positional match is least reliable.

## References

Only sources checked to exist are listed.

- Bovy et al., *Think Outside the Color Box* (XDQSO) — <https://arxiv.org/abs/1011.6392>
- Bovy et al., *Photometric redshifts and quasar probabilities from a single,
  data-driven generative model* (XDQSOz) — <https://arxiv.org/abs/1105.3975>
- Bovy, Hogg & Roweis, *Extreme deconvolution* — <https://arxiv.org/abs/0905.2979>
- Lyke et al., *SDSS Quasar Catalog: Sixteenth Data Release* — <https://arxiv.org/abs/2007.09001>
- Hennawi et al., *Binary Quasars in the SDSS* — <https://arxiv.org/abs/astro-ph/0504535>
- Eftekharzadeh et al., *Clustering on very small scales from a large sample of
  confirmed quasar pairs* — <https://arxiv.org/abs/1702.03491>
- Legacy Surveys DR10 catalogue columns — <https://www.legacysurvey.org/dr10/catalogs/>
- q3c (Koposov & Bartunov 2006) — <https://ui.adsabs.harvard.edu/abs/2006ASPC..351..735K>
