# Unified model and empirical contaminant support: practical plan

The user separates North/South unification from extreme-outlier rejection and
sets the science scope at the 5-sigma detection limits. Proceed with the shared
latent model. Do not launch the previously suggested faint-end stellar refit
campaign. This document is a plan; no production retraining has started.

## Detection scope

**Clarification accepted after the plan:** train on all quality-eligible rows,
including sources below five sigma, with the existing held-out roles and no
arbitrary caps. Five sigma defines the main validation/reporting domain; it
is not a new training cut or an automatic exclusion from likelihood evaluation.
Keep below-limit results as diagnostics and do not let their density changes
alone block release. Use native flux/error and actual survey detection limits,
not a universal magnitude-24 cut. Preserve weak/negative measurements and
arbitrary nonempty subsets of the 41 inputs. Detection metadata can come from
the parent catalogue even when its detection band is omitted from the chosen
likelihood subset. Keep background counts and population priors consistent
with the actual parent selection; a reporting threshold alone does not change
that selection or justify silently truncating the fitted density.

When a requested likelihood subset contains a band detected at >=5 sigma,
choose a qualifying reference band using a fixed recorded priority. Keep the
native reference-band prior and error covariance consistent. If the subset
contains only weak bands, retain likelihood evaluation with an explicit
weak-reference flag; do not silently drop the bands or invent colour support.
Single-band input has no colour information and an uninformative colour-support
test. These cases do not motivate a new faint-source fitting campaign.

A read-only re-summary of saved optical predictions finds northern stellar
mean pooled-minus-separate conditional log density -0.02793 nats among 35,992
objects with >=5-sigma detections in the original reference band, versus
-0.17063 across all 60,825. The corresponding northern QSO loss is -0.03042
nats; both are inside the old 0.1-nat tolerance. However, selecting >=5 sigma
in *any* grz band leaves 59,104 northern stars and a -0.17012-nat loss with the
old reference choice. Of 13,821 northern stars with reference asinh magnitude
>=24, 13,122 have a >=5-sigma detection in another grz band. Thus the earlier
faint bin cannot simply be called undetected. Re-evaluate the saved models
with the stronger reference choice before fitting anything further; measure
ranking and conditional-density changes consistently under that same choice.
Update: the reference change is implemented; the full-survey pilot and matched
comparison passed the density/ranking checks. See UNIFIED_PILOT_2026-09-30.md.

Numbers and scope are recorded in
`POOLED_OPTICAL_DETECTION_DOMAIN_2026-09-30.json`. The re-summary uses the exact
inverse of the saved asinh transformation/error propagation and does not
recompute a likelihood or change a model.

## Incorporating the supplied outlier note

Source: the user's `docs/xd_qso_contaminant_ood.tex` (left unchanged).

1. **Reuse the existing empirical background.** Despite its name,
   `clean_stellar_rows` keeps mask-clean PSF catalogue sources and removes known
   QSOs; it applies no Gaia-star, stellar-colour or positive-flux selection.
   It therefore already contains unresolved contaminants, including objects
   not confirmed to be stars. Describe the new component as the **PSF
   non-QSO/contaminant background**, dominated by stars, rather than a pure
   stellar model. It excludes extended objects and unusable blends because
   these are outside the deployment population. Verify the existing training
   and deployment quality selection once, reusing stored masks/photometry.
   Keep known-QSO removal: their small global abundance does not guarantee
   negligible contamination specifically in QSO-like colour space.

2. **Keep the existing hypotheses.** Same-redshift QSO, other-redshift QSO,
   empirical PSF contaminant background, and the broad unmodelled component.
   The note's binary QSO formula explains a generic classification task; it
   must not remove our other-redshift competitor or replace the project's
   window-independent `log_r_per_unit_z` ranking statistic. Preserve the
   Student-t catch-all and the existing hard guard while testing the addition.
   Use one background intensity partitioned between fitted contaminants and
   the catch-all as currently defined; do not count a second full background
   population on top of it.

3. **Add a separate, calibrated QSO-support percentile.** Evaluate the
   candidate's QSO conditional log likelihood under the same native system,
   observed-band mask, reference magnitude and measurement covariance as the
   scoring likelihood. Generate the comparison distribution by sampling that
   same conditional predictive mixture, including the projection and noise
   once. For same-redshift ranking, test support at the target redshift/window;
   an all-redshift QSO-support diagnostic answers a different question and
   must be labelled separately. The same conditional noise treatment includes
   the reference-band uncertainty and induced correlations.

4. **Calibrate and abstain.** Use calibration-role spectroscopic QSOs at their
   known redshifts to measure retained completeness and set the support cutoff
   in configuration. Validate on real held-out QSOs and the known regression
   points. A model-generated percentile alone cannot certify empirical
   training coverage of a broad or poorly constrained component. Report the
   percentile separately from evidence/posterior; do not multiply it into a
   posterior and call that calibrated. Mark failed support as unsupported and
   exclude it from science ranking, retaining diagnostic likelihoods. For a
   first implementation, compute support for proposed high-ranking candidates
   and calibration/validation samples; cache only reference distributions
   with matching conditioning, masks and noise conventions. No exhaustive
   enumeration of all 41-band subsets is required.

This adds one support mechanism to the existing background/catch-all machinery.
It does not require a new contaminant catalogue, an extra pure-star fit, a
global evidence statistic, or covariance tuning to suppress selected outliers.

## Implementation and retraining order

1. **Freeze the selection and mapping.** Preserve the existing quality-based
   training selection, every eligible row, and frozen fit/select/calib/test
   roles. Record the 5-sigma validation domain and reference rule separately.
   Re-summarize the optical prototype in that science domain. Establish row
   counts before launching training; do not start new acquisitions or impose
   an object-level or per-band 5-sigma training cut.

2. **Extend the tested observation operator to the full model.** Keep all 41
   external band labels. Tie Legacy North/South grz through the tested forward
   relation in one latent model; all other surveys retain their own systems.
   Use shared Legacy W1/W2 latent coordinates after expressing those native
   fluxes in common units and a common asinh softening convention. They remain
   distinct from AllWISE W1/W2 measurements. Fully sharing the five duplicated
   Legacy coordinates gives 36 latent coordinates behind the 41-input
   interface. Verify missing-band projection, negative-flux handling, native
   reference conditioning, serialization and optional local background refits.
   Retain the prototype's simple class offsets and instrumental scatter;
   do not fit 43 separate redshift-dependent correction laws.

3. **Run one pooled full-data training campaign.** Refit all 43 joint QSO
   slices and one joint empirical contaminant model. Reuse the existing
   component counts and frozen data roles, with a declared iteration budget,
   checkpoints and, following the user's later request, eight numerical workers:
   four QSO slice workers plus four cooperating projected stellar E-step
   workers. The full launcher and preflight are now prepared; see
   `UNIFIED_FULL_PREFLIGHT_2026-09-30.md`. Report completed work and estimated fraction of compute.
   Do not reopen a broad component-count search. No production launch is part
   of this planning step.

4. **Complete the new bundle.** Re-estimate HEALPix component weights and
   contaminant surface densities for the selected parent catalogue, preserving
   magnitude dependence, Galactic dependence and candidate-local refits.
   Rebuild the catch-all in the new coordinates and calibrate support on the
   reserved calibration role. Reuse QSO abundance priors only where their
   native reference system and selection remain applicable; record any
   unsupported selection/area rather than inventing a calibrated posterior.
   This does not turn full probability calibration into a new prerequisite
   for a working ranking model.

5. **Run focused release checks and activate the replacement.** Check QSO
   ranking/completeness at supported detection levels, North/South paired
   consistency, support-cut QSO retention, real PSF high-score incidence,
   the original northern and remaining southern regression points, and
   representative arbitrary band masks. Random unlabelled PSF objects measure
   high-score incidence, not true contamination without labels. Keep the
   existing model available for rollback; replace the active pointer when the
   agreed checks pass. Document limitations without expanding into unrelated
   convergence, ultra-faint-source or all-subset calibration campaigns.

The old test data have been inspected and helped choose the prototype scatter
model. Describe them as reserved development/regression data, not untouched
confirmation. Fit the new support threshold on the calibration role and keep
the test role out of that fit.

## User-facing classification inputs

The intended convenience interface accepts candidate RA and Dec (ICRS degrees),
named survey bands with fluxes/errors or magnitudes/errors, and the primary
QSO redshift whose same-redshift hypothesis is being tested. It also needs
PSF/quality/blend information for science eligibility. Fluxes are preferred
for weak or negative measurements. Convert RA/Dec internally to Galactic l,b
for spatial weights/counts and to the default Legacy North/South footprint.
Use actual survey/release provenance when supplied, especially in the overlap;
position alone does not identify which instrument produced a measurement.

The current low-level PSF scorer instead takes l,b explicitly and identifies
the native system through survey-labelled bands. The RA/Dec convenience layer
is now implemented and tested by the full-survey pilot. The input redshift is
the primary QSO redshift, not an assumed known redshift of its candidate.
