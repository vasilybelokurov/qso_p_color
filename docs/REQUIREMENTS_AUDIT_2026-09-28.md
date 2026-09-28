# Implementation against the scientific requirements

This is a requirements audit, not a new method write-up. Its reference is the
original specification, the explicit PSF-source restriction, and the user's
requirement for both a survey-wide HEALPix background and a candidate-local
refit. A later plan's deferral does not remove those requirements.

The September 28 recovery report checked scoring safety and predictive
performance but understated missing functionality. In particular, its statement
that the intended design had recovered was too broad. The pooled PSF bundle
did not implement the required dependence on Galactic position.

## Requirements and disposition

| Requirement | State before this revision | Disposition |
|---|---|---|
| PSF-like base population | Implemented in the baseline selection; unknown and extended types rejected | Retain. The field includes unresolved non-quasars, not a spectroscopically purified stellar sample. |
| Consistent morphology, hemisphere and quality rules | Candidate/field adapters share the rule. Published quasar training predates the release-9012, latitude and nobs corrections | Future training fixed in the preceding recovery; historical model provenance remains explicit. |
| Colour- and magnitude-dependent field | Four magnitude-bin mixtures per hemisphere | Retain their noise-deconvolved colour components. |
| Galactic-position-dependent field colours | Missing from the saved PSF baseline, although old hierarchy code survives | Implemented and active: fitted component proportions in HEALPix cells with parent/global pooling. |
| Galactic-position-dependent field surface density | Missing from the saved PSF baseline | Implemented and active: counts divided by usable area, with separate pooling of cell/parent rates. |
| Configurable resolution, selected from data | HEALPix existed only as sampling/split infrastructure in the current baseline | Implemented: select the field nside and pooling on reserved cones inside populated fitting cells. Test whole cells separately. |
| Candidate-local background refit | Earlier helper lacks the current PSF selection and bundle integration | Implemented: a PSF cone fit of population proportions and surface density, tied to the baseline; exclude primary and companion and require measured area. |
| Quasar colour-redshift likelihood | Implemented as overlapping XD slices | Retain. Slices instead of a joint colour-redshift model are an explicit early design choice, not a missing sky capability. |
| Quasar magnitude dependence | Deliberately absent from colour slices; retained in surface-density prior | Retain the documented choice to avoid imprinting targeting-related magnitude trends. |
| Same-z, other-z quasar and background competition | Implemented, with an additional unmodelled term | Retain all terms; no two-class replacement of the ranking. |
| Likelihoods separate from priors/posteriors | Implemented in the shared scorer | Retain. R includes the adopted population prior and is not a prior-free Bayes factor. |
| Full colour covariance, negative flux, missing bands | Implemented in the feature and mixture layers | Retain and exercise through spatial/local tests. |
| Optical-only operation | Core marginalisation works, but the current baseline requires three relative-flux dimensions; grz supplies only two | Still incomplete in the science adapter. Requires an explicit band-mode policy and matched validation, not a claim that the current route already supports it. |
| Low reference-S/N handling | A low-S/N flag is produced, but the current science eligibility gate does not exclude it | Still a limitation. The saved min_ref_snr describes a flag threshold, not an enforced selection cut. |
| Blend protection and model-support diagnostics | Required explicit policies; missing blend measurements and outside-both-model cases excluded | Retain; neither gate establishes probability calibration. |
| Surface-density priors and selection bias | Empirical priors implemented; completeness and unrecognised-quasar correction remain assumptions | Retain transparently and assess independently; the tail discrepancy is not resolved merely by restoring sky dependence. |
| Redshift-window definition and output provenance | Explicit match object, effective width and bundle ID implemented; row-level policy configuration was missing | This revision adds the complete match/blend/support configuration and its hash, plus background mode and nside to each scored row. General catalogue exports remain incomplete. |
| Calibrated probabilities and selection performance | Ranking comparisons exist; complete representative calibration by magnitude, position and error does not | Still incomplete; ROC/AUC is not a substitute for reliability, log loss and precision/recall. |
| General candidate CLI and diagnostic report | Examples and validation scripts exist | General catalogue-to-Parquet scoring and per-candidate reports remain incomplete. |
| Physical-pair/clustering probability | Deliberately deferred in the original specification | Outside this restoration; no invented clustering multiplier. |
| Other survey systems and broader morphology | Separate multi-survey model exists with known field limitations | Do not claim these are validated by the Legacy PSF restoration. |

## Spatial restoration now active

The survey-wide field has the form

`Sigma_B(m, l, b) * sum_k w_k(m, l, b) N_k(colours; measurement covariance)`.

The existing component means and covariances are shared over the sky.
Proportions are fitted per sky/magnitude cell, then pooled cell -> parent ->
hemisphere. The count rate has its own pooling strength. Keeping component
shapes fixed follows the spatial design already recorded in
`PLAN_FIELD_MODEL.md` section 3; it also isolates spatial structure from changes
to the quasar model or the underlying colour coordinates.

The local mode refits the same proportions and counts in a declared cone,
pooled towards the spatial prediction there. It is a local mixture-weight fit,
not a claim that every Gaussian mean and covariance has been re-estimated.
The scorer refuses to use this cone background outside its recorded scope.
The original all-parameter local XD helper remains a separate legacy path.

The current unrecognised-quasar correction is inherited for this comparison.
Its probabilities are calculated against the fixed pooled baseline; restoring
spatial variation does not by itself establish that this correction is right.
The broad outlier component is likewise retained. All shared Gaussian shapes
are unchanged, preserving its covariance-envelope construction.

## Validation that addresses the missing capability

1. Controlled skies must recover both a colour-population gradient and a
   count-rate gradient. The same photometry must receive a different field
   likelihood at different positions for a known reason.
2. Empty coverage falls back; an observed cell with zero counts contributes a
   measured low density. These are different cases.
3. Pooling selection withholds cones inside populated cells, so it can see
   local adaptation. Withholding every child and its parent cannot tune it.
4. Reserved field cones measure predictive density and counts. Cones crossing
   a partition boundary are excluded whole, with their areas, before fitting.
5. Local tests verify PSF selection, source exclusions, area requirements,
   scope restrictions, provenance and the effect of the fitted background.

The component shapes and original priors were already fitted with the baseline
partitions. Within-fit-cone selection is therefore conditional validation of
the new spatial parameters, not a fresh validation of those global shapes.
Northern fit/select checks are available, but no untouched northern field test
cells exist in the current sample.

## Interpreting the earlier convergence finding

The median 0.028 change in ln R after further quasar fitting is about 2.8% in
R. It narrowly exceeds a chosen 0.02 tolerance and has not been shown to cause
worse ranking. The archived southern Legacy fit also capped 35/43 slices,
against 36/43 now. This is a secondary numerical check, not evidence that the
method has dramatically deteriorated. Missing required sky dependence is a
more fundamental implementation gap and is the priority of this revision.

The active artifact and measured spatial results are recorded in
[SPATIAL_BACKGROUND_2026-09-28.md](SPATIAL_BACKGROUND_2026-09-28.md).
The full-mixture legacy alternative was discussed but is not the implementation:
this revision uses the shared-shape, variable-weight design requested by the user.
