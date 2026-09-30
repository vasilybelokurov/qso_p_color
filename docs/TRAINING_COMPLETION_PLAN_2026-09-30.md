# Proposed plan to complete full-sample training

This is a proposed follow-up plan, not a new training launch. The original
density-EM budget is complete: 100% overall, QSO and stellar. The saved candidate
is `models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/`.

## 1. Diagnose the saved fits

Use the saved histories to report recent likelihood gains, convergence tests,
component weights and covariance conditioning for all 43 QSO slices and the
stellar fit. Distinguish slow improvement from numerical failure. Check the
selection-stage histories as well as final fits. Deliver a per-fit table and
an explicit continuation list before committing substantial compute.

Known facts: 25 QSO final fits converged; 18 reached 300 iterations. The stellar
K=20 fit also reached 300. Its last recorded likelihood gain is about 0.000194
per object, versus a convergence threshold of about 0.0000381. This establishes
that the configured criterion was not met; it does not establish unstable
predictions. Twenty-three QSO slices and the stars selected the largest tested
component count, 20. Selection candidates were allotted only 40 iterations.

## 2. Resolve component selection before unnecessary final refits

Continue relevant selection-stage checkpoints on the original fit-only rows,
evaluate on the frozen selection rows, and check that candidate rankings settle.
Expand the component-count grid where the settled results still favor its upper
edge, using a documented configuration and held-out score differences with
uncertainty across sky blocks. Determine the expanded grid and compute budget
from the diagnostic report. Do not assume that reaching the upper edge proves
that more components are needed.

The final shapes already include selection rows: their density on those rows
cannot serve as held-out model-selection evidence. Calibration rows remain for
catch-all fitting; test rows must not be used to choose K or stopping rules.

## 3. Finish the selected shapes and rebuild spatial weights

For unchanged K, continue unconverged final fits from their saved checkpoints.
Retain already converged final fits when their selected K is unchanged. If K
changes, refit the affected final model on all its fit+selection rows. Preserve
the original artifacts and record checkpoint lineage in a new candidate run.
The restart path must explicitly handle changed iteration budgets and saved
final-result files; merely rerunning the existing command would reuse results.

Use up to four QSO slice workers, then four stellar E-step workers, keeping
the expensive phases within the established worker budget. Continue in measured
blocks with checkpoints and unchanged convergence tolerance. A block ending is
a review point, not a declaration of convergence. Investigate stalled or invalid
fits rather than weakening the criterion. Rebuild spatial stellar weights after
the final shapes settle. Verify finite densities and valid covariances.

The last measured stellar iteration took about 85 seconds with four workers:
100 additional iterations would take roughly 2.4 hours at unchanged K, excluding
selection work, final evaluations and spatial assembly. This is a compute-block
estimate, not an estimate of iterations needed to converge.

## 4. Complete population priors and the catch-all

Adapt the population-completion path to the current input and frozen-role layout.
Rebuild QSO and stellar surface-density priors with explicit coverage, effective
area and completeness assumptions. Preserve Galactic spatial dependence and
all 41 reference-band interfaces; keep unsupported priors explicit. Fit the
active noise-convolved Student-t catch-all on calibration-role data using the
settled shapes, retaining the hard support guard. Do not claim that fitting the
catch-all establishes full posterior calibration.

## 5. Validate the complete candidate

Freeze the candidate and acceptance criteria before opening the reserved test
results. Check predictive densities and score stability by redshift, magnitude,
hemisphere and Galactic position; spatial stellar counts; optical-only and
other observed band subsets; negative fluxes; single-band behavior; and sparse
colour regions where both fitted populations have low density. Numerical tests
must verify joint marginalization across the 41-band interface. Empirical
validation cannot certify every possible subset; report support limitations.
Repeat the known erroneous high-QSO tail cases and compare with the active model
on identical rows. Keep likelihood evidence separate from calibrated probability
claims. Full probability calibration remains a separate, previously deferred task.

## 6. Activate and document the result

Promote only after the convergence/capacity review and declared validation gates
pass. Preserve a rollback pointer and deliver a concise release report. Update
the method description for any method change, PROJECT_STATE.md after substantive
progress, and JOURNAL.md with measured findings. No new downloads are needed.

For additional fitting, report a new, explicitly scoped compute budget and ETA.
Keep the original run's 100% completion distinct from follow-up fitting and the
remaining scientific pipeline. Report outcomes and next actions at each stage.
