# Current project state

**Standing priority:** deliver a pragmatic working model. Use the simplest
defensible solution that meets the requested functionality and focused
validation. Each additional investigation must resolve a concrete decision or
measured failure and have a stopping criterion. Keep optional improvements
separate from release blockers; document non-blocking limitations and proceed.
Report the conclusion, practical impact and next action. The user authorized
the bounded pooled-grz prototype on 30 September; production remains unchanged.

**FULL UNIFIED REFIT RUNNING (resumed 30 September ~22:40 BST).**
Run `models/multisurvey_psf/work/unified_full/20260930/a637f155f7f48d00` (MAP update,
tol 1e-4, max 300, predictive stopping every 20 updates at 0.02 nats; stopping panels
in `stopping/`, excluded from final assessment). Launch `python scripts/train_unified_full.py --fit`
(detached, log `train.log`); `touch <run>/PAUSE` pauses every fit at its next stopping
check, rerunning --fit resumes exactly. Paused once by design and checked
(`pause_check.json`, `scripts/check_refit_pause.py`), all pre-set criteria PASS:
background held-out density -1.980 -> -1.254 nats at update 40; with only the background
swapped, complete-score AUC change over 43 slices median -0.0001, range -0.0014..+0.0004;
random 1,024-object panel high-QSO count 5 -> 4 (six tail objects move both ways); one
score-panel object 0.417 -> 0.826. Live stopping records valid; qso_15 stopped at 60 on
a plateau; large-slice trajectories reproduce the MAP test exactly. Remaining: per-slice
first-block checks for the other 39 slices (`--stopping-only`, read-only) as they arrive;
then bundle completion, rejection calibration, release checks. Active model unchanged.

**MAP TEST FINISHED (30 September).**
Report: `models/multisurvey_psf/work/map_update_test/20260930/c108415a6a7a695f/report.json`.
Formal result 1/4 checks. (1) Zero MAP objective declines in all six (additive: 41 on the
z1.75 control): pass. (2) Tolerance 1e-4 not reached by 120 in any MAP fit: replaced by
predictive early stopping (implemented, `09bbd00`). (3) Held-out density MAP-additive
+0.473/+0.481/+0.538/+0.440/+0.449 nats (z1.45/1.65/1.75/2.35/3.25; block-bootstrap SE
0.05-0.08) but -0.566 at z4.05 (2,152 rows, overfits; user accepts z>4 weakness; the
stopping rule keeps the warm start there). (4) AUC loss >0.002 at z1.65 (-0.0025) and
z1.75 (-0.0033); bootstrap 95% intervals include zero, and the loss is almost entirely
one background-panel object (row 81) scored p_quasar=1.000 by BOTH models, i.e. an
ordering change among certain-QSO scores, not a classification change.
Prepared refit: `models/multisurvey_psf/work/unified_full/20260930/de2901455863c3bc`
(MAP, tol 1e-4, max 300, stopping every 20 updates at 0.02 nats; 34,259 QSO + 1,024
background stopping rows excluded from final assessment; 0 fits). Launch:
`python scripts/train_unified_full.py --fit`. Not launched pending the user's decision
on the formally failed AUC check.

**MAP COVARIANCE UPDATE: BOUNDED FULL-ROW TEST (30 September, user-authorized).**
Diagnosis from saved histories: the additive update (`c + w I` after each M
step) maximises no fixed objective. 24/43 historical QSO slices show likelihood
declines, up to 0.078 nats/object below their own best (z=1.45); the old stop
rule (`0 <= change < tol`) can never fire on a declining fit, so these run to
the cap. Edge slices and the background are monotone but slow: a separate,
harmless issue handled by tolerance 1e-4 with 300 as a ceiling.
Fix: `fit_projected(covariance_update='map')`, the conjugate-prior update of
Bovy, Hogg & Roweis 2011 (eqs. 19-20), V=(qS+nu w I)/(q+nu), nu=1, w=0.001 mag^2;
history is the mean log posterior; stop on |relative change| < tol. Additive
remains the default. Unit tests pass (379 total, commit `1d1bb99`).
Test: `scripts/test_map_update.py`, `configs/map_update_test.json`, run
`models/multisurvey_psf/work/map_update_test/20260930/c108415a6a7a695f`:
both updates on all rows of qso_13/15/16/22/31/39 (16 is a converged control),
identical warm starts, tol 1e-4, <=120 updates, 8 processes, 2.5 h wall budget.
Acceptance: zero MAP objective declines; MAP reaches tolerance; mean development
density MAP-additive >= -0.02 nats per slice; ranking AUC loss <= 0.002.
4,070 new role-3 QSO rows are added to final-assessment exclusions. This
supersedes the stopped full-row comparison as the covariance decision. If it
passes: set `covariance_update: map`, tol 1e-4 in the full config, re-prepare,
then launch the 44-fit refit. If it fails, report; no further tuning.

**FULL-ROW COVARIANCE TRIAL STOPPED BY USER (30 September, 17:04 UTC).**
The user stopped the trial because its full-background cost contradicted the
requested light diagnostic, and requested a rethink of scope and purpose.
Coordinator 52878 and all nine child processes were terminated and verified
absent. **No automatic restart or replacement fitting is authorized.**
Last complete checkpoints: both z1.45 methods at15 updates, both z4.05 methods
at20, background additive at5; background flooring never started. No matched
three-case comparison completed. Partial E steps are discarded. Existing
results may be inspected without fitting, but do not present these unequal
background runs as a comparison or as evidence of convergence.
`execution.json`, `stopped.json` and `PROGRESS.md` record the stop.
The original launch was at16:51:36 UTC. The active pointer is verified unchanged.
Read `docs/UNIFIED_COVARIANCE_TRIAL_2026-09-30.md` for the trial policy.
The computer has 48 GiB RAM, 14 cores (10 performance / 4 efficiency), and
1.2 TiB available disk; the immediate constraint was CPU contention, not disk.
At the original pause, other jobs were `real_oc.py --workers 6` (parent 46447) and
`inject_recover_macho.py --workers 4` (parent 46927); these PIDs are historical
snapshots, not commands to terminate. At resumption, CPU was approximately
60% idle with five other numerical workers; no swap activity was observed in
the short sample. The eight-worker diagnostic was launched.

Prepared comparison: all 114,657 training QSOs at z~1.45, 2,152 at z~4.05,
and 1,975,894 background objects; identical canonical warm starts, two updates,
20-update blocks, maximum 60 updates per fit / four-hour fitting budget.
Root: `models/multisurvey_psf/work/unified_covariance_trial/20260930/8176eab7484bf5e1`.
Launcher: `scripts/trial_unified_covariance.py` (prepare only; `--fit` runs).
The saved development manifest permanently excludes 1,646 QSO and 2,045
background source rows from future independent final assessment. Roles remain
unchanged and calibration objects are untouched. Complete-score comparisons
hold pilot priors/catch-all/spatial weights fixed; these are diagnostics, not
a completed scientific refit. No method or active-pointer change.
An initial attempt failed during baseline scoring, before any fit: legitimate
negative-infinite ranks from zero target-redshift priors were rejected by the
diagnostic. Reporting now retains these objects at the bottom of the ranking,
counts zero-weight transitions, and has a regression test. The corrected run
was launched and subsequently stopped as recorded above. Full suite: **372 passed in 94.86 s**,
`/tmp/unified_covariance_trial_fullsuite_final.log`.

The earlier recommendation to retain the additive covariance update for
production is superseded; the covariance decision now rests on the MAP-update
test above, not on this stopped comparison. Tol 1e-4 stopped the miniature additive runs;
none of the miniature floor runs reached it by 120 updates. Do not forecast
the additive stopping-time savings for flooring.

**CONVERGENCE ASSESSMENT AND QUICK TEST COMPLETE (30 September).**
The user asks to address iteration limits, not hypothetical premature stopping.
Read `docs/CONVERGENCE_PRACTICAL_ASSESSMENT_2026-09-30.md` and its three JSON
reports. Historical full-data QSO fits: 18/43 hit 300 (11 improving at redshift
edges, seven declining at intermediate centres); background also capped.
Extra 100 iterations gave only +0.000255 mean held-out QSO log density and
+.0142 global-background density, despite some moving individual predictions.
The larger model remained substantially better than the old small model.

New unified diagnostic: five cases x two covariance updates, 2,048 training
rows each, up to 512 non-training >=5-sigma development objects, 120 iterations,
four workers; completed in 206.2 seconds. This is deliberately a small diagnostic,
not capped production training. With additive updates, tol 1e-4 would stop at
38/46/100/59/105 iterations; tol 1e-5 reaches the rule only for two cases by120.
Continuing from the 1e-4 stop to120 slightly worsens mean held-out density in
all five. Individual changes are not negligible everywhere; this does not
certify complete classification scores. Eigenvalue floors give monotone
training improvement but no strict convergence by120; predictive differences
against additive are -.0391 (z.55), -.4675 (z4.05), +.0913 (z1.45), +.1339
(z1.75), -.0169 (background). Do not globally promote the floor as a cure.

Recommendation for discussion before launch: a less stringent, validated
numerical tolerance and actual validation-based early stopping/checkpoint
selection, with a separate non-training stopping sample. That avoids waste;
it is not proof of the original strict mathematical convergence criterion.
Do not add patience for hypothetical premature stopping, extend all fits, or
start a broad component search. EM acceleration is documented as an option,
not tested or recommended as the first intervention here.
Production update/tolerance and prepared run identity are UNCHANGED. No full
fit, download or active-pointer change. All diagnostic workers have finished.
Saved run: models/multisurvey_psf/work/unified_convergence_probe/20260930/d820e0835ac64b92.
Full suite: **368 passed in 82.59 seconds**; `/tmp/unified_convergence_fullsuite.log`.

**FULL UNIFIED LAUNCHER PREPARED: 4 QSO + 4 STELLAR WORKERS (30 September).**
The user requested a pragmatic loose-end audit before launch, then specified
four QSO and four stellar workers. Read `docs/UNIFIED_FULL_PREFLIGHT_2026-09-30.md`.
Full-data preflight passed: all 39 input hashes match; 1,049,260 QSO and
1,975,894 background fit/select rows; all 43 QSO slices populated; all 41
native bands have some measurements. Rare SkyMapper u/v support remains
explicit; no acquisitions needed. Active pointer unchanged.
Use `scripts/train_unified_full.py` and `configs/unified_full_training.json`.
Default is preparation only; `--fit` explicitly starts the candidate fits.
**No production fit has been launched: 44 tasks prepared, zero fit results.**
Run: `models/multisurvey_psf/work/unified_full/20260930/5fd837e5cafee6c1`.
The launcher runs independent QSO slices on four processes while four other
processes cooperate on one projected stellar E step. One numerical thread per
process. Checkpoints restore model, full history and iteration after identity
verification; completed fits are reused. Maximum 300 iterations, tol 1e-5,
fixed existing K/regularization. Likelihood declines and iteration limits
remain reported. Full bundle completion accepts this explicit config/root.
Four-worker versus serial equivalence and interruption/worker-failure tests
pass, as does the 4+4 launcher integration test. Full suite: **366 passed
in 78.45 seconds**; `/tmp/unified_full_launcher_suite.log`.
Next: report readiness to the user before launching, then run the prepared
refit. Rejection tuning is a subsequent calibration task, not a fit blocker.

**UNIFIED PILOT PASSES MATCHED DENSITY/RANKING TEST (30 September).**
The user explicitly separates rejection tuning from deciding whether to unify.
Rejection thresholds can be tuned later; they must not block density retraining.
The matched native-coordinate control uses the same pilot rows, initial native
predictions, K and eight iterations. All 12 checks pass, with no rejection
rules in the comparison. Full-band ranking AUC control -> unified is
.9961 -> .9957 South and .9921 -> .9925 North. Mean held-out QSO log-density
changes are +.014 South / +.129 North; background +.027 / +.069 nats.
Legacy-optical comparisons also pass. Read
`docs/UNIFIED_DENSITY_CONTROL_2026-09-30.json` and the pilot report.
Conclusion: proceed to the planned full-data unified refit; tune rejection
separately on calibration roles afterward, before promotion. This is a bounded
warm-start pilot, not convergence or posterior-calibration certification.
No full-data unified fit is launched and the active model is unchanged.

**UNIFIED FULL-SURVEY PILOT COMPLETE; PILOT ITSELF IS NOT A RELEASE.**
The requested end-to-end pilot finished. Read
`docs/UNIFIED_PILOT_2026-09-30.md` and its numerical JSON/guard diagnostic.
No pilot worker remains to restart. All 43 QSO slices plus the contaminant
fit completed eight iterations in 86.6 seconds; 27,686 QSOs and 49,809
contaminants, no S/N training cut or row caps. All selected QSOs entered the
existing slice coverage. Spatial weights/counts and calibration-role catch-all
(28,673 objects) completed in 13.2 seconds; support/score audit in 36 seconds.
All 44 fits hit the deliberate pilot budget; no convergence claim.

Run: `models/multisurvey_psf/work/unified_pilot/20260930/b5c55adf3698419d/`.
Config: `configs/unified_pilot.json`; train cells22/63/169, test28/62/170 at
Galactic nside4, saved roles preserved. Canonical full-data warm starts make
this a limited-area update, not independent small-area training. Bundle has
36 latent coordinates / 41 input labels and the RA/Dec convenience interface,
with native catalogue provenance overrides, optional local refits and a
separate QSO-support percentile. NSC u/VR and VHS H have no pilot training
measurements; do not claim empirical validation of those coordinates.

Known northern high-score grid tails 72/79 become zero; southern grids also
have zero, already before the new support veto. Functional subset checks pass
where measured. Overall 13/21 declared checks pass: full-band/Legacy-optical
QSO retention/ranking fail. New support cut retains 98.2% of 614 calibration
mask-evaluations (192 distinct QSOs), but 60/64 test QSOs per hemisphere.
Both guards together leave only 56/64 QSOs eligible per hemisphere.

A saved-intensity diagnostic shows much of the full-band AUC loss is the old
fixed component-distance veto: raw ungated AUC south/north .9993/.9785 versus
control .9985/.9958, while the old gate gives pilot .9254/.8899. Four of the
five southern and four of six northern QSOs rejected by the old gate pass
the new calibrated support test. This does not endorse removing all guards.
The user's subsequent direction supersedes the proposed pretraining gate
investigation. The matched control above now establishes that unification
preserves ranking and predictive density. Defer rejection calibration until
after the full-data fit; do not reopen ultra-faint/capacity investigations.
All pilot and matched-control workers have exited.

Software suite 359 passed in 70.73s (`/tmp/unified_pilot_final_suite.log`),
plus all-41-single-band marginal assertions passed. Native WISE residual
variance 1e-6 mag^2 is explicit in the operator; affected abundance-prior grids
carry the coordinate Jacobian. Active pointer remains unchanged. No downloads
or full-production fitting. Logs `/tmp/unified_pilot_{training,completion,validation}.log`.

**LATEST USER DIRECTION: TRAIN ALL ELIGIBLE ROWS; VALIDATE MAINLY ABOVE 5 SIGMA.**
The user separates unification from extreme-outlier classification and rejects
an investigation of sources below the detection limits as a release blocker.
The previous proposed faint-end stellar refit is superseded. The requested
plan is `docs/UNIFIED_MODEL_RETRAINING_PLAN_2026-09-30.md`, incorporating the
user's `docs/xd_qso_contaminant_ood.tex`. The user endorsed steps 2--5 and
clarified step 1: keep all quality-eligible training rows, including below
5 sigma, without caps. Five sigma sets the main validation/reporting domain,
not a training cut or automatic scoring rejection. The pilot above is complete.
The intended user interface accepts candidate RA/Dec, named photometry with
errors, and primary-QSO redshift; derive l,b and the default Legacy hemisphere
internally, respecting actual photometric provenance in overlap regions. The
current low-level scorer takes l,b and survey-labelled bands; the convenience
RA/Dec wrapper is now implemented and tested in the pilot.

The existing stellar-named sample is already empirical PSF catalogue background
with known QSOs removed, not a pure-star selection. Reuse it as the contaminant
population. Keep same-z/other-z QSOs, contaminant background and catch-all;
add a calibrated QSO-support percentile as a separate ranking eligibility
criterion, not a replacement binary score or a probability multiplier.

A saved-prediction re-summary gives northern stellar loss 0.0279 nats on
35,992 objects with >=5-sigma reference-band detection, within the existing
0.1-nat tolerance. Requiring >=5 sigma in any grz band leaves a 0.1701-nat loss
under the old reference choice: most faint-reference objects are detected in
another band. Use actual flux/error for the validation domain, preserve weak/negative
other bands, and prefer a >=5-sigma reference when present. The full-survey pilot and matched comparison above subsequently evaluated
that reference rule successfully for density/ranking.
Numbers: `docs/POOLED_OPTICAL_DETECTION_DOMAIN_2026-09-30.json`.
The next implementation is the full shared observation operator, empirical
background/support integration, then one uncapped pooled campaign and focused
validation, as outlined in the plan. No downloads or ultra-faint refit campaign.

**POOLED OPTICAL PROTOTYPE COMPLETE (30 September); FEASIBLE, ONE FAILED CHECK.**
All fitting and validation have finished. No prototype worker needs restarting.
Read `docs/POOLED_OPTICAL_PROTOTYPE_2026-09-30.md` for the decision, comparison,
plots and exact saved locations; the accompanying JSON contains all metrics.
The experiment used all 1,049,258 eligible optical fit/select QSOs, all 43
redshift slices and all 1,975,894 fit/select stars, without sample caps.

The final pooled model preserves QSO ranking: AUC 0.99182 -> 0.99181 south and
0.98807 -> 0.98759 north. Mean held-out conditional log-density changes relative
to separate optical fits are -0.0016/-0.0306 nats for southern/northern QSOs
and +0.0038/-0.1706 for stars. The northern stellar loss fails the declared
0.1-nat tolerance. Most of that loss comes from reference asinh magnitude
24--30 (13,821 objects, 22.7% of northern test stars; mean loss 0.525 nats).
The fixed affine relation near the asinh softening regime is a candidate
explanation, not a demonstrated cause. No classification collapse was measured.

The original northern bright/intermediate low-density high-QSO grid counts
72/79 become 0/0. Both separate optical refits and pooled fits remove this
failure; pooling cannot claim sole credit. Four southern intermediate grid
points remain above 0.5 total-QSO score, versus three in the separate optical
control and zero in the original full-data candidate. These use the original
fixed low-density mask and are stress tests, not measured contamination.
All seven nonempty grz subsets pass numerical/guard checks. Paired absolute
score disagreement improves slightly, 0.0829 -> 0.0806. Overall 16/17 declared
checks pass; do not describe this as a fully validated release.

Original run: `models/multisurvey_psf/work/pooled_optical_prototype/20260930/aa79cd8f3486dee8/`.
Final variant: its `instrumental_scatter_6d29f313e9caa5d7/` subdirectory.
The original paired-QSO scatter caused a northern density loss of 0.790 nats.
One completed targeted refit uses the stellar instrumental residual covariance
with the fitted QSO offset; excess QSO variation belongs to the learned
population for this prototype. Separate fits and stellar fits were reused.
The final variant contains 132 fits: 103 converged within the common budget,
29 reached 40 iterations, with no recorded likelihood decreases. No automatic
convergence/refit campaign is authorized by these results. The QSO scatter
choice used held-out diagnostics, so the reused test sample is development
evidence rather than independent confirmation of that choice.

The earlier proposal to investigate and refit the faint stellar population
is superseded by the latest 5-sigma-domain plan above. The prototype metrics
remain the historical measurements; do not relabel the original failed check
as a pass or treat the entire faint bin as undetected. Production remains
`models/multisurvey_psf/630f47f63b6f0694` until the planned replacement is ready.

**COMPLETE-SCORE AUDIT FINISHED; ACTIVATION HELD (30 September).**
The full-data candidate improves real-object ranking but fails the northern
low-density colour-grid check. The active pointer remains
`models/multisurvey_psf/630f47f63b6f0694`. Do not promote the candidate before
repairing and retesting this specific support failure. Do not restart broad
convergence/capacity work, downloads or completed density workers.

Completed candidate: `models/multisurvey_psf/work/full_sample_completion/e894d3c77baab446/`.
Its shapes/spatial weights are the frozen assembled 300-iteration fit. All 41
prior pairs and the calib-only catch-all are complete: 1,975,894 fit/select
stellar count rows; 502,251 catch-all calibration rows. QSO abundance priors
remain numerically inherited. Six band areas are measured and 35 explicitly
approximate. Items 1–3 remain complete; do not redo them.

Item 4 real-data results: all-band AUC 0.9429 -> 0.9743; optical
0.9829 -> 0.9962; Legacy optical 0.9864 -> 0.9859. Other tested subsets remain
within the declared tolerance. Complete-score mathematical/guard-behaviour
checks pass. These are small balanced test-role samples and synthetic clean
blend fixtures, not full probability calibration or labelled close-pair tests.

Release blocker: northern Legacy colour grids have 72 high total-QSO
scores in common low-density bright points after the existing guard, versus 0
for the old model; the intermediate-magnitude comparison is 12 vs 79.
An explicit unsupported colour point receives 0.995 total-QSO score. Its nearest
bright measured northern training QSO is 3.914 mag away in the colour plane.
The affected low-z mixture components have large global weight but only
1.1--2.7 effective northern members and essentially no bright northern members.
Their extrapolated Gaussian shapes nevertheless certify support through the
current nearest-component distance guard. Prior/catch-all substitution alone
and retaining magnitude in a joint-distance check do not remove this failure.

Further survey/redshift tracing (30 September): the failing example's QSO
redshift profile peaks near z=0.85, with about 53% below z=1, 25% at z=1--2
and 19% at z=2--3. The earlier three low-z components are examples, not an
exhaustive localization. At z=2.25 a contributing component has about 14,777
southern but only 12 northern effective members; SDSS and PS1 also strongly
support that component. At z=0.85 the dominant component has essentially no
bright members in any optical survey, adding magnitude extrapolation to sparse
northern support. The same artificial luptitudes scored in the southern system
have total-QSO score 0.020 versus 0.995 in the north: different QSO and stellar
densities produce different competition. Neither fact proves all other band
subsets free of extrapolation. Numerical profiles and per-survey component
responsibilities: `docs/NORTHERN_TAIL_EXPLANATION_2026-09-30.json`.

Practical impact remains unmeasured: these colour-grid counts are artificial
stress tests, not catalogue contamination rates. The user requested visual
evidence before deciding on a repair. Four figures now show the saved old/new,
north/south score maps, actual bright training colours, published reference
passbands and measured QSO colour tracks. See
`docs/NORTHERN_REVIEW_PLOTS_2026-09-30.md` and `plots/northern_review/`.
At 20 <= r < 21, the median absolute north/south difference between binned
median QSO colours is 0.019 mag in g-r and 0.038 mag in r-z. These compare
different populations, not same-object filter corrections. They contrast with
the several-magnitude displacement of the explicit synthetic failing example.

**North-South unification experiments completed (30 September).** The user
requested tests of photometric unification and then required discussion of
the shared-model design before further implementation or training. The
DESI correction and a locally fitted cubic were compared on the same objects
in cached overlap fields. On reserved bright samples (3,788 stellar-background
objects and 116 QSOs), DESI leaves median g/r/z residuals of
0.004/-0.006/-0.005 mag for stars and 0.005/0.011/0.033 mag for QSOs.
QSO robust scatter remains 0.19/0.13/0.12 mag; only three independent bright
evaluation fields contribute. The cubic has no clear overall QSO advantage.

A separate diagnostic ties northern optical coordinates to existing southern
shapes through the DESI relation and measured total residual scatter. The
72 and 79 northern high-QSO low-density grid points become zero; the explicit
probe changes from 0.995 to 0.007 total-QSO score. Legacy-optical AUC on the
saved small real-object sample changes from 0.9859 to 0.9878. This is not a
pooled fit: priors, catch-all and spatial weights are numerically inherited,
and Legacy W1/W2 are not tied. No production model was changed or promoted.
Reports, figures, limitations and reproducible locations are in
`docs/LEGACY_UNIFICATION_2026-09-30.md`.

Before prototype authorization, the next step was to discuss the relation, including faint/negative
flux, missing bands, residual uncertainty and WISE measurement conventions.
DESI is the recommended starting relation, not a finalized production design.
If adopted, pooled fitting requires the joint QSO slices and stellar model to
be refitted, spatial component weights re-estimated, the catch-all rebuilt and
affected priors reviewed. Reuse the existing data and frozen roles. Do not
launch full production fitting; the separately authorized prototype is described above.
Component-level training support remains an additional proposed guard, not an
implemented repair. Real-catalogue incidence and full calibration remain open.

The user's external latent-forward-model proposal has been reviewed against
the code and primary sources. Recommendations and implementation gaps are
recorded in the unification report's design-review section: distinguish 41
input labels from 38 optical-only or 36 fully shared latent coordinates,
calibrate an errors-in-both-systems affine relation before density fitting,
start with simple QSO residuals rather than 43 independent offsets, and preserve
native reference-band conditioning and priors. These were discussion recommendations at that stage. General projected XD
and pooled optical training have since been implemented and tested as recorded
above; full-band integration remains to be done.

Main evidence: `docs/FULL_SAMPLE_RELEASE_2026-09-30.md` and JSON.
Reproduce with `scripts/validate_full_sample_release.py` and
`configs/full_sample_release.json`. Focused support evidence and scripts are
linked in that report. The candidate, the prior active bundle, individual
continuations and covariance-floor experiments are all preserved. No downloads,
production fits, prior changes or promotion occurred during the audit.

Updated 30 September 2026. This status supersedes older operational statements.
The test data have been inspected; do not call them untouched. JOURNAL.md
retains the history. Full probability calibration remains unfinished.

## Objective and authorization

Prepare uncapped QSO and stellar training data for the active PSF model, which
supports any nonempty subset of 41 bands, spatial stellar variation and local
refitting. Preserve held-out data. The readiness report was delivered and the
user explicitly authorized retraining on 29 September.
The current model is unchanged; full probability calibration remains unfinished.

**AllWISE and Pan-STARRS are complete; acquisition has stopped as requested.**
Both have saved results for all 1,657,444 targets, including nonmatches. The
worker exited after these two surveys. No acquisition or independent audit
worker is running. NSC, SkyMapper and VHS acquisition remains paused.
Retain and use existing measurements from those surveys, with unavailable
measurements masked; the model's 41-band interface is unchanged. Do not turn
their incomplete acquisition into a requirement to resume downloads. The
independent association auditor remains stopped. The scheduled uncapped density fits are complete; see the run record below.

## Completed data

| Product | Verified state | Location |
|---|---|---|
| Combined SDSS+DESI master | 2,051,328 deduplicated objects; original memberships retained | `~/data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/` |
| Eligible QSO target list | 1,657,444 objects in frozen order | QROOT `targets.npz` |
| QSO SDSS photometry | Results for every eligible target saved, including nonmatches | QROOT; `progress_sdss.json` and the existing batch reader locate old and new caches |
| QSO Legacy photometry | Complete; all 34 batch hashes and identities rechecked; correct hemisphere throughout | QROOT `legacy_local/4d418c295df8aa17/photometry.npz` |
| QSO AllWISE and Pan-STARRS photometry | Complete; all targets covered, row identities and band counts checked in 34 saved batches per survey | QROOT `acquired/allwise_*.npz`, `acquired/ps1_*.npz`; completion manifest linked below |
| Stellar sample | 2,978,857 cleaned entries; seven-survey photometry for all 274 regions | SROOT `photometry/`, `cleaning_report.json` |
| Stellar Legacy selection areas | All 274 regions measured; external-survey effective areas remain a separate issue | SROOT `areas/`, `coverage_report.json` |
| Spatial roles | Fit, selection, calibration and test roles saved, including northern test regions | SROOT `spatial_roles.json` |

QROOT: `models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/`.
SROOT: `models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0/`.
The full location index is [LOCAL_DATA_LOCATIONS.md](docs/LOCAL_DATA_LOCATIONS.md).

## Partially acquired data

NSC, SkyMapper and VHS remain paused and incomplete by the user's decision.
The original inventory established reusable results for about 68,400 targets
per survey, including nonmatches. Further queries and corrected PS1 ID matches
are saved under QROOT `gap_acquisition/`.

Each of these three paused surveys has a 50,000-target assembled prefix, plus
reusable results elsewhere in the target list. This prefix is not total local
coverage or the number of detections. Their full acquisition is not a gate for
the current data effort.

## Association checks: disposition

The stopped Legacy count query includes all catalogue releases, without the
required target hemisphere. It cannot distinguish overlapping survey entries
from alternative counterparts. Its `multiple_matches` flags must not become
training exclusion masks. Preserve the results as historical diagnostics;
do not restart this bulk audit or replace it with another blanket query.

Local checks have already verified Legacy source provenance, target identities,
separations, hemispheres and file hashes. A further local check found two target
rows sharing an exact Legacy release and counterpart position; their identities
are saved in QROOT `legacy_local/4d418c295df8aa17/shared_counterpart_review.npz`
for focused review. No selection masks were changed. This local check does not
certify the absence of every possible nearby alternative counterpart.

Use saved candidate counts from new positional acquisitions and existing
association results where applicable. Keep unknown counts explicit. Before any
additional query, establish which missing fact affects the training selection,
why local evidence cannot supply it, and the smallest necessary target set.

## QSO fitting completed (29 September, 15:28 UTC)

All 43 QSO final slice files are saved: 25 meet the configured convergence
criterion; 18 reached the 300-iteration limit and require convergence review.
The QSO worker pool has exited. The subsequent four-worker stellar fit has also finished.
Verified per-slice row counts and frozen roles, 41-dimensional finite mixtures,
positive-definite covariances and unchanged active pointer. Evidence is in
`models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d/qso_completion.json`.
At QSO completion, density-EM compute was about 53.0% overall and 22.3% stellar.
The final run is now 100% for both populations.
This completes the scheduled QSO fits, not scientific validation or release.
See the completed stellar run details below.

## Training run and remaining work

The requested local assembly and uncapped density-training integration are
complete. Inputs are in
`models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8/`, with
checksummed memory-mapped arrays and an assembly report. The final shape fits
used all 1,049,260 QSO fit+selection rows and 1,975,894 stellar fit+selection
rows. Calibration and test roles remain separate. No new downloads were launched.

The current launch record, log and preflight remain in
`models/multisurvey_psf/work/full_training_runs/20260929T045901Z/`.
The parent `current.json` locates this run. `stellar_parallel_train.log` is
now the training log; the earlier serial and QSO-parallel logs and launch
records are retained as history. The original coordinator and its four idle
QSO workers exited at a checkpoint-safe handover.

Historical handover command: `scripts/train_stellar_parallel.py --workers 4 --task-rows 8192
--resume-from models/multisurvey_psf/work/full_training_fits/5d1429d22b40d24d`
was used to continue the saved fit. This run is complete; do not relaunch it
from this historical command. Both parent and destination run locks reject
duplicate fitting. The completed continuation directory is
`models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/`.
Its `checkpoint_lineage.json` records all transferred hashes. The changed
streaming-engine implementation has its own identity; the old directory is
preserved. The statistical model, frozen roles, selected K=20, regularization,
convergence tolerance and total iteration limit are unchanged.

Four workers evaluate disjoint batches, then their sufficient statistics are
combined before one global M step. All 1,975,894 stars enter every iteration.
A real-data benchmark measured 26.84 seconds serial versus 7.81 seconds with
four workers (3.43x), with only floating-point rounding differences. Tests
cover serial-checkpoint continuation, worker failure without partial updates,
and unchanged completed QSO fits. Details: `docs/STELLAR_PARALLEL_2026-09-29.md`.
The active model pointer remains unchanged.

Progress monitoring ran independently every 60 seconds through
`scripts/monitor_full_training.py --watch`. In the run directory,
`PROGRESS.md` is the readable live summary, `progress.json` is its structured
form, and `progress_history.jsonl` records successive checks. The monitor
reports checkpoint iterations, likelihoods, completed fits, CPU/memory,
the four stellar workers, within-pass row counts and worker exit; it stops when the density candidate completes or the worker exits.
Every progress report must include estimated overall compute completion,
plus QSO and stellar percentages. Weight saved EM iterations by fitted rows
and component count; budget unfinished fits at their iteration limits and
maximum K until selection is known. Clearly label this as density-EM work,
not wall-clock progress or completion of priors/catch-all/validation.
It does not deliver chat notifications. Check `monitor.lock` and `monitor.log`
for the monitor PID and errors before starting another monitor.

The unchanged scientific configuration is `configs/full_sample_training.json`.
`scripts/train_full_sample.py` remains the serial/preflight entry point, but
the current continuation uses `scripts/train_stellar_parallel.py` as above. Read
[the readiness report](docs/FULL_TRAINING_READINESS_2026-09-29.md) before launch.
The old capped recovery scripts are historical and are not this run's entry point.

1. Items 1–3 are complete. Item 4 exposed the northern component-support
   failure documented above; activation is held for that measured reason.
2. Complete the newly authorized pooled-grz prototype and its bounded checks.
   Full joint retraining is a later decision; historical commands remain inactive.
3. Validate any agreed candidate on reserved objects and the saved tail tests
   before promotion; retain the old bundle for rollback. Acquisition is complete.

## Working rules

- Automatically commit and push each coherent verified step without another
  user reminder. Update state and findings first, verify the remote commit,
  and report any failure explicitly. Check outstanding task changes before
  ending a work turn. Stage only relevant files; the full rule is in AGENTS.md.
- Read this state, the location index and recent journal entries before acting.
- Before expensive work, record the missing fact, its purpose, local evidence,
  proposed query and measurable completion criterion. An old checklist item is
  not sufficient justification.
- Update this file after substantive progress or a changed decision. Keep
  historical findings in the journal rather than appending conflicting status
  paragraphs to several documents.
- Report acquisition, local assembly and validation separately. Use saved
  output counts and measured timings; a running process alone is not progress.
- Acquisition is finished for the agreed scope. Do not restart downloads from
  an older checklist. The readiness gate was satisfied and the user authorized this training run.

Evidence: [local reconciliation](docs/PREPARATION_RECONCILIATION_2026-09-28.json).
Download completion: [AllWISE and Pan-STARRS verification](docs/ALLWISE_PS1_COMPLETION_2026-09-29.json).
