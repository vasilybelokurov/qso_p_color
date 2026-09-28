# Full-sample training preparation — 28 September 2026

Current operational status is maintained in [PROJECT_STATE.md](../PROJECT_STATE.md).
Acquisition is paused. The Legacy bulk count audit is stopped and its
cross-release counts must not be used as training ambiguity masks. Status
statements below describe earlier preparation stages.

Status: selection policy recorded; data preparation is incomplete; **retraining
has not started**. The active model and its release pointer are unchanged.
The user requested a report before any retraining is launched.

Preparation update: the unrestricted DESI query and all 274 stellar source
queries have completed. DESI returned 1,645,842 unique identifiers, including
all previous identifiers and 4,599 outside the old redshift interval. The new
combined master is `77d7aa514e9e3e47`, with 2,051,328 positional objects and
all 2,396,256 original memberships preserved. It is saved under the same
`~/data/qso_p_color/catalogues/qso_sdss_desi/` root and selected by `current.json`.
Reproduce it with `configs/qso_master_full.json`; the initial version below
remains available. The source query still requires primary, zwarn=0 DESI QSOs
with a matching DR1 photometry-table entry, but has no redshift restriction.

The stellar queries returned 3,053,059 raw PSF, mask-clean entries without a
flux floor. Hemisphere checks, catalogue deduplication, known-QSO removal,
seven-survey photometry and the final role/area audit remain outstanding.
The inherited cone roles have no northern final-test region and will not be
adopted unchanged. The QSO seven-survey acquisition has now been launched via
`scripts/fetch_full_qso_photometry.py` and `configs/full_sample_photometry.json`.
Every eligible target is visited; its 50,000-row batches are assembly
checkpoints. SDSS now has a separate configurable sky-block query size;
the other surveys retain the earlier query schedule. Raw nearest matches still require association
and hemisphere checks before they become training measurements.

The next preparation stage is now running through
`scripts/prepare_stellar_sample.py --config configs/stellar_preparation.json`.
Cache `2d43d81d0b5f2af0` contains the completed cleaning and spatial-role
manifests and the accumulating photometry and area measurements. Cleaning
retained 2,978,857 entries: 67,609 failed the hemisphere/latitude rule, 74 were
duplicate catalogue keys, and 6,519 matched known quasars. There are no duplicate
catalogue keys across retained regions. Negative fluxes and single-band objects
are retained; no flux, magnitude or random-count cut was added.

The spatial geometry is frozen before fitting: 9 northern and 40 southern test
cones; 34/112 northern/southern fitting cones; 8/26 selection cones; and 10/35
calibration cones. The added northern nside=4 test cells are 5, 12, 24, 28 and
94. All historical test cells and cones remain reserved. Among 12,121 historical
held-out QSO positions matching current eligible targets, none was reassigned
outside the test role. Current QSO role counts are 854,202 fit, 195,058 select,
239,371 calibration and 368,813 test. After model selection, the final shape fit
uses fit plus selection roles, as specified below. Coverage certification remains
pending; a fixed geometry does not establish that every band is supported.

The stellar pipeline now performs seven-survey matching and area measurement
concurrently. Survey measurements with multiple eligible associations or a
catalogue source shared between distinct targets are masked in the prepared
photometry, with raw values and ambiguity flags retained. Legacy measurements
come directly from the selected catalogue entries. Areas use actual mask and
exposure images and the union of g, r and z coverage, rather than requiring r.
These are the Legacy population-selection areas; external-survey footprints
and band-specific detection completeness require separate treatment. The final
coverage report explicitly retains that limitation.

`scripts/audit_qso_associations.py --watch --cache
models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff` audits QSO
batches as they arrive. It checks association counts, separations, hemisphere
consistency and catalogue sources shared across the full target set. Its flags
must enter final photometry assembly; the raw acquisition caches are preserved.
Both jobs checkpoint their results and produce reports without launching fits.

## 1. What changes

Use all eligible quasars from the combined SDSS DR16Q and DESI DR1 catalogues.
Remove the 2,000-object cap per redshift bin. For the stellar background, use
every eligible source in the selected training regions; remove the 20,000-row
shape-fit cap. Control memory with batches that accumulate the likelihood and
sufficient statistics over every training row. A batch size must never become
a sample-size limit.

These changes preserve the PSF population, one joint 41-band distribution,
exact marginalisation over missing bands, continuous magnitude conditioning,
Galactic HEALPix variation, optional local refitting, and the broad catch-all
with its support guard. They do not establish probability calibration.

## 2. The initial QSO parent catalogue (retained for provenance)

`scripts/build_qso_master.py --config configs/qso_master.json` builds a versioned
master under `~/data/qso_p_color/catalogues/qso_sdss_desi/`. The initial version
is `b0401c528c65a6e5`; `current.json` identifies it. This is a parent catalogue,
not a training subset.

| Input or result | Rows |
|---|---:|
| Local DESI DR1 snapshot | 1,641,243 |
| Local SDSS DR16Q snapshot | 750,414 |
| All input memberships | 2,391,657 |
| Distinct positional objects | 2,047,156 |
| DESI only | 1,296,744 |
| SDSS only | 415,586 |
| Present in both catalogues | 334,826 |

The DESI snapshot already requires `spectype='QSO'`, `zwarn=0`,
`zcat_primary`, and **0.1 < z < 4.4**, where z is spectroscopic redshift.
It is not an all-redshift DESI catalogue. The SDSS input contains all 750,414
DR16Q entries, including their original quality flags and invalid-redshift
sentinels. The master preserves every row from both snapshots. Before calling
the parent complete over all redshifts, fetch the DESI identity/redshift list
without its inherited redshift restriction and publish a new version.

`objects.npz` has one row per positional group, coordinates, an adopted
redshift and source identifier, membership counts, and diagnostic flags.
`members.npz` contains every original identifier and redshift, with
`object_index` linking to the object row and `input_row` linking back to the
hashed input catalogue. DESI `targetid` and SDSS `sdss_name` remain the original
identifiers. Photometry need not be copied into this master.

`manifest.json` records input paths and SHA-256 hashes, source SQL/provenance,
the grouping and representative policies, output hashes, units, and counts.
The builder preserves completed versions rather than overwriting them.

### Positional grouping and redshifts

Group sources connected by separations of at most 1 arcsec, including repeats
within DESI. Redshift agreement is not a condition for grouping. Preserve all
members even when their redshifts disagree. Adopt a finite positive redshift,
prefer a row with clean spectroscopic flags, then DESI over SDSS, then the
original identifier in deterministic order. The representative is bookkeeping;
all alternative measurements remain accessible.

There are 11 transitive groups whose members extend more than 1 arcsec from
the representative. Flag and inspect these associations before fitting.
The disagreement diagnostic is the redshift range divided by one plus the
smallest positive redshift in the group. Its 0.001 flag catches 114,240 groups;
only 3,225 exceed 0.01, and 1,791 exceed 0.05. **The 0.001 flag is not a training
exclusion:** small differences between spectroscopic estimates must not remove
a large fraction of the overlap sample. Audit the large disagreements and
record any changes to adopted redshifts in a separate, versioned decision table
before launch. Never edit the parent measurements to hide disagreement.

Independent readback checked every original identifier, input-row index,
coordinate and redshift, and verified group sizes. Results are in
`QSO_MASTER_AUDIT_2026-09-28.json`.

## 3. QSO training selection

Use the catalogue's adopted, finite positive spectroscopic redshift. SDSS
catalogue membership is `IS_QSO_FINAL > 0`; preserve its confidence flags for
validation. Do not impose an extra blanket SDSS `ZWARNING == 0` cut: the final
catalogue incorporates visual inspection and corrected redshifts. This follows
the [SDSS DR16Q catalogue description](https://www.sdss4.org/dr16/algorithms/qso_catalog/).
The stricter `spectroscopic_quality_clean` field in the master is a diagnostic,
not an eligibility flag. Inspect uncertain labels and report their sensitivity.

For a comparison within the current model's scope, retain 0.1 < z < 4.4 and
absolute Galactic latitude |b| >= 25 degrees. These are inherited scope limits,
not random sampling. Keep objects outside them in the parent and report their
counts. Widening the fitted redshift or sky domain requires extending the
stellar/prior coverage and validation as well; do not silently discard the
parent rows or claim coverage outside the fitted domain.

Require Legacy DR9 PSF morphology and `maskbits == 0`, with releases 9010 and
9012 recognised as south and 9011 as north. Morphology is selection metadata;
it does not require the user to supply a Legacy flux. Carry the photometric
system labels through every survey adapter.

Crossmatch all eligible objects to the seven surveys: SDSS, Legacy Surveys,
AllWISE, Pan-STARRS1, NSC, SkyMapper and VHS. Retain the native flux and error
for each of the 41 survey-labelled bands. Apply the survey's measurement masks
per band. Keep negative and low-S/N fluxes. Require at least one usable band;
do not require an r detection, complete survey coverage, a minimum of two
bands, or a particular reference band. Record ambiguous associations rather
than silently substituting a nearby object.

Using the adopted source's existing Legacy metadata, the preliminary cutflow is:

| Cumulative selection | Objects |
|---|---:|
| Valid catalogue redshift | 2,047,148 |
| Current redshift interval | 2,045,818 |
| Current Galactic-latitude domain | 1,915,155 |
| Legacy DR9 PSF | 1,746,425 |
| Mask clean | 1,657,469 |

Of the last group, 292,087 lie in the original reserved sky blocks and
1,365,382 elsewhere. These are **not final training counts**: full photometry,
association checks and the additional spatial partitions remain to be applied.
Both Legacy hemispheres are represented (427,326 northern objects).

## 4. Stellar-background training selection

The stellar component is the PSF field population after removing known QSOs.
It is star-dominated, not a catalogue of spectroscopically certified stars.
Unrecognised quasars and compact contaminants remain a modelling limitation.
Do not train only on spectroscopic stars, which would impose their targeting
selection on the background.

Choose sky regions across Galactic longitude and latitude, both Legacy
hemispheres, imaging depth and extinction, and all seven survey footprints and
their overlaps. Use the existing 274 strictly contained, 0.3-degree-radius
cones as a starting design, after auditing their footprint coverage. Add regions
where a band or overlap lacks independent training and validation coverage.
The 274 cones are a starting point, not a completeness claim or a fixed ceiling.

Within each selected region, retrieve **all** Legacy PSF, mask-clean sources.
Remove known SDSS/DESI quasars positionally within 1 arcsec using the complete
parent membership list, including objects outside the QSO fitting redshift
interval. Crossmatch to all seven surveys and apply the same per-band masks
and at-least-one-band rule as for QSOs. No flux floor, S/N threshold,
mandatory r band, magnitude-bin quota or random row cap is added.

The old wide-field caches cannot be reused as complete inputs: their query
requires `flux_r > 0.1` nanomaggies. The old fitter also uses an r-magnitude
range. Reuse suitable region centres and verified mask assets, but re-query
the source lists without these restrictions. Deduplicate regions if they
overlap and preserve the area of their union.

Use all fit-region sources for the shared Gaussian shapes and spatial mixture
weights. Estimate surface densities from all qualifying counts divided by
measured usable area. Record footprint/mask area separately from detection
completeness; positive flux is not a geometric coverage criterion. Each band
needs a consistent area and source-selection definition for its reference-band
prior. Retain magnitude dependence by conditioning the joint model, not by
replacing it with four independently fitted magnitude bins.

Start spatial fits at the current Galactic NESTED HEALPix nside=4 with parent
nside=2. These resolutions remain configurable. Select pooling strength using
independent subregions within populated parent cells, so validation actually
tests pooling rather than forcing every point to the global fallback. Report
coverage and local/parent/global fallback rates in both hemispheres.

## 5. Spatial partitions and fitting

Freeze an object/region manifest before fitting. Preserve the historical final
QSO holdout blocks and held-out stellar cones. Add northern final-test regions
and coverage for sparse surveys; a southern holdout alone is insufficient.
All aliases of a positional object receive the same role. Any exclusion region
applies consistently to both populations, field counts and prior construction.

Keep four roles explicit: fitting, model selection, catch-all calibration, and
final testing. Selection regions choose mixture capacity, covariance floors,
initialisation and spatial pooling. After those choices, fit the selected
model on all fit plus selection rows. Keep catch-all calibration regions out
of the quasar/stellar shapes, spatial weights and count priors; use them to fit
catch-all parameters and shares. Final-test regions enter none of those steps.
Persist seed, coordinate frame, HEALPix ordering, block IDs and region geometry;
never reconstruct the split later from a changed sample.

Numerical fractions and the additional region IDs must be frozen after the
coverage audit, before any likelihood is inspected. They are not yet a completed
partition manifest. Ensure selection subregions can test within-cell spatial
predictions as well as transfer to unseen cells.

Fit using every eligible row in its assigned role. Choose quasar capacity per
redshift slice and stellar capacity on selection data; the previous winning
K=12 at the top of a short grid does not settle stellar capacity. Record the
number of rows and observed bands per slice/component-selection run. Check
convergence and score stability explicitly. Preserve the current model until a
separately versioned candidate passes the declared checks.

## 6. What must be ready before launch

SDSS acquisition now uses the DR16Q photometric `objid` wherever an eligible
master object has an unambiguous SDSS membership link, including DESI-preferred
objects. The identifier is joined to indexed DR14 `photoobjall.objid`.
Completed whole-list results are reused; missing results use sequential sky
blocks with independent checkpoints. This retrieves the existing DR14
photometric system and quality fields; embedded DR16Q fluxes are not substituted.
Nonprimary, unresolved, conflicting or out-of-radius links use positional
fallback. Completed positional batches and exact association-count caches are
retained. The independent multiple/shared-association masks remain mandatory.
Resume with `fetch_full_qso_photometry.py --config configs/full_sample_photometry.json
--resume-cache models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff --surveys sdss`.
The shared batch reader also serves the auditor so it never rematches a batch
whose switched acquisition is already saved.
Outstanding objects without usable SDSS IDs are downloaded
with `scripts/fetch_sdss_without_ids.py --cache <cache>`, using sequential
indexed positional sky blocks for new runs. The original unsaved whole-list positional download was stopped at the user's
request at 17:58 CEST and replaced by the block run. Completed results are reused. Each route
keeps its own results and target identities; the main fetcher combines them
without requerying the positional results, including recorded nonmatches.
The local 50,000-row assembly files are separate from the configurable SDSS
database block size. Failed ID links are handled afterwards. Block limits
never cap the sample and never restrict the survey side of a match to the
target block. See `SDSS_SKY_BLOCKS_2026-09-28.md` for the timing assessment.

SDSS acquisition is complete for all 1,657,444 targets. Legacy photometry has
also been assembled and verified for every target, entirely from existing local
catalogues: 1,312,233 DESI-preferred objects matched by exact integer target ID
and 345,211 SDSS-preferred objects taken from their recorded source rows.
Both hemispheres retain their own band labels. Eight WISE observation-count
differences affect only zero-inverse-variance measurements and leave final masks
unchanged. The saved product, hashes and checks are described in
`LEGACY_LOCAL_ASSEMBLY_2026-09-28.md`. Independent association checks remain
pending. Before acquiring any other survey, inventory its existing local
photometry and retrieve only demonstrated gaps. The fetcher now requires an
explicit survey selection; Legacy uses only local assembly.

1. Verified parent master and explicit source/redshift scope; reviewed ambiguous
   associations and large redshift disagreements.
2. Uncapped, aligned 41-band QSO photometry and fresh stellar source lists.
3. A frozen sky-region design, all four role manifests and usable-area records,
   with northern and sparse-survey coverage demonstrated.
4. A trainer configured to use all eligible rows, with batch accounting that
   proves no row cap remains and disjoint catch-all calibration is enforced.
5. A preflight report listing exact counts after every cut, counts per role,
   band and overlap, expected runtime/memory, and the immutable output path.

Only then report **ready to launch** to the user. The saved master and this
selection policy are preparation, not evidence that these remaining gates
have passed. No retraining is authorised by an elapsed wait or by completing
this document.


## Five-survey acquisition update

The local inventory is complete and recorded in `LOCAL_DATA_LOCATIONS.md`.
`fetch_qso_photometry_gaps.py` now acquires the remaining data without another
approval pause. It reuses every verified measurement and nonmatch, retrieves
PS1 measurements through corrected original-FITS IDs where available, and uses
plan-checked Q3C queries for the rest. Results are checkpointed and reassembled
in the frozen target order. `QSO_GAP_ACQUISITION_2026-09-28.md` records the SQL
choices, live checks, ID-rounding correction and restart command. Association
validation and the readiness report remain prerequisites for retraining.
