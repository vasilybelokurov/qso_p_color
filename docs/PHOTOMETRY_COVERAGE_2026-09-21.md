WSDB photometric-survey exploration, 21 September 2026
===================================================

This records the exploration before the seven-survey extension was requested.
Its suggested next step below is historical. The implemented extension covers
all seven requested surveys; see the [current model choices](../README.md#choose-a-saved-model)
and [validation report](MULTISURVEY_VALIDATION.md).

**SDSS and Pan-STARRS are the first surveys to consider for quasars missing
Legacy Surveys DR9 photometry.** In a random sample of 2,000 such quasars,
SDSS supplies at least three optical bands for 91.0%, Pan-STARRS for 90.2%,
and their union for 93.25%. The difference between their individual coverage
estimates is too small to choose between them on coverage alone.

The simplest first extension would use SDSS, because the spectroscopic
catalogue already contains native SDSS fluxes and errors. Before choosing a
new survey model, a small comparison of nearby-source recovery in SDSS and
Pan-STARRS would establish which better serves the actual companion search.

**Scope and meaning of “missing”.** This exploration uses the project's
DESI DR1 and SDSS DR16Q parent selections at 0.1 < z < 4.4: DESI
`spectype='QSO'`, `zwarn=0`, `zcat_primary`; SDSS `zwarning=0`.
The existing caches contain 1,641,243 DESI rows and 637,122 SDSS rows.
These are catalogue row counts, with overlap between surveys. A live count
of the DESI spectroscopic selection also returned 1,641,243 rows.

The WSDB table `decals_dr9.main` contains both southern DECam and northern
BASS/MzLS photometry. Thus “missing DR9” here means no counterpart in that
combined catalogue within 1 arcsec; it does not mean outside the southern
photometric system supported by the current model. The northern and southern
systems remain distinct. See the [Legacy Surveys DR9 description](https://c.legacysurvey.org/dr9/description/).

The existing SDSS-to-LS match retained 604,235 rows after its photometric
cuts, leaving 32,887 to check. A further 651 DESI rows had an invalid LS
release or no positive optical inverse variance. Removing seven positional
duplicates within 1 arcsec left **33,531 positions** for a fresh DR9 check
without photometric quality cuts:

| Result of the fresh DR9 check | Distinct positions |
|---|---:|
| Counterpart within 1 arcsec | 28,626 |
| No counterpart within 1 arcsec | **4,905** |
| No counterpart even within 3 arcsec | 4,787 |

Most failures of the earlier clean match therefore still have a DR9 catalogue
entry. Changing surveys and recovering objects rejected by quality cuts are
different tasks. The 118 counterparts at 1–3 arcsec also show why an unmatched
position should not automatically be described as outside the footprint.

Of the 4,905 unmatched positions, 4,283 come from the SDSS input and 622 from
DESI. Their distribution is strongly northern: 4,506 (91.9%) have Dec > +30°,
and 3,198 (65.2%) lie between +34° and +36°. This helps explain the small
coverage from southern surveys.

**Measured optical availability.** The following results use the same
uniform random sample of 2,000 of the 4,905 positions, without replacement,
with seed 20260921. Each query selects the nearest source within 1 arcsec.
SDSS imaging requires `mode=1`; Pan-STARRS stacks require
`primarydetection=1`. All listed tables and columns were verified in WSDB.

“Three bands” means at least three finite reported measurements with positive
errors or inverse variances, after excluding magnitude sentinels. Negative
fluxes are retained. These counts measure data availability: survey quality,
signal-to-noise, and close-pair deblending cuts have not been applied.
Percentages are sample estimates, not full-catalogue counts.

| Survey | WSDB table | Bands checked | Counterparts / 2,000 | At least 3 bands / 2,000 | At least 3 bands |
|---|---|---|---:|---:|---:|
| SDSS imaging | `sdssdr14.photoobjall` | ugriz | 1,820 | 1,820 | **91.0%** |
| Pan-STARRS DR1 stacks | `panstarrs_dr1.stackobjectthin` | grizy | 1,804 | 1,804 | **90.2%** |
| Gaia DR3 | `gaia_dr3.gaia_source` | G, BP, RP | 1,207 | 1,203 | 60.2% |
| Legacy Surveys DR11 | `decals_dr11.main` | griz | 110 | 109 | 5.5% |
| DELVE DR3 | `delve_dr3.main` | griz | 76 | 76 | 3.8% |
| NSC DR2 | `nsc_dr2.object` | ugrizy | 121 | 55 | 2.8% |
| SkyMapper DR4 | `skymapper_dr4.main` | uvgriz | 40 | 39 | 2.0% |
| HSC DR3 wide | `hsc_dr3.wide_summary_small` | grizy | 32 | 32 | 1.6% |
| KiDS DR4 | `kids_dr4.main` | ugri | 4 | 3 | 0.15% |
| DES DR2 | `des_dr2.main` | grizy | 2 | 1 | 0.05% |

All five SDSS bands are present for all 1,820 matches. Pan-STARRS has all five
for 1,793 (89.65%). DR11 has all four checked bands for 70 (3.5%); switching
to DR11 alone would therefore recover only a small part of this missing
sample. Gaia offers three broad bands and is a possible supplement, but
these counts do not establish its performance for the colour-redshift task.

The SDSS/Pan-STARRS overlap is 1,759 objects, with 61 available only in SDSS
and 45 only in Pan-STARRS. Their union is **1,865/2,000 (93.25%)**.
Combining all the other optical imaging surveys in the table raises this to
**1,900/2,000 (95.0%)**, just 35 extra objects. HSC contributes 28 beyond
SDSS/Pan-STARRS, DELVE 17, and DR11 14; these additions overlap.

The aggregate is dominated by SDSS-selected quasars. Among the 250 DESI-origin
objects in the sample, SDSS covers 73, Pan-STARRS 116, and their union 118.
Among the 1,750 SDSS-origin objects, the corresponding counts are 1,747,
1,688, and 1,747. Pan-STARRS is therefore particularly useful for the smaller
DESI-origin remainder.

**Native SDSS photometry is already available for most of the full missing
sample.** Querying `sdssdr16qso.main` for all 4,905 positions found 4,278
(87.2%) with all five native ugriz fluxes and positive inverse variances.
Of those, 4,277 also have W1 and W2 fluxes with positive inverse variances.
These are full-sample counts. The native quasar catalogue can supply training
photometry; the general imaging table is needed for companions and background
objects. The [SDSS DR16Q documentation](https://www.sdss4.org/dr17/algorithms/qso_catalog/)
describes its attached multi-survey measurements.

**Infrared supplements.** These use the same 2,000 targets. WISE matches use
2 arcsec; UKIDSS and VHS use 1 arcsec. The columns report all the listed bands
with finite measurements and positive errors, under the same availability
definition as above.

| Survey | WSDB table | Bands checked | Counterparts | All checked bands | Fraction of 2,000 |
|---|---|---|---:|---:|---:|
| unWISE | `unwise_1901.main` | W1, W2 | 1,724 | 1,609 | **80.5%** |
| AllWISE | `allwise.main` | W1, W2 | 1,353 | 1,353 | 67.7% |
| VHS DR5 | `vhs_dr5.main` | J, H, Ks | 40 | 27 | 1.35% |
| UKIDSS LAS | `ukidssdr10.lassource` | Y, J, H, K | 11 | 5 | 0.25% |

unWISE supplies both bands for 1,600 of the 1,865 objects covered by SDSS or
Pan-STARRS. It is the first general infrared supplement to investigate; the
[unWISE catalogue paper](https://arxiv.org/abs/1901.03337) describes its W1/W2
measurements. The forced WISE measurements already attached to DR16Q have
different selection and measurement procedures from a standalone unWISE
positional match. Their higher availability must not be interpreted as an
equivalent, interchangeable catalogue. Close companions also require checking
whether the infrared measurement separates the two objects.

**Suggested course.** Start by comparing SDSS and Pan-STARRS around a small,
representative set of these missing-DR9 quasars, measuring clean companion
recovery and photometric errors at the separations and magnitudes of interest.
SDSS is the simplest initial route because its native fluxes and errors are
already attached to the quasar sample and the package has an SDSS training-data
reader. Pan-STARRS deserves the comparison because it provides stacked grizy
imaging across the northern sky and covers more of the DESI-origin remainder;
see the [official Pan-STARRS overview](https://outerspace.stsci.edu/spaces/PANSTARRS/overview).

Choose one optical extension first. Use the same survey measurements and
quality policy for quasars, companions, and background objects. If the pilot
justifies deployment, fit a separate model for that photometric system while
retaining the existing DECaLS model. Add WISE only with a consistent treatment
across those populations. The measured 1.75 percentage-point gain from all
the smaller optical surveys does not justify building several additional
models at this stage.

There is also a separate expansion route: the DESI cache already contains
337,861 rows with northern LS release 9011. These have photometry but are
outside the current southern model. That route matters if “without DECaLS”
means outside the southern system rather than absent from the combined DR9
catalogue; the 337,861 figure is a catalogue row count, not a deduplicated
population estimate.

**Reproducibility.** Local results are under `data/photometry_coverage/`:
`exploration_scope.json` records the parent selection;
`classified_missing.npz` records the DR9 checks;
`survey_sample.npz` records the sampled positions;
`summary_*.json` and `coverage_*.npz` record survey counts and per-object
availability; `overlap_summary.json` records the unions. Query results are
cached as `match_*.npz`, with SQL, radius, and timings in matching JSON files.
The local exploration scripts are `probe_catalogues.py` and `survey_probe.py`.
The source parent caches are `desi_qso_full_5cd2db61c2.npz`,
`dr16q_positions_323d81e938.npz`, and `dr16q_ls_c31d74110c.npz` under `data/`.
The caches and exploration scripts are local working artifacts, not distributed
with the repository. This exploration made no model or scoring-code changes.
