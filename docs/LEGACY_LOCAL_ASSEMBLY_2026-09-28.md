# Legacy QSO photometry: local assembly, 28 September 2026

All 1,657,444 eligible QSO targets now have verified, target-aligned Legacy
photometry saved on disk. Assembly took 19.6 seconds and issued no
new database queries. No model was fitted or replaced.

The sources are `~/data/qso_p_color/catalogues/desi_dr1_qso.npz` for
1,312,233 DESI-preferred objects and `dr16q_dr9.npz` in the same directory for
345,211 SDSS-preferred objects. The expanded DESI master has a different row
order: exact integer `targetid` values recover the photometry. SDSS source rows
refer to the original checksum-verified catalogue; `ls_ra`, `ls_dec` identify
the Legacy counterpart. Every counterpart lies within the declared 1 arcsec
radius (maximum 0.989872455 arcsec) and has the required hemisphere.

## The eight flag differences

DESI cache provenance records W1/W2 `nobs` as a positive-inverse-variance proxy,
not an actual exposure count. Eight objects have zero inverse variance, hence
proxy count zero, while DR9 reports positive exposure counts. Both paths reject
these measurements because inverse variance is zero. Their final band masks
agree. No corrective query or changed quality rule is required.

Comparison against 150,000 already downloaded rows gives 148,361 exact matches
in all five fluxes and inverse variances. The other 1,639 have a different
release: the generic nearest-position query selected the opposite hemisphere,
whereas the reused catalogue has the required hemisphere. At unchanged release
there are no flux, inverse-variance or final band-mask differences.

## Saved products and verification

The complete aligned product is `models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff/legacy_local/4d418c295df8aa17/photometry.npz`.
It contains `object_id`, `master_index`, `target_index`, `flux`, `variance`,
`observed` and `bands`. North and south occupy distinct band columns; unused
hemisphere columns are masked. All 12,890 usable negative-flux measurements
are retained. Row identities, all fluxes and variances, band labels and masks
were reread and checked after writing.

Its SHA256 is `be8ebe742db6369e45bad3a1e4d2505b288ea2e3cd0f7c8049b97ab05aba7e33`.
The adjacent `report.json` records source hashes, implementation hash, comparison
counts and hashes of all 34 target-aligned raw batches under `acquired/`.
The existing acquisition reader and association auditor use those same batches.
The pointer is `legacy_local_sources.json` in the acquisition root.

Reproduce with:

```bash
python scripts/fetch_full_qso_photometry.py \
  --config configs/full_sample_photometry.json \
  --resume-cache models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff \
  --surveys decals
```

Legacy has no database fallback. Acquisition now requires explicit survey
selection, preventing unintended downloads of subsequent surveys. Existing
acquired batches are verified and conflicting files are rejected.

The independent multiple/shared-association audit remains pending. The other
five QSO surveys require a local-cache inventory before further acquisition.
This completes Legacy assembly, not all 41-band training preparation.
