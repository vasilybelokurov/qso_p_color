#!/usr/bin/env python
"""Quaia quasars without usable Legacy r: how many, where, and what other photometry they have.

Quaia G<20 (Storey-Fisher et al. 2024, ApJ 964, 69; local ~/data/catalogues/quaia_G20.0.fits) at |b| >= 25
is matched to Legacy DR9 (nearest within 1 arcsec, the training match) with the training quality rules
(configs/full_sample_photometry.json: clean=True, i.e. maskbits == 0 and nobs > 0). Objects are split into
  has_r        usable Legacy r (South or North), as the promoted bundles require;
  masked_r     a Legacy counterpart whose r is unusable (maskbits != 0, nobs_r = 0 or ivar <= 0);
  no_legacy    no Legacy DR9 source within 1 arcsec.
For masked_r + no_legacy the other six surveys are matched with the training radii and quality rules,
and their native fluxes saved (input for the fallback-reference scorer).

Writes  models/multisurvey_psf/work/fallback_reference/quaia_no_legacy_r.npz (photometry; local)
        docs/fallback_reference/quaia_no_legacy_r.json (counts, coverage, S/N, positions)
Usage:  python scripts/fallback_reference/quaia_no_legacy_r.py
"""
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from qso_pcolor.multisurvey_data import Photometry, catalogue_photometry, match_catalogue

CFG = json.loads(Path('configs/full_sample_photometry.json').read_text())
WORK = Path('models/multisurvey_psf/work/fallback_reference'); CACHE = WORK/'queries'
DOC = Path('docs/fallback_reference/quaia_no_legacy_r.json')
QUAIA = Path('~/data/catalogues/quaia_G20.0.fits').expanduser()
BATCH = 50000


def match(survey, ra, dec):
    """Training-identical nearest match and native photometry, in cached batches."""
    parts = []
    for lo in range(0, len(ra), BATCH):
        raw = match_catalogue(survey, ra[lo:lo+BATCH], dec[lo:lo+BATCH], CACHE,
                              radius_arcsec=CFG['match_radius_arcsec'][survey])
        parts.append((raw, catalogue_photometry(survey, raw, clean=CFG['clean'], vhs_bad_bits=CFG['vhs_bad_bits'])))
        print(f'{survey}: {min(lo+BATCH, len(ra)):,}/{len(ra):,}', flush=True)
    raw = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    ph = parts[0][1]
    return raw, Photometry(np.concatenate([p[1].flux for p in parts]), np.concatenate([p[1].variance for p in parts]), ph.bands)


def snr(ph, band):
    j = ph.bands.index(band)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ph.observed[:, j], ph.flux[:, j]/np.sqrt(ph.variance[:, j]), 0.)


def main():
    q = fits.getdata(QUAIA, 1)
    keep = np.abs(q['b']) >= CFG['min_abs_b_deg']
    q = q[keep]
    # sky-ordered queries (as fetch_full_qso_photometry: HEALPix nest order) keep the survey reads local
    import healpy as hp
    q = q[np.argsort(hp.ang2pix(CFG['query_order_nside'], np.asarray(q['ra'], float), np.asarray(q['dec'], float),
                                nest=True, lonlat=True), kind='stable')]
    ra, dec = np.asarray(q['ra'], float), np.asarray(q['dec'], float)
    print(f'Quaia G<20: {len(keep):,}; |b|>=25: {len(q):,}', flush=True)
    raw, leg = match('decals', ra, dec)
    found = np.isfinite(np.asarray(raw['ra'], float))
    has_r = leg.observed[:, leg.bands.index('decals_dr9_south:r')] | leg.observed[:, leg.bands.index('decals_dr9_north:r')]
    cls = np.where(has_r, 'has_r', np.where(found, 'masked_r', 'no_legacy'))
    out = np.flatnonzero(~has_r)
    phot = {'decals': leg.subset(out)}
    for s in ('sdss', 'allwise', 'ps1', 'nsc', 'skymapper', 'vhs'):
        _, phot[s] = match(s, ra[out], dec[out])
    bands = sum((p.bands for p in phot.values()), ())
    F = np.concatenate([p.flux for p in phot.values()], axis=1); V = np.concatenate([p.variance for p in phot.values()], axis=1)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK/'quaia_no_legacy_r.npz', source_id=np.asarray(q['source_id'])[out], ra=ra[out], dec=dec[out],
             l=np.asarray(q['l'], float)[out], b=np.asarray(q['b'], float)[out], z=np.asarray(q['redshift_quaia'], float)[out],
             G=np.asarray(q['phot_g_mean_mag'], float)[out], cls=cls[out], flux=F, variance=V, bands=np.array(bands))
    rep = dict(input=str(QUAIA), n_quaia=int(len(keep)), n_abs_b_ge_25=int(len(q)),
               classes={c: int((cls == c).sum()) for c in ('has_r', 'masked_r', 'no_legacy')})
    for c in ('masked_r', 'no_legacy'):
        m = cls[out] == c; d = dec[out][m]
        sub = {}
        for name, band in (('ps1_r', 'ps1:r'), ('skymapper_r', 'skymapper:r'), ('sdss_r', 'sdss:r'), ('nsc_r', 'nsc:r')):
            s = snr(phot[name.split('_')[0]], band)[m]
            sub[name] = dict(observed=int((s > 0).sum()), snr_ge_10=int((s >= 10).sum()), snr_ge_20=int((s >= 20).sum()),
                             median_snr_observed=float(np.median(s[s > 0])) if (s > 0).any() else None)
        ps1, sm = snr(phot['ps1'], 'ps1:r')[m] >= 10, snr(phot['skymapper'], 'skymapper:r')[m] >= 10
        sub['ps1_or_skymapper_r_snr_ge_10'] = int((ps1 | sm).sum())
        sub['neither'] = int((~ps1 & ~sm).sum())
        sub['any_survey_observed'] = {s: int(p.observed[m].any(1).sum()) for s, p in phot.items() if s != 'decals'}
        sub['dec_quantiles_5_25_50_75_95'] = [round(float(x), 1) for x in np.percentile(d, [5, 25, 50, 75, 95])] if m.any() else []
        sub['dec_bins'] = {f'{lo}..{hi}': int(((d >= lo) & (d < hi)).sum()) for lo, hi in
                           ((-90, -65), (-65, -30), (-30, -20), (-20, 0), (0, 32), (32, 40), (40, 90))}
        rep[c] = sub
    DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
