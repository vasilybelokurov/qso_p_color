#!/usr/bin/env python
"""Validate the non-Legacy point-source test (qso_pcolor.point_source) against Legacy DR9 ``type``.

Samples, inside the Legacy footprint at |b| >= 25, cleaned as the scorer's inputs (maskbits = 0, Legacy r
S/N >= 10):
  field   every Legacy source in 40 cones of 0.15 deg radius centred on random Quaia quasars with Legacy r
          (so the cones lie in the footprint; the central quasars are dropped), all Tractor types;
  quasars 20,000 random Quaia G<20 quasars with Legacy r (docs/fallback_reference/quaia_no_legacy_r.npz
          classes), with their own Legacy type.
Each source gets Gaia DR3 (nearest within 1 arcsec: G, BP-RP, excess factor) and PS1 (StackObjectThin primary
detection within 1 arcsec, its PS1-PSC ps_score), then the test's label. Reported per Legacy r bin and per
test (Gaia, PS1-PSC): completeness = fraction of Legacy PSF sources labelled PSF; leakage = fraction of Legacy
extended sources labelled PSF; purity = fraction of sources labelled PSF that are Legacy PSF (field sample);
for quasars, the fraction labelled PSF among Legacy-PSF quasars.

Writes docs/point_source/validation.json; WSDB pulls cached in models/multisurvey_psf/work/point_source/.
Usage: python scripts/point_source/validate_point_source.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from qso_pcolor.point_source import point_source_morphology

WORK = Path('models/multisurvey_psf/work/point_source'); DOC = Path('docs/point_source/validation.json')
QUAIA = Path('~/data/catalogues/quaia_G20.0.fits').expanduser()
SEED = 20261013
LEGACY_COLS = 'c.ra, c.dec, c.type, c.flux_r, c.flux_ivar_r, c.maskbits, c.release'


def cached(name, query, cols, arrays):
    import sqlutilpy as sqlutil
    key = hashlib.sha256(query.encode() + b''.join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()[:16]
    path = WORK/f'{name}_{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    res = sqlutil.local_join(query, 'mytmptable', arrays, cols, asDict=True, intNullVal=-1,
                             preamb="SET jit=off; SET statement_timeout='7200s'")
    res = {k: (v.astype(str) if v.dtype == object else v) for k, v in res.items()}
    WORK.mkdir(parents=True, exist_ok=True); np.savez(path, **res)
    return res


def legacy_cones(ra, dec, radius):
    # per-cone LATERAL radial query (OFFSET 0 fences the filters out): with the filters in the join the
    # planner chose a sequential scan of all 152M rows
    q = f"""SELECT m.cone, x.* FROM mytmptable m CROSS JOIN LATERAL (
              SELECT {LEGACY_COLS} FROM decals_dr9.main c
              WHERE q3c_radial_query(c.ra, c.dec, m.ra, m.dec, {radius}) OFFSET 0) x
            WHERE x.maskbits = 0 AND x.flux_ivar_r > 0 AND x.flux_r*sqrt(x.flux_ivar_r) >= 10"""
    return cached('legacy_cones', q, ('cone', 'ra', 'dec'), (np.arange(len(ra)), ra, dec))


def nearest(name, table, cols, ra, dec, where='TRUE', radius=1.):
    q = f"""SELECT m.idx, x.* FROM mytmptable m LEFT JOIN LATERAL (
              SELECT {cols}, q3c_dist(m.ra, m.dec, c.ra, c.dec)*3600 AS sep FROM {table} c
              WHERE q3c_join(m.ra, m.dec, c.ra, c.dec, {radius}/3600.) AND ({where})
              ORDER BY q3c_dist(m.ra, m.dec, c.ra, c.dec) LIMIT 1) x ON TRUE"""
    r = cached(name, q, ('idx', 'ra', 'dec'), (np.arange(len(ra)), ra, dec))
    o = np.argsort(r['idx']); return {k: v[o] for k, v in r.items()}


def ps1_score(objid):
    q = """SELECT m.idx, h.ps_score FROM mytmptable m LEFT JOIN panstarrs_dr1.hlsp h ON h.objid = m.objid"""
    r = cached('ps1_psc', q, ('idx', 'objid'), (np.arange(len(objid)), objid))
    o = np.argsort(r['idx']); return r['ps_score'][o]


def label(ra, dec):
    g = nearest('gaia', 'gaia_dr3.gaia_source', 'c.phot_g_mean_mag, c.bp_rp, c.phot_bp_rp_excess_factor', ra, dec)
    p = nearest('ps1', 'panstarrs_dr1.stackobjectthin', 'c.objid', ra, dec, where='c.primarydetection = 1')
    objid = np.where(np.asarray(p['objid']) > 0, p['objid'], -1).astype(np.int64)
    ps = ps1_score(objid).astype(float); ps[objid < 0] = np.nan
    ps[~np.isfinite(ps)] = np.nan
    f = lambda k: np.asarray(g[k], float)
    lab, used = point_source_morphology(gaia_g=f('phot_g_mean_mag'), gaia_bp_rp=f('bp_rp'),
                                        gaia_excess_factor=f('phot_bp_rp_excess_factor'), ps_score=ps)
    lab_gaia, _ = point_source_morphology(gaia_g=f('phot_g_mean_mag'), gaia_bp_rp=f('bp_rp'),
                                          gaia_excess_factor=f('phot_bp_rp_excess_factor'))
    lab_psc, _ = point_source_morphology(ps_score=ps)
    return dict(label=lab, used=used, gaia=lab_gaia, psc=lab_psc, g=f('phot_g_mean_mag'))


def rmag(d):
    with np.errstate(all='ignore'):
        return 22.5 - 2.5*np.log10(np.asarray(d['flux_r'], float))


def table(is_psf, lab, r, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi); ps, ex = m & is_psf, m & ~is_psf
        known = lab != 'unknown'
        out.append(dict(r=[lo, hi], n_psf=int(ps.sum()), n_ext=int(ex.sum()),
            tested_psf=float((ps & known).sum()/max(ps.sum(), 1)),
            completeness=float(((lab == 'PSF') & ps).sum()/max((ps & known).sum(), 1)),
            leakage=float(((lab == 'PSF') & ex).sum()/max((ex & known).sum(), 1)),
            purity=float(((lab == 'PSF') & ps).sum()/max(((lab == 'PSF') & m).sum(), 1))))
    return out


def main():
    rng = np.random.default_rng(SEED)
    q = fits.getdata(QUAIA, 1); nl = np.load('models/multisurvey_psf/work/fallback_reference/quaia_no_legacy_r.npz')
    keep = (np.abs(q['b']) >= 25) & ~np.isin(q['source_id'], nl['source_id'])          # Quaia with Legacy r
    q = q[keep]
    cen = q[rng.choice(len(q), 40, replace=False)]
    field = legacy_cones(np.asarray(cen['ra'], float), np.asarray(cen['dec'], float), .15)
    sep = np.min([np.hypot((field['ra'] - c['ra'])*np.cos(np.radians(c['dec'])), field['dec'] - c['dec'])
                  for c in cen], axis=0)*3600
    field = {k: v[sep > 2] for k, v in field.items()}
    qq = q[rng.choice(len(q), 20000, replace=False)]
    leg = nearest('quaia_legacy', 'decals_dr9.main', LEGACY_COLS.replace('c.ra, c.dec, ', ''),
                  np.asarray(qq['ra'], float), np.asarray(qq['dec'], float))
    okq = (np.asarray(leg['maskbits']) == 0) & (np.asarray(leg['flux_ivar_r'], float) > 0)
    with np.errstate(all='ignore'):
        okq &= np.asarray(leg['flux_r'], float)*np.sqrt(np.asarray(leg['flux_ivar_r'], float)) >= 10
    rep = dict(n_cones=40, cone_radius_deg=.15, gaia_n_sigma=3., psc_threshold=.83)
    edges = [14, 17, 18, 19, 20, 20.5, 21, 21.5, 22, 23, 24.5]
    for name, d, ra, dec in (('field', field, field['ra'], field['dec']),
                             ('quasars', {k: np.asarray(v)[okq] for k, v in leg.items()},
                              np.asarray(qq['ra'], float)[okq], np.asarray(qq['dec'], float)[okq])):
        L = label(np.asarray(ra, float), np.asarray(dec, float))
        is_psf = np.char.strip(np.asarray(d['type']).astype(str)) == 'PSF'; r = rmag(d)
        rep[name] = dict(n=int(len(r)), legacy_psf_fraction=float(is_psf.mean()),
                         used={k: int((L['used'] == k).sum()) for k in ('gaia', 'ps1_psc', '')},
                         combined=table(is_psf, L['label'], r, edges), gaia_only=table(is_psf, L['gaia'], r, edges),
                         psc_only=table(is_psf, L['psc'], r, edges))
        print(name, rep[name]['n'], rep[name]['used'], flush=True)
    DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
