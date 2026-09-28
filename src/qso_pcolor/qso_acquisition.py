"""Reuse catalogue identity links during QSO photometry acquisition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import _load_npz, _save_npz, cached_query
from .multisurvey_data import SURVEYS, match_catalogue


def sdss_photometric_ids(members: dict, master_indices: np.ndarray,
                         catalogue: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve dimensionless SDSS photoobj IDs through all master memberships.

    Return int64 IDs (-1 when unresolved) and a conflicting-ID flag per target.
    The Legacy brick ``objid`` in the local spectroscopy cache is unrelated.
    Never convert the 64-bit SDSS identifier through floating point.
    """
    names = np.asarray(catalogue['sdss_name']).astype(str)
    if len(np.unique(names)) != len(names):
        raise ValueError('duplicate SDSS catalogue names')
    order = np.argsort(names)
    names = names[order]
    parsed = np.full(len(names), -1, np.int64)
    for i, value in enumerate(np.asarray(catalogue['objid'])[order]):
        text = str(value).strip()
        if text.isascii() and text.isdigit() and 0 < int(text) <= np.iinfo(np.int64).max:
            parsed[i] = int(text)
    selected = np.asarray(members['catalogue']) == 'sdss'
    sources = np.asarray(members['source_id'])[selected].astype(str)
    groups = np.asarray(members['object_index'])[selected]
    pos = np.searchsorted(names, sources)
    if np.any(pos >= len(names)) or np.any(names[pos] != sources):
        raise ValueError('SDSS membership missing from identity catalogue')
    ids = parsed[pos]
    size = max(int(np.max(master_indices, initial=-1)), int(np.max(groups, initial=-1))) + 1
    low = np.full(size, np.iinfo(np.int64).max, np.int64)
    high = np.full(size, -1, np.int64)
    valid = ids > 0
    np.minimum.at(low, groups[valid], ids[valid])
    np.maximum.at(high, groups[valid], ids[valid])
    conflict = (high > 0) & (low != high)
    high[conflict] = -1
    return high[master_indices], conflict[master_indices]


def fetch_sdss_by_id(ra: np.ndarray, dec: np.ndarray, objid: np.ndarray,
                     cache: Path) -> dict:
    """Fetch DR14 native fluxes by indexed photoobj ID, preserving every row.

    Positions are ICRS degrees; returned separation is arcsec. This is an
    identifier join, not a positional search. Unmatched/nonprimary IDs are
    retained as null rows for subsequent positional fallback.
    """
    import sqlutilpy as sqlutil

    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    objid = np.asarray(objid, np.int64)
    if ra.ndim != 1 or ra.shape != dec.shape or ra.shape != objid.shape:
        raise ValueError('ID and coordinate shapes must agree')
    query = f'''SELECT m.idx, c.objid AS sdss_photometric_objid,
        {SURVEYS['sdss'].columns},
        q3c_dist(m.ra,m.dec,c.ra,c.dec)*3600 AS match_sep_arcsec
        FROM mytmptable m LEFT JOIN sdssdr14.photoobjall c
        ON c.objid=m.objid AND c.mode=1'''
    digest = hashlib.sha256(query.encode()+ra.tobytes()+dec.tobytes()+objid.tobytes()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'sdss_by_id_{digest}.npz'
    if path.exists():
        return _load_npz(path)
    started = time.monotonic()
    conn = sqlutil.getConnection(db='wsdb', driver='psycopg')
    try:
        sqlutil.upload('mytmptable', (np.arange(len(ra)), ra, dec, objid),
                       ('idx', 'ra', 'dec', 'objid'), conn=conn,
                       noCommit=True, temp=True, analyze=True)
        plan = sqlutil.get('EXPLAIN '+query, conn=conn, notNamed=True, asDict=True,
                          strLength=20000, preamb="SET jit=off; SET statement_timeout='7200s'")
        lines = next(iter(plan.values())).tolist()
        if any('Seq Scan on photoobjall' in line for line in lines):
            raise RuntimeError('SDSS ID lookup planned a full photometry-table scan')
        raw = sqlutil.get(query, conn=conn, asDict=True, intNullVal=-1)
    finally:
        conn.rollback()
        conn.close()
    order = np.argsort(raw['idx'])
    result = {k: v[order] for k, v in raw.items()}
    if not np.array_equal(result['idx'], np.arange(len(ra))):
        raise ValueError('SDSS ID join did not preserve one row per target')
    _save_npz(path, **result)
    path.with_suffix('.json').write_text(json.dumps(dict(query=query, plan=lines,
        n=len(ra), elapsed_s=time.monotonic()-started), indent=2)+'\n')
    return result


def usable_id_matches(raw: dict, radius_arcsec: float) -> np.ndarray:
    """ID links usable within the existing positional association radius."""
    sep = raw['match_sep_arcsec']
    return (raw['sdss_photometric_objid'] > 0) & np.isfinite(raw['ra']+raw['dec']+sep) & (sep >= 0) & (sep <= radius_arcsec)


def prepare_sdss_id_photometry(root: Path, targets: dict, master: Path,
                               start: int) -> tuple[dict, dict]:
    """Resolve all remaining linked targets in one ID query, with cached inputs."""
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+master.read_bytes()).hexdigest()[:16]
    saved = root/'id_links'/f'sdss_{fingerprint}.npz'
    record = saved.with_suffix('.json')
    if saved.exists() and record.exists():
        return _load_npz(saved), json.loads(record.read_text())
    catalogue = cached_query('SELECT sdss_name,objid FROM sdssdr16qso.main',
                             root/'queries'/'sdss_identity.npz',
                             preamb="SET jit=off; SET statement_timeout='7200s'")
    members = _load_npz(master.parent/'members.npz')
    ids, conflict = sdss_photometric_ids(members, targets['master_index'], catalogue)
    rows = np.flatnonzero((ids > 0) & (np.arange(len(ids)) >= start))
    print(f'SDSS ID join: {len(rows):,} remaining linked targets in one query', flush=True)
    raw = fetch_sdss_by_id(targets['ra'][rows], targets['dec'][rows], ids[rows], root/'queries') if len(rows) else {}
    if raw:
        raw['target_index'] = rows
    report = dict(valid_link_targets=int((ids > 0).sum()), conflicting_link_targets=int(conflict.sum()),
                  remaining_id_targets=len(rows), preserved_prefix=start,
                  source='sdssdr16qso.main.objid -> sdssdr14.photoobjall.objid',
                  native_dr16q_fluxes_substituted=False)
    _save_npz(saved, **raw)
    record.write_text(json.dumps(report, indent=2)+'\n')
    return raw, report


def acquire_sdss_batch(root: Path, targets: dict, lo: int, hi: int,
                       id_rows: dict, radius_arcsec: float) -> dict:
    """Combine validated ID links and positional fallback, in target order."""
    path = root/'acquired'/f'sdss_{lo:07d}.npz'
    if path.exists():
        return _load_npz(path)
    n = hi-lo
    use = np.zeros(n, bool)
    rows = np.array([], dtype=int)
    if id_rows:
        select = (id_rows['target_index'] >= lo) & (id_rows['target_index'] < hi)
        select &= usable_id_matches(id_rows, radius_arcsec)
        rows = np.flatnonzero(select)
        rows = rows[np.argsort(id_rows['target_index'][rows])]
        if len(np.unique(id_rows['target_index'][rows])) != len(rows):
            raise ValueError('duplicate ID-linked target rows')
        use[id_rows['target_index'][rows]-lo] = True
    result = {}
    if (~use).any():
        raw = match_catalogue('sdss', targets['ra'][lo:hi][~use], targets['dec'][lo:hi][~use],
                              root/'queries', radius_arcsec=radius_arcsec)
        result = {k: np.empty(n, dtype=v.dtype) for k,v in raw.items()}
        for k,v in raw.items():
            result[k][~use] = v
    if use.any():
        for k,v in id_rows.items():
            if k in ('target_index', 'sdss_photometric_objid'):
                continue
            if k not in result:
                result[k] = np.empty(n, dtype=v.dtype)
            result[k][use] = v[rows]
    result['idx'] = np.arange(n)
    result['sdss_id_link_used'] = use
    result['sdss_photometric_objid'] = np.full(n, -1, np.int64)
    if use.any():
        result['sdss_photometric_objid'][use] = id_rows['sdss_photometric_objid'][rows]
    _save_npz(path, **result)
    return result


def read_acquired_batch(root: Path, survey: str, targets: dict, lo: int, hi: int,
                        radius_arcsec: float) -> dict:
    """Read a switched batch, or the original query cache for historical rows."""
    path = root/'acquired'/f'{survey}_{lo:07d}.npz'
    if path.exists():
        return _load_npz(path)
    return match_catalogue(survey, targets['ra'][lo:hi], targets['dec'][lo:hi],
                           root/'queries', radius_arcsec=radius_arcsec)
