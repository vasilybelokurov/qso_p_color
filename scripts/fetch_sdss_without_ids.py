#!/usr/bin/env python
"""Download outstanding SDSS photometry in resumable spatial blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from build_qso_master import sha256
from fetch_full_qso_photometry import write_json
from qso_pcolor.data import _load_npz, _save_npz, cached_query
from qso_pcolor.qso_acquisition import (sdss_photometric_ids, fetch_sdss_positions,
                                      sdss_position_query as query_sql)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--block-config', type=Path, default=Path('configs/photometry_sky_blocks.json'))
    parser.add_argument('--whole-list', action='store_true', help='Reproduce the earlier one-query acquisition')
    args = parser.parse_args()
    root = args.cache
    provenance = json.loads((root/'provenance.json').read_text())
    cfg = provenance['config']
    master = Path(cfg['master_manifest']).expanduser()
    if sha256(master) != provenance['master_sha256']:
        raise ValueError('master manifest changed')
    out = root/'sdss_without_ids'
    out.mkdir(exist_ok=True)
    request_path = out/'request.json'
    targets = _load_npz(root/'targets.npz')
    fingerprint = sha256(root/'targets.npz')
    radius = cfg['match_radius_arcsec']['sdss']
    sql = query_sql(radius)
    if request_path.exists():
        request = json.loads(request_path.read_text())
        if (request['targets_sha256'] != fingerprint or request['query'] != sql or
                request['master_sha256'] != provenance['master_sha256']):
            raise ValueError('independent positional query provenance differs')
        selection = _load_npz(out/'targets.npz')
    else:
        catalogue = cached_query('SELECT sdss_name,objid FROM sdssdr16qso.main',
                                 root/'queries'/'sdss_identity.npz',
                                 preamb="SET jit=off; SET statement_timeout='7200s'")
        manifest = json.loads(master.read_text())
        if sha256(master.parent/'members.npz') != manifest['files']['members.npz']:
            raise ValueError('master memberships changed')
        ids, conflict = sdss_photometric_ids(_load_npz(master.parent/'members.npz'),
                                             targets['master_index'], catalogue)
        progress = root/'progress_sdss.json'
        done = json.loads(progress.read_text())['processed'] if progress.exists() else 0
        selected = np.flatnonzero((ids <= 0) & (np.arange(len(ids)) >= done))
        selection = dict(target_index=selected, ra=targets['ra'][selected], dec=targets['dec'][selected])
        _save_npz(out/'targets.npz', **selection)
        request = dict(targets_sha256=fingerprint, master_sha256=provenance['master_sha256'],
                       query=sql, targets=len(selected), completed_prefix=done,
                       conflicting_ids=int(conflict[selected].sum()),
                       selection='all outstanding targets with no unique valid SDSS photometric ID',
                       row_cap=None, radius_arcsec=radius)
        write_json(request_path, request)
    (out/'query.sql').write_text(sql+'\n')
    n = len(selection['target_index'])
    if (out/'photometry.npz').exists():
        print(f'Already downloaded all {n:,} targets without IDs', flush=True)
        return
    print(f'{n:,} targets without IDs; {"whole-list" if args.whole_list else "sequential sky-block"} acquisition', flush=True)
    if args.prepare_only:
        return
    started = time.monotonic()
    def status(stage, **extra):
        write_json(out/'status.json', dict(stage=stage, targets=n,
                   elapsed_seconds=time.monotonic()-started, **extra))
        print(stage, flush=True)
    import sqlutilpy as sqlutil
    conn = None
    try:
        if not args.whole_list:
            status('querying sequential sky blocks')
            raw = fetch_sdss_positions(selection['ra'], selection['dec'], out/'blocks',
                radius_arcsec=radius, block_config=json.loads(args.block_config.read_text()),
                progress_path=out/'status.json')
            raw['target_index'] = selection['target_index']
            _save_npz(out/'photometry.npz', **raw)
            status('complete', matched=int(np.isfinite(raw['ra']+raw['dec']).sum()))
            return
        status('uploading targets')
        conn = sqlutil.getConnection(db='wsdb', driver='psycopg')
        sqlutil.upload('mytmptable', (np.arange(n),selection['ra'],selection['dec']),
                       ('idx','ra','dec'), conn=conn, noCommit=True, temp=True, analyze=True)
        status('checking full-list query plan')
        plan = sqlutil.get('EXPLAIN '+sql, conn=conn, notNamed=True, asDict=True,
                          strLength=30000, preamb="SET jit=off; SET statement_timeout='7200s'")
        lines = next(iter(plan.values())).tolist()
        (out/'plan.txt').write_text('\n'.join(lines)+'\n')
        if (any('Seq Scan on photoobjall' in line for line in lines) or
                not any('photoobjall_q3c_ang2ipix_idx' in line for line in lines)):
            raise RuntimeError('full-list SDSS positional query does not use the Q3C index')
        status('querying and downloading', database_pid=int(conn.info.backend_pid))
        raw = sqlutil.get(sql, conn=conn, asDict=True, intNullVal=-1)
        order = np.argsort(raw['idx'])
        if not np.array_equal(raw['idx'][order], np.arange(n)):
            raise ValueError('positional query lost or duplicated target identities')
        raw = {k:v[order] for k,v in raw.items()}
        raw['target_index'] = selection['target_index']
        _save_npz(out/'photometry.npz', **raw)
        status('complete', matched=int(np.isfinite(raw['ra']+raw['dec']).sum()))
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if conn is not None:
            conn.rollback()
            conn.close()


if __name__ == '__main__':
    main()
