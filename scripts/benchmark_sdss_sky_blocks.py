#!/usr/bin/env python
"""Measure positional block costs on disjoint sky cells outside this acquisition.

No benchmark object enters the training sample. A first EXPLAIN ANALYZE measures
server execution and buffer reads; a subsequent retrieval verifies and saves
actual rows. Their sum is a conservative operational timing (two executions),
not the end-to-end cost of a single cold download. Shared WSDB load is recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import healpy as hp
import numpy as np
from scipy.spatial import cKDTree
import sqlutilpy as sqlutil

from build_qso_master import sha256
from qso_pcolor.data import _load_npz, _save_npz
from qso_pcolor.qso_acquisition import sdss_position_query, fetch_sdss_positions
from qso_pcolor.sky_acquisition import write_record


def xyz(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--min-sdss-members-per-cell', type=int, required=True,
                        help='Restrict timing targets to cells with known SDSS coverage')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or min(args.sizes) < 1:
        raise ValueError('distinct positive block sizes required')
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.cache/'provenance.json').read_text())
    master = Path(provenance['config']['master_manifest']).expanduser()
    objects = _load_npz(master.parent/'objects.npz')
    targets = _load_npz(args.cache/'targets.npz')
    selection_path = args.out/'selection.json'
    if not selection_path.exists():
        pix = hp.ang2pix(64, objects['ra'], objects['dec'], lonlat=True, nest=True)
        used = np.unique(pix[targets['master_index']])
        # Exclude all production target cells and touching cells, including cells
        # the live download has not reached. No claim about other WSDB users.
        used = np.unique(np.r_[used, hp.get_all_neighbours(64, used, nest=True).ravel()])
        eligible = ~np.isin(pix, used)
        covered, counts = np.unique(pix[objects['n_sdss'] > 0], return_counts=True)
        eligible &= np.isin(pix, covered[counts >= args.min_sdss_members_per_cell])
        regions_path = Path('models/multisurvey_psf/work/full_sample_preparation/stellar_seed_regions.json')
        regions = json.loads(regions_path.read_text())
        tree = cKDTree(xyz(np.array([c['ra'] for c in regions]), np.array([c['dec'] for c in regions])))
        separation = tree.query(xyz(objects['ra'], objects['dec']))[0]
        eligible &= separation > 2*np.sin(np.deg2rad(.6)/2)
        fine = hp.ang2pix(1024, objects['ra'], objects['dec'], lonlat=True, nest=True)
        parent = fine//1024**2
        rng = np.random.default_rng(args.seed)
        trials = []
        for size in sorted(args.sizes, reverse=True):
            parents, counts = np.unique(parent[eligible], return_counts=True)
            available = parents[counts >= size]
            if not len(available):
                raise ValueError(f'insufficient untouched targets for size {size}; use smaller sizes')
            cell = int(rng.choice(available))
            rows = np.flatnonzero(eligible & (parent == cell))
            rows = rows[np.argsort(fine[rows], kind='stable')][:size]
            _save_npz(args.out/f'targets_{size}.npz', master_index=rows,
                      ra=objects['ra'][rows], dec=objects['dec'][rows])
            trials.append(dict(size=size, parent_nside=1, parent_cell=cell,
                nside64_cells=np.unique(pix[rows]).tolist(),
                sdss_membership_fraction=float((objects['n_sdss'][rows] > 0).mean())))
            touched = np.unique(pix[rows])
            touched = np.unique(np.r_[touched, hp.get_all_neighbours(64, touched, nest=True).ravel()])
            eligible &= ~np.isin(pix, touched)
        # Shortest job first gives an early runtime check; regions are disjoint.
        selection = dict(seed=args.seed, master_sha256=sha256(master),
            min_sdss_members_per_cell=args.min_sdss_members_per_cell,
            targets_sha256=sha256(args.cache/'targets.npz'), trials=sorted(trials,key=lambda t:t['size']),
            stellar_regions_sha256=sha256(regions_path),
            limitation='Outside production target cells and neighbours at nside64, and >0.6 deg from stellar cones. External cache/load uncontrolled; these are different sky regions, not a causal speedup comparison.')
        write_record(selection_path, selection)
    selection = json.loads(selection_path.read_text())
    if (selection['master_sha256'] != sha256(master) or
            selection['targets_sha256'] != sha256(args.cache/'targets.npz') or
            sorted(t['size'] for t in selection['trials']) != sorted(args.sizes) or
            selection['seed'] != args.seed or
            selection['min_sdss_members_per_cell'] != args.min_sdss_members_per_cell):
        raise ValueError('benchmark request differs from its saved selection')
    print(json.dumps({k:v for k,v in selection.items() if k != 'trials'}), flush=True)
    if args.prepare_only:
        return
    query = sdss_position_query(provenance['config']['match_radius_arcsec']['sdss'])
    (args.out/'query.sql').write_text(query+'\n')
    for trial in selection['trials']:
        n = trial['size']
        out = args.out/f'trial_{n}'
        out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            continue
        inputs = _load_npz(args.out/f'targets_{n}.npz')
        conn = sqlutil.getConnection(db='wsdb', driver='psycopg')
        started = time.monotonic()
        try:
            active = sqlutil.get("SELECT pid,state,wait_event_type,wait_event FROM pg_stat_activity WHERE usename=current_user AND state='active' AND pid<>pg_backend_pid()", conn=conn, asDict=True, notNamed=True)
            write_record(out/'load_before.json', {k:v.tolist() for k,v in active.items()})
            sqlutil.upload('mytmptable', (np.arange(n), inputs['ra'], inputs['dec']),
                ('idx','ra','dec'), conn=conn, noCommit=True, temp=True, analyze=True)
            plan = sqlutil.get('EXPLAIN '+query, conn=conn, asDict=True, notNamed=True,
                strLength=30000, preamb="SET jit=off; SET cursor_tuple_fraction=1; SET statement_timeout='1200s'")
            lines = next(iter(plan.values())).tolist()
            if (any('Seq Scan on photoobjall' in l for l in lines) or
                    not any('photoobjall_q3c_ang2ipix_idx' in l for l in lines)):
                raise RuntimeError('benchmark plan does not use Q3C')
            preparation = time.monotonic()-started
            print(f'{n:,}: first server execution, backend {conn.info.backend_pid}', flush=True)
            plan = sqlutil.get('EXPLAIN (ANALYZE,BUFFERS,TIMING OFF) '+query, conn=conn,
                asDict=True, notNamed=True, strLength=50000)
            text = '\n'.join(next(iter(plan.values())).tolist())
            (out/'first_execution.plan').write_text(text+'\n')
            server_s = float(re.search(r'Execution Time: ([\d.]+)', text).group(1))/1000
            buffers = next(l for l in text.splitlines() if 'Buffers:' in l)
            reads = re.search(r'read=(\d+)', buffers)
            hits = re.search(r'hit=(\d+)', buffers)
        finally:
            conn.rollback()
            conn.close()
        first_elapsed = time.monotonic()-started
        print(f'{n:,}: server {server_s:.2f}s; retrieving and verifying rows through block runner', flush=True)
        retrieval = time.monotonic()
        raw = fetch_sdss_positions(inputs['ra'], inputs['dec'], out/'download',
            radius_arcsec=provenance['config']['match_radius_arcsec']['sdss'],
            block_config=dict(max_targets=n, parent_nside=1, order_nside=1024))
        retrieval = time.monotonic()-retrieval
        if not np.array_equal(raw['idx'], np.arange(n)):
            raise ValueError('benchmark retrieval identity mismatch')
        result = dict(trial, preparation_seconds=preparation, first_server_seconds=server_s,
            first_execution_wall_seconds=first_elapsed,
            shared_reads=int(reads.group(1)) if reads else 0,
            shared_hits=int(hits.group(1)) if hits else 0,
            warm_retrieval_seconds=retrieval, conservative_two_execution_seconds=first_elapsed+retrieval,
            matches=int(np.isfinite(raw['ra']+raw['dec']).sum()), rows_returned=len(raw['idx']))
        write_record(out/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'nside64_cells'}), flush=True)

    # Exact same targets and measurements, partitioned differently. This warm
    # check is a correctness test, never another cold timing observation.
    n = max(args.sizes)
    inputs = _load_npz(args.out/f'targets_{n}.npz')
    radius = provenance['config']['match_radius_arcsec']['sdss']
    original = fetch_sdss_positions(inputs['ra'], inputs['dec'],
        args.out/f'trial_{n}'/'download', radius_arcsec=radius,
        block_config=dict(max_targets=n, parent_nside=1, order_nside=1024))
    partitioned = fetch_sdss_positions(inputs['ra'], inputs['dec'],
        args.out/'partition_check', radius_arcsec=radius,
        block_config=dict(max_targets=min(args.sizes), parent_nside=1, order_nside=1024))
    identical = set(original) == set(partitioned) and all(
        np.array_equal(original[k], partitioned[k], equal_nan=True) for k in original)
    write_record(args.out/'partition_check.json', dict(targets=n,
        block_size=min(args.sizes), identical=identical, columns=list(original)))
    if not identical:
        raise ValueError('partitioning changed downloaded photometry')
    print(f'Partition check: all {n:,} rows and all returned columns identical', flush=True)


if __name__ == '__main__':
    main()
