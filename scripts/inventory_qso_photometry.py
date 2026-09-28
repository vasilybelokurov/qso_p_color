#!/usr/bin/env python
"""Audit discovered local native photometry; never query or change acquisitions.

The discovery manifest lists NPZ files whose headers contain every required
native measurement and quality column. Its paths are resolved to avoid counting
symlinks twice. This audit saves per-target references, not a training catalogue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from qso_pcolor.data import _load_npz, _save_npz
from qso_pcolor.legacy_acquisition import file_hash
from qso_pcolor.multisurvey_data import SURVEYS, catalogue_photometry

SURVEY_NAMES = ('allwise', 'ps1', 'nsc', 'skymapper', 'vhs')


def xyz(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """ICRS angles in degrees to unit vectors, including across RA zero."""
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)))


def chord(degrees: float) -> float:
    return 2*np.sin(np.deg2rad(degrees)/2)


def exact_target_rows(tree: cKDTree, targets: dict, source: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map only identical input coordinates; nearby queries are not equivalent."""
    dist, idx = tree.query(xyz(source['ra'], source['dec']))
    same = (targets['ra'][idx] == source['ra']) & (targets['dec'][idx] == source['dec'])
    return np.flatnonzero(same), idx[same]


def cone_targets(tree: cKDTree, ra: float, dec: float, radius: float,
                 match_arcsec: float) -> np.ndarray:
    """Targets whose entire matching aperture lies inside a cached cone."""
    safe = radius - match_arcsec/3600
    if safe <= 0:
        return np.empty(0, int)
    return np.sort(tree.query_ball_point(xyz([ra], [dec])[0], chord(safe)))


def discover_native_files(roots: list[Path]) -> dict:
    """Read NPZ headers once per resolved path; do not load measurement arrays."""
    paths = set()
    for root in roots:
        result = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', str(root.expanduser())],
                                capture_output=True, text=True, check=True)
        paths.update(Path(p).resolve() for p in result.stdout.splitlines() if p.endswith('.npz'))
    records, errors = [], []
    for p in sorted(paths):
        try:
            with zipfile.ZipFile(p) as z:
                keys = {k.removesuffix('.npy') for k in z.namelist()}
            for name in SURVEY_NAMES:
                spec = SURVEYS[name]
                required = {'ra', 'dec', *spec.extra} | {
                    f'{prefix}_{b}' for prefix in ('value', 'error') for b in spec.bands}
                if required <= keys:
                    records.append(dict(path=str(p), survey=name, bytes=p.stat().st_size, keys=sorted(keys)))
        except (OSError, zipfile.BadZipFile) as error:
            errors.append(dict(path=str(p), error=str(error)))
    return dict(scan_roots=[str(p.expanduser().resolve()) for p in roots],
                npz_headers_scanned=len(paths), files=records, errors=errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--scan-roots', nargs='+', type=Path,
                        default=[Path.home()/'data', Path('models')])
    args = parser.parse_args()
    root, out = args.cache, args.inventory
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'discovered_native.json').exists():
        (out/'discovered_native.json').write_text(
            json.dumps(discover_native_files(args.scan_roots), indent=2)+'\n')
    cfg = json.loads((root/'provenance.json').read_text())['config']
    targets = _load_npz(root/'targets.npz')
    tree = cKDTree(xyz(targets['ra'], targets['dec']))
    n = len(targets['ra'])
    discovery = json.loads((out/'discovered_native.json').read_text())
    records, summary = [], {}
    for survey in SURVEY_NAMES:
        spec = SURVEYS[survey]
        radius = cfg['match_radius_arcsec'][survey]
        source_file = np.full(n, -1, np.int32)
        source_row = np.full(n, -1, np.int64)
        pool_file, pool_row = source_file.copy(), source_row.copy()
        observed = np.zeros((n, len(spec.bands)), bool)
        counterpart = np.zeros(n, bool)
        seen = {}
        for entry in [r for r in discovery['files'] if r['survey'] == survey]:
            p = Path(entry['path'])
            digest = file_hash(p)
            record = dict(entry, sha256=digest, file_index=len(records))
            if digest in seen:
                record.update(kind='byte_identical_duplicate', duplicate_of=seen[digest])
                records.append(record)
                continue
            seen[digest] = len(records)
            rows = _load_npz(p)
            meta = p.with_suffix('.json')
            if meta.exists():
                query = json.loads(meta.read_text())['query']
                record.update(metadata=str(meta), metadata_sha256=file_hash(meta))
            else:
                with np.load(p) as z:
                    query = str(z['_query']) if '_query' in z else ''
            record.update(rows=len(rows['ra']), query=query)
            if f'FROM {spec.table} c' not in query or f'AND ({spec.where})' not in query:
                record.update(kind='unverified_survey_provenance')
                records.append(record)
                continue
            phot = catalogue_photometry(survey, rows, clean=cfg['clean'], vhs_bad_bits=cfg['vhs_bad_bits'])
            finite = np.isfinite(rows['ra']) & np.isfinite(rows['dec'])
            valid_rows = np.flatnonzero(finite)
            # Existing measurements near QSO positions are useful, but a query
            # centred elsewhere cannot establish nearest association/nonmatch.
            nearby = tree.query_ball_point(xyz(rows['ra'][finite], rows['dec'][finite]),
                                          chord(radius/3600))
            has = np.array([len(a)>0 for a in nearby])
            hit_rows = np.flatnonzero(has)
            for j in hit_rows:
                ti = np.asarray(nearby[j], int)
                fresh = pool_file[ti] < 0
                pool_file[ti[fresh]] = len(records)
                pool_row[ti[fresh]] = valid_rows[j]
            record['nearby_target_references'] = sum(len(nearby[j]) for j in hit_rows)
            src, dst = np.empty(0, int), np.empty(0, int)
            record['kind'] = 'measurement_pool_only'
            target_path = p.parent.parent/'targets.npz'
            if 'mytmptable' in query and target_path.exists():
                old = _load_npz(target_path)
                expected = hashlib.sha256(query.encode()+old['ra'].tobytes()+old['dec'].tobytes()).hexdigest()[:16]
                if p.stem == survey+'_'+expected:
                    if not np.array_equal(rows['idx'], np.arange(len(old['ra']))):
                        raise ValueError(f'cache row indices differ: {p}')
                    pattern = rf'q3c_join\(m.ra,m.dec,c.{spec.ra},c.{spec.dec},([0-9.]+)/3600.\)'
                    match = re.search(pattern, query)
                    if match is None or float(match[1]) != radius:
                        raise ValueError(f'cache match radius differs: {p}')
                    src, dst = exact_target_rows(tree, targets, old)
                    record.update(kind='exact_target_query', input_targets=str(target_path),
                                  input_targets_sha256=file_hash(target_path))
            cone = re.search(r'q3c_radial_query\(c\.[^,]+,c\.[^,]+,([^,]+),([^,]+),([^\)]+)\)', query)
            if cone and 'mytmptable' not in query:
                centre = tuple(float(x) for x in cone.groups())
                # Only the complete, unfiltered cone template qualifies.
                expected = (f'SELECT {spec.columns} FROM {spec.table} c '
                            f'WHERE q3c_radial_query(c.{spec.ra},c.{spec.dec},'
                            f'{cone[1]},{cone[2]},{cone[3]}) AND ({spec.where})')
                if ' '.join(query.split()) == ' '.join(expected.split()):
                    dst = cone_targets(tree, *centre, radius)
                    src = np.full(len(dst), -1, int)
                    if len(valid_rows) and len(dst):
                        distance, nearest = cKDTree(xyz(rows['ra'][finite], rows['dec'][finite])).query(
                            tree.data[dst], distance_upper_bound=chord(radius/3600))
                        ok = np.isfinite(distance)
                        src[ok] = valid_rows[nearest[ok]]
                    record.update(kind='complete_cone', centre_radius_deg=centre)
            record['reusable_query_results'] = len(dst)
            fresh = source_file[dst] < 0
            dst, src = dst[fresh], src[fresh]
            source_file[dst] = len(records)
            source_row[dst] = src
            matched = src >= 0
            if np.any(matched):
                observed[dst[matched]] = phot.observed[src[matched]]
                counterpart[dst[matched]] = finite[src[matched]]
            record['new_target_results'] = len(dst)
            records.append(record)
        covered = source_file >= 0
        gap = np.flatnonzero(~covered)
        potential = (~covered) & (pool_file >= 0)
        _save_npz(out/f'{survey}_target_inventory.npz', object_id=targets['object_id'],
                  target_index=np.arange(n), reusable_source_file=source_file,
                  reusable_source_row=source_row, counterpart=counterpart, observed=observed,
                  nearby_source_file=pool_file, nearby_source_row=pool_row,
                  unresolved_target_index=gap)
        kinds = np.array([records[i]['kind'] if i >= 0 else 'none' for i in source_file])
        summary[survey] = dict(targets=n, reusable_query_results=int(covered.sum()),
            exact_target_results=int((kinds=='exact_target_query').sum()),
            complete_cone_results=int((kinds=='complete_cone').sum()),
            known_no_counterpart=int((covered & ~counterpart).sum()),
            counterpart_present=int((covered & counterpart).sum()),
            usable_any_band=int(observed.any(axis=1).sum()), usable_per_band=observed.sum(axis=0).tolist(),
            bands=list(spec.bands), unresolved_targets=len(gap),
            unresolved_with_nearby_measurements=int(potential.sum()),
            unresolved_without_local_measurements=int(((~covered) & (pool_file<0)).sum()),
            reference_file=str(out/f'{survey}_target_inventory.npz'))
        print(survey, json.dumps(summary[survey]), flush=True)
    (out/'native_files.json').write_text(json.dumps(records, indent=2)+'\n')
    result = dict(target_file=str(root/'targets.npz'), target_sha256=file_hash(root/'targets.npz'),
                  implementation_sha256=file_hash(Path(__file__)),
                  discovery_manifest=str(out/'discovered_native.json'),
                  discovery_sha256=file_hash(out/'discovered_native.json'),
                  native_file_manifest=str(out/'native_files.json'),
                  npz_headers_scanned=discovery['npz_headers_scanned'],
                  native_files=len(records), surveys=summary, new_database_queries=0,
                  association_audit_complete=False, model_fitted=False)
    result['output_sha256']={p.name:file_hash(p) for p in out.glob('*_target_inventory.npz')}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
