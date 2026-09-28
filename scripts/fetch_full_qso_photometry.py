#!/usr/bin/env python
"""Fetch uncapped QSO photometry in resumable sky-ordered batches; never fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import healpy as hp
import numpy as np

from build_qso_master import sha256
from qso_pcolor.data import _save_npz, galactic_from_equatorial
from qso_pcolor.multisurvey_data import SURVEYS, band_labels, catalogue_photometry, match_catalogue
from qso_pcolor.qso_acquisition import (prepare_sdss_id_photometry, acquire_sdss_batch,
                                      read_acquired_batch, wait_for_sdss_positions)


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--surveys', nargs='+', choices=tuple(SURVEYS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-cache', type=Path,
                        help='Reuse a previous acquisition with identical config and master')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['batch_size'] < 1:
        raise ValueError('batch_size must be positive; it is not a row cap')
    master = Path(cfg['master_manifest']).expanduser()
    manifest = json.loads(master.read_text())
    for name, expected in manifest['files'].items():
        if sha256(master.parent / name) != expected:
            raise ValueError(f'master checksum mismatch: {name}')
    identity = dict(config=cfg, master_sha256=sha256(master),
                    script_sha256=sha256(Path(__file__)))
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = args.resume_cache or Path(cfg['cache_dir']).expanduser() / version
    root.mkdir(parents=True, exist_ok=True)
    if args.resume_cache:
        previous = json.loads((root/'provenance.json').read_text())
        if previous['config'] != cfg or previous['master_sha256'] != identity['master_sha256']:
            raise ValueError('resume cache config or master differs from requested acquisition')
        if not (root/'targets.npz').exists():
            raise ValueError('resume cache has no target list')
    else:
        write_json(root / 'provenance.json', identity)
    write_json(root / 'acquisition_implementation.json', dict(identity,
        adapter_sha256=sha256(Path(__file__).resolve().parents[1]/'src/qso_pcolor/qso_acquisition.py'),
        sdss_strategy=cfg.get('sdss_use_object_ids', True)))
    target_path = root / 'targets.npz'
    if not target_path.exists():
        with np.load(master.parent / 'objects.npz') as data:
            objects = {k: data[k] for k in data.files}
        n = len(objects['ra'])
        psf = np.zeros(n, bool)
        clean = np.zeros(n, bool)
        release = np.full(n, -1, int)
        for name, source in manifest['sources'].items():
            path = Path(source['path']).expanduser()
            if sha256(path) != source['sha256']:
                raise ValueError(f'input catalogue checksum mismatch: {name}')
            chosen = objects['preferred_catalogue'] == name
            rows = objects['preferred_input_row'][chosen]
            with np.load(path) as data:
                psf[chosen] = np.char.strip(data['type'][rows].astype(str)) == 'PSF'
                clean[chosen] = data['maskbits'][rows] == 0
                release[chosen] = data['release'][rows]
        l, b = galactic_from_equatorial(objects['ra'], objects['dec'])
        z = objects['zspec']
        keep = np.isfinite(z) & (z > cfg['z_min']) & (z < cfg['z_max'])
        counts = dict(parent_objects=n, redshift_domain=int(keep.sum()))
        keep &= np.abs(b) >= cfg['min_abs_b_deg']
        counts['sky_domain'] = int(keep.sum())
        keep &= psf & np.isin(release, cfg['legacy_releases'])
        counts['psf'] = int(keep.sum())
        keep &= clean
        indices = np.flatnonzero(keep)
        counts['mask_clean'] = len(indices)
        # This ordering improves database locality. Every eligible row is kept.
        cells = hp.ang2pix(cfg['query_order_nside'], objects['ra'][indices],
                          objects['dec'][indices], lonlat=True, nest=True)
        indices = indices[np.argsort(cells, kind='stable')]
        _save_npz(target_path, master_index=indices,
                  **{key: objects[key][indices] for key in
                     ('object_id', 'ra', 'dec', 'zspec', 'redshift_conflict',
                      'extended_duplicate_group', 'preferred_catalogue')})
        write_json(root / 'selection.json', dict(counts=counts,
            status='Photometry acquisition selection only; final association checks and role assignments pending.',
            random_subsampling=False, row_cap=None))
    targets = dict(np.load(target_path))
    n = len(targets['ra'])
    if not n:
        raise ValueError('no eligible targets')
    print(f'{n:,} targets; all retained; cache {root}', flush=True)
    if args.prepare_only:
        return
    for survey in args.surveys or cfg['surveys']:
        started = time.monotonic()
        progress_path = root/f'progress_{survey}.json'
        previous_done = json.loads(progress_path.read_text())['processed'] if progress_path.exists() else 0
        if previous_done == n:
            print(f'{survey}: already complete; retaining all cached rows', flush=True)
            continue
        if previous_done < 0 or previous_done > n or previous_done % cfg['batch_size']:
            raise ValueError('invalid acquisition checkpoint')
        by_id = survey == 'sdss' and cfg.get('sdss_use_object_ids', True)
        id_rows = {}
        position_rows = None
        if by_id:
            id_rows, report = prepare_sdss_id_photometry(root, targets, master, previous_done)
            write_json(root/'sdss_id_links.json', report)
            position_rows = wait_for_sdss_positions(root)
        processed = 0
        band_counts = np.zeros(len(band_labels((survey,))), np.int64)
        for lo in range(0, n, cfg['batch_size']):
            hi = min(n, lo + cfg['batch_size'])
            if lo < previous_done:
                raw = read_acquired_batch(root, survey, targets, lo, hi, cfg['match_radius_arcsec'][survey])
            elif by_id:
                raw = acquire_sdss_batch(root, targets, lo, hi, id_rows, cfg['match_radius_arcsec'][survey],
                                         position_rows=position_rows)
            else:
                raw = match_catalogue(survey, targets['ra'][lo:hi], targets['dec'][lo:hi],
                    root / 'queries', radius_arcsec=cfg['match_radius_arcsec'][survey])
            phot = catalogue_photometry(survey, raw, clean=cfg['clean'],
                                         vhs_bad_bits=cfg['vhs_bad_bits'])
            band_counts += phot.observed.sum(axis=0)
            processed += hi - lo
            if processed < previous_done:
                continue  # never move a checkpoint backwards during replay
            write_json(root / f'progress_{survey}.json', dict(survey=survey,
                processed=processed, total=n, complete=processed == n,
                bands=list(phot.bands), observed_counts=band_counts.tolist(),
                elapsed_seconds=time.monotonic()-started,
                status='Raw nearest-match photometry; ambiguity/hemisphere audit pending. Not a training artifact.'))
            print(f'{survey}: {processed:,}/{n:,} targets cached', flush=True)
        if processed != n:
            raise RuntimeError('batch accounting lost targets')


if __name__ == '__main__':
    main()
