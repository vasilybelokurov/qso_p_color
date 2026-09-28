"""Assemble Legacy QSO photometry from existing catalogues, without a query."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from .data import _load_npz, _save_npz
from .legacy import BANDS, is_north
from .multisurvey_data import catalogue_photometry
from .sky_acquisition import acquisition_lock, write_record


def file_hash(path: Path) -> str:
    """SHA256 of a source file, read without modifying the source catalogue."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def legacy_rows_from_sources(targets: dict, objects: dict, desi_identity: dict,
                             desi: dict, sdss: dict, *, radius_arcsec: float) -> dict:
    """Return native nanomaggies/ivar in target order using exact source IDs.

    Coordinates are ICRS degrees; separation is arcsec. The DESI cache's WISE
    ``nobs`` values are documented validity proxies (ivar > 0), not exposure
    counts. Preserve that provenance explicitly. Missing IDs, wrong hemisphere
    or an out-of-radius stored counterpart fail instead of triggering a query.
    """
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError('positive finite association radius required')
    master_rows = np.asarray(targets['master_index'])
    if not np.issubdtype(master_rows.dtype, np.integer) or np.any(
            (master_rows < 0) | (master_rows >= len(objects['object_id']))):
        raise ValueError('invalid master row indices')
    if len(np.unique(master_rows)) != len(master_rows):
        raise ValueError('duplicate master target identities')
    if not np.array_equal(targets['object_id'], objects['object_id'][master_rows]):
        raise ValueError('target object identities differ from master')
    channel = objects['preferred_catalogue'][master_rows]
    if not np.isin(channel, ['desi', 'sdss']).all():
        raise ValueError('unsupported source catalogue')
    source_rows = objects['preferred_input_row'][master_rows].copy()
    selected = channel == 'desi'
    if not np.issubdtype(source_rows.dtype, np.integer) or np.any(source_rows < 0) or np.any(
            source_rows >= np.where(selected, len(desi_identity['targetid']), len(sdss['ra']))):
        raise ValueError('invalid source row indices')
    requested_ids = desi_identity['targetid'][source_rows[selected]]
    ids = desi['targetid']
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(requested_ids.dtype, np.integer):
        raise ValueError('DESI identifiers must be exact integers')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate DESI target IDs in local cache')
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], requested_ids)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[order[positions]], requested_ids):
        raise ValueError('local DESI photometry does not cover every requested identity')
    source_rows[selected] = order[positions]
    n = len(master_rows)
    result = dict(idx=np.arange(n), target_index=np.arange(n), master_index=master_rows,
        object_id=np.asarray(targets['object_id']), legacy_source_catalogue=channel,
        legacy_source_row=source_rows, legacy_nobs_wise_is_proxy=selected)
    for ch, source in [('desi', desi), ('sdss', sdss)]:
        use = channel == ch
        rows = source_rows[use]
        mapping = {'ra': 'ra' if ch == 'desi' else 'ls_ra',
                   'dec': 'dec' if ch == 'desi' else 'ls_dec',
                   'release': 'release', 'maskbits': 'maskbits', 'type': 'type'}
        for band in BANDS:
            mapping.update({f'value_{band}': f'flux_{band}', f'error_{band}': f'flux_ivar_{band}',
                            f'nobs_{band}': f'nobs_{band}'})
        for key, source_key in mapping.items():
            values = source[source_key][rows]
            if key not in result:
                result[key] = np.empty(n, dtype=values.dtype)
            else:
                result[key] = result[key].astype(np.promote_types(result[key].dtype, values.dtype), copy=False)
            result[key][use] = values
    if not np.isfinite(result['ra']+result['dec']).all():
        raise ValueError('missing cached Legacy positions')
    result['match_sep_arcsec'] = SkyCoord(targets['ra']*u.deg, targets['dec']*u.deg).separation(
        SkyCoord(result['ra']*u.deg, result['dec']*u.deg)).arcsec
    if np.any(result['match_sep_arcsec'] > radius_arcsec):
        raise ValueError('cached Legacy association outside declared radius')
    north = is_north(targets['ra'], targets['dec'])
    correct = np.where(north, result['release'] == 9011, np.isin(result['release'], [9010, 9012]))
    if not correct.all():
        raise ValueError('cached Legacy photometry has wrong hemisphere')
    if not (np.char.strip(result['type'].astype(str)) == 'PSF').all():
        raise ValueError('cached Legacy morphology differs from PSF selection')
    return result


def compare_legacy_rows(cached: dict, queried: dict) -> dict:
    """Check like-release measurements and final band masks against saved pulls."""
    same = cached['release'] == queried['release']
    equal = np.ones(len(same), bool)
    quality = np.ones(len(same), bool)
    for band in BANDS:
        for key in [f'value_{band}', f'error_{band}']:
            a, b = cached[key], queried[key]
            equal &= (a == b) | (np.isnan(a) & np.isnan(b))
        quality &= (cached[f'nobs_{band}'] > 0) == (queried[f'nobs_{band}'] > 0)
    quality &= (cached['maskbits'] == 0) == (queried['maskbits'] == 0)
    a, b = (catalogue_photometry('decals', rows, clean=True, vhs_bad_bits=0)
            for rows in (cached, queried))
    mask_equal = np.all(a.observed == b.observed, axis=1)
    if np.any(same & (~equal | ~mask_equal)):
        raise ValueError('same-release Legacy measurements or usable-band masks disagree')
    return dict(rows=len(same), equal_fluxes_and_ivars=int(equal.sum()),
        different_release=int((~same).sum()), same_release_quality_flag_differences=int((same & ~quality).sum()),
        same_release_band_mask_differences=int((same & ~mask_equal).sum()))


def assemble_legacy_from_cache(root: Path, *, desi_photometry: Path) -> dict:
    """Verify local sources, save aligned batches, then publish completion.

    Never import a database client or fall back to a positional query. Source
    catalogues and older raw query caches remain unchanged. The existing
    acquisition reader already prefers these target-aligned acquired batches.
    """
    started = time.monotonic()
    cfg = json.loads((root/'provenance.json').read_text())['config']
    master = Path(cfg['master_manifest']).expanduser()
    manifest = json.loads(master.read_text())
    if file_hash(master) != json.loads((root/'provenance.json').read_text())['master_sha256']:
        raise ValueError('master manifest changed')
    sources = {name: Path(info['path']).expanduser() for name, info in manifest['sources'].items()}
    for name, path in sources.items():
        if file_hash(path) != manifest['sources'][name]['sha256']:
            raise ValueError(f'master input changed: {name}')
    for name, expected in manifest['files'].items():
        if file_hash(master.parent/name) != expected:
            raise ValueError(f'master output changed: {name}')
    inputs = dict(targets=root/'targets.npz', master=master, objects=master.parent/'objects.npz',
        desi_identity=sources['desi'], desi_photometry=desi_photometry, sdss=sources['sdss'],
        desi_provenance=desi_photometry.with_suffix('.json'))
    provenance = {k: dict(path=str(p), sha256=file_hash(p)) for k, p in inputs.items()}
    identity = dict(inputs=provenance, implementation=file_hash(Path(__file__)), config=cfg)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = root/'legacy_local'/version
    out.mkdir(parents=True, exist_ok=True)
    with acquisition_lock(root/'legacy_local'/'worker.lock'):
        targets = _load_npz(inputs['targets'])
        raw = legacy_rows_from_sources(targets, _load_npz(inputs['objects']),
            _load_npz(inputs['desi_identity']), _load_npz(desi_photometry), _load_npz(inputs['sdss']),
            radius_arcsec=cfg['match_radius_arcsec']['decals'])
        n, size = len(raw['idx']), cfg['batch_size']
        comparisons = []
        # Compare only exact target/query caches already on disk. No rematching.
        for meta_path in sorted((root/'queries').glob('decals_*.json')):
            meta = json.loads(meta_path.read_text())
            for lo in range(0, n, size):
                hi = min(n, lo+size)
                digest = hashlib.sha256(meta['query'].encode()+targets['ra'][lo:hi].tobytes()+
                                        targets['dec'][lo:hi].tobytes()).hexdigest()[:16]
                if meta_path.stem == 'decals_'+digest:
                    comparisons.append(dict(start=lo, **compare_legacy_rows(
                        {k:v[lo:hi] for k,v in raw.items()}, _load_npz(meta_path.with_suffix('.npz')))))
                    break
        phot = catalogue_photometry('decals', raw, clean=cfg['clean'], vhs_bad_bits=cfg['vhs_bad_bits'])
        report = dict(identity=identity, targets=n, source_counts={ch:int((raw['legacy_source_catalogue']==ch).sum())
            for ch in ['desi','sdss']}, comparisons=comparisons,
            max_separation_arcsec=float(raw['match_sep_arcsec'].max()), wrong_hemisphere=0,
            bands=list(phot.bands), observed_counts=phot.observed.sum(axis=0).tolist(),
            negative_observed_fluxes=int((phot.observed & (phot.flux < 0)).sum()),
            desi_wise_nobs='Validity proxy from positive inverse variance; not an exposure count. Actual band masks also require positive finite inverse variance.',
            new_database_queries=0, association_audit_complete=False, model_fitted=False)
        _save_npz(out/'photometry.npz', object_id=raw['object_id'], master_index=raw['master_index'],
            target_index=raw['target_index'], flux=phot.flux, variance=phot.variance,
            observed=phot.observed, bands=np.array(phot.bands))
        saved = _load_npz(out/'photometry.npz')
        for key, expected in dict(object_id=raw['object_id'], master_index=raw['master_index'],
                target_index=raw['target_index'], flux=phot.flux, variance=phot.variance,
                observed=phot.observed, bands=np.array(phot.bands)).items():
            if not np.array_equal(saved[key], expected,
                                  equal_nan=np.issubdtype(expected.dtype, np.inexact)):
                raise ValueError(f'saved Legacy photometry differs: {key}')
        del saved
        batches = {}
        for lo in range(0, n, size):
            hi = min(n, lo+size)
            values = {k:v[lo:hi] for k,v in raw.items()}
            values['idx'] = np.arange(hi-lo)
            path = root/'acquired'/f'decals_{lo:07d}.npz'
            if path.exists():
                old = _load_npz(path)
                if set(old) != set(values) or any(not np.array_equal(old[k],v,
                    equal_nan=np.issubdtype(v.dtype, np.inexact)) for k,v in values.items()):
                    raise ValueError('existing acquired Legacy batch differs; refusing overwrite')
            else:
                _save_npz(path, **values)
            verified = _load_npz(path)
            if not np.array_equal(verified['object_id'], targets['object_id'][lo:hi]):
                raise ValueError('saved Legacy target identities differ')
            check = catalogue_photometry('decals', verified, clean=cfg['clean'], vhs_bad_bits=cfg['vhs_bad_bits'])
            if not np.array_equal(check.observed, phot.observed[lo:hi]):
                raise ValueError('saved Legacy band masks differ')
            batches[str(path)] = file_hash(path)
            print(f'Legacy local assembly: {hi:,}/{n:,} saved and verified', flush=True)
        report.update(elapsed_seconds=time.monotonic()-started, batch_sha256=batches,
                      photometry_sha256=file_hash(out/'photometry.npz'))
        write_record(out/'report.json', report)
        write_record(root/'legacy_local_sources.json', dict(directory=str(out), report=str(out/'report.json')))
        write_record(root/'progress_decals.json', dict(survey='decals', processed=n,total=n,complete=True,
            bands=report['bands'],observed_counts=report['observed_counts'],elapsed_seconds=report['elapsed_seconds'],
            status='Locally assembled source photometry verified; independent association audit pending. Not a training artifact.'))
        return report
