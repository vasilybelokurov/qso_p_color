"""Local, memory-mapped training data and replayable full-population batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .multisurvey_data import Photometry, band_labels, catalogue_photometry, SURVEYS
from .qso_acquisition import read_acquired_batch
from .data import galactic_from_equatorial

ROLES = ('fit', 'select', 'calib', 'test')


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def shared_counterparts(ra: np.ndarray, dec: np.ndarray, release: np.ndarray | None = None) -> np.ndarray:
    """Flag exact reused catalogue positions, distinguishing Legacy releases."""
    matched = np.isfinite(ra + dec)
    result = np.zeros(len(ra), bool)
    columns = [ra[matched], dec[matched]]
    if release is not None:
        columns.insert(0, release[matched])
    _, inv, count = np.unique(np.rec.fromarrays(columns), return_inverse=True, return_counts=True)
    result[matched] = count[inv] > 1
    return result


def _new_arrays(path: Path, n: int, bands: tuple[str, ...]) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    specs = dict(flux=('f8', (n, len(bands))), variance=('f8', (n, len(bands))),
                 role=('u1', (n,)), ra=('f8', (n,)), dec=('f8', (n,)),
                 l=('f8', (n,)), b=('f8', (n,)), zspec=('f8', (n,)),
                 field=('i4', (n,)), eligible=('?', (n,)),
                 known_result=('?', (n, 7)), association_ambiguous=('?', (n, 7)),
                 association_count=('i4', (n, 7)))
    arrays = {k: np.lib.format.open_memmap(path / (k+'.npy'), mode='w+', dtype=dtype, shape=shape)
              for k, (dtype, shape) in specs.items()}
    for k, a in arrays.items():
        a[:] = np.nan if k in ('flux', 'zspec') else np.inf if k == 'variance' else -1 if k in ('field', 'association_count') else 0
    return arrays


def _summary(arrays: dict, bands: tuple[str, ...], batch_size: int) -> dict:
    reports = {role: dict(rows=0, eligible=0, bands=[0]*len(bands), negative_fluxes=0,
                         unknown_association_counts=[0]*7, ambiguous=[0]*7) for role in ROLES}
    for lo in range(0, len(arrays['role']), batch_size):
        hi = min(lo+batch_size, len(arrays['role']))
        obs = Photometry(arrays['flux'][lo:hi], arrays['variance'][lo:hi], bands).observed
        arrays['eligible'][lo:hi] = obs.any(axis=1)
        for code, role in enumerate(ROLES):
            use = arrays['role'][lo:hi] == code
            r = reports[role]; r['rows'] += int(use.sum()); r['eligible'] += int(obs[use].any(axis=1).sum())
            r['bands'] = (np.array(r['bands'])+obs[use].sum(axis=0)).tolist()
            r['negative_fluxes'] += int((obs[use] & (arrays['flux'][lo:hi][use] < 0)).sum())
            r['unknown_association_counts'] = (np.array(r['unknown_association_counts'])+
                ((arrays['association_count'][lo:hi][use] < 0) & arrays['known_result'][lo:hi][use]).sum(axis=0)).tolist()
            r['ambiguous'] = (np.array(r['ambiguous'])+arrays['association_ambiguous'][lo:hi][use].sum(axis=0)).tolist()
    return reports


def _saved_counts(root: Path, survey: str, ra: np.ndarray, dec: np.ndarray, radius: float) -> np.ndarray | None:
    # The stopped Legacy count audit mixes releases. Never read it here.
    if survey == 'decals':
        return None
    spec = SURVEYS[survey]
    sql = f"""SELECT m.idx, x.match_count FROM mytmptable m CROSS JOIN LATERAL (
        SELECT count(*) AS match_count FROM {spec.table} c
        WHERE q3c_join(m.ra,m.dec,c.{spec.ra},c.{spec.dec},{radius}/3600.)
          AND ({spec.where})) x"""
    digest = hashlib.sha256(sql.encode()+ra.tobytes()+dec.tobytes()).hexdigest()[:16]
    for path in sorted((root/'association_audit').glob(f'*/counts/{survey}_{digest}.npz')):
        raw = load_npz(path)
        if str(raw['query']) == sql and raw['match_count'].shape == ra.shape:
            return raw['match_count']
    return None


def assemble_qso(root: Path, stellar: Path, out: Path, *, batch_size: int) -> dict:
    """Assemble existing native measurements only; no database fallback."""
    targets = load_npz(root/'targets.npz'); roles = load_npz(stellar/'qso_roles.npz')
    if not all(np.array_equal(targets[k], roles[k]) for k in ('object_id', 'master_index')):
        raise ValueError('frozen QSO role identities differ')
    cfg = json.loads((root/'provenance.json').read_text())['config']
    surveys = tuple(cfg['surveys']); bands = band_labels(surveys); n = len(targets['ra'])
    if tuple(surveys) != tuple(SURVEYS):
        raise ValueError('unexpected survey order')
    arrays = _new_arrays(out, n, bands)
    for k in ('ra', 'dec', 'zspec'):
        arrays[k][:] = targets[k]
    arrays['l'][:], arrays['b'][:] = galactic_from_equatorial(targets['ra'], targets['dec'])
    if not np.isin(roles['role'], ROLES).all():
        raise ValueError('unknown frozen QSO role')
    arrays['role'][:] = np.array([ROLES.index(r) for r in roles['role']], dtype='u1')
    np.save(out/'object_id.npy', targets['object_id']); np.save(out/'master_index.npy', targets['master_index'])
    for k in ('redshift_conflict', 'extended_duplicate_group'):
        np.save(out/(k+'.npy'), targets[k])
    sources = {}; diagnostics = {}
    for si, survey in enumerate(surveys):
        cols = [bands.index(b) for b in band_labels((survey,))]
        counterpart_ra, counterpart_dec = np.full(n, np.nan), np.full(n, np.nan)
        release = np.full(n, -1, int) if survey == 'decals' else None
        reuse_path = root/'gap_acquisition'/survey/'reuse.npz'
        reuse = load_npz(reuse_path) if reuse_path.exists() else None
        if reuse is not None:
            sources[str(reuse_path)] = file_hash(reuse_path)
        def put(idx, raw):
            phot = catalogue_photometry(survey, raw, clean=cfg['clean'], vhs_bad_bits=cfg['vhs_bad_bits'])
            arrays['flux'][np.ix_(idx, cols)] = phot.flux
            arrays['variance'][np.ix_(idx, cols)] = phot.variance
            arrays['known_result'][idx, si] = True
            if 'match_count' in raw and survey != 'decals':
                arrays['association_count'][idx, si] = raw['match_count']
            counterpart_ra[idx], counterpart_dec[idx] = raw['ra'], raw['dec']
            if release is not None:
                release[idx] = raw['release']
        if reuse is not None:
            idx = reuse['target_index']
            if len(np.unique(idx)) != len(idx) or np.any((idx < 0) | (idx >= n)):
                raise ValueError('invalid reusable target indices')
            put(idx, reuse)
        progress = json.loads((root/f'progress_{survey}.json').read_text())
        for lo in range(0, progress['processed'], cfg['batch_size']):
            hi = min(n, lo+cfg['batch_size'])
            raw = read_acquired_batch(root, survey, targets, lo, hi, cfg['match_radius_arcsec'][survey], cache_only=True)
            if 'object_id' in raw and not np.array_equal(raw['object_id'], targets['object_id'][lo:hi]):
                raise ValueError('acquisition identities changed')
            if 'object_id' not in raw and not np.array_equal(raw['idx'], np.arange(hi-lo)):
                raise ValueError('invalid historical positional cache indices')
            put(np.arange(lo, hi), raw)
            saved = _saved_counts(root, survey, targets['ra'][lo:hi], targets['dec'][lo:hi], cfg['match_radius_arcsec'][survey])
            if saved is not None:
                arrays['association_count'][lo:hi, si] = saved
            path = root/'acquired'/f'{survey}_{lo:07d}.npz'
            if path.exists():
                sources[str(path)] = file_hash(path)
        shared = shared_counterparts(counterpart_ra, counterpart_dec, release)
        multiple = arrays['association_count'][:, si] > 1
        bad = shared | multiple
        arrays['association_ambiguous'][:, si] = bad
        arrays['flux'][np.ix_(np.flatnonzero(bad), cols)] = np.nan
        arrays['variance'][np.ix_(np.flatnonzero(bad), cols)] = np.inf
        np.save(out/(survey+'_shared_targets.npy'), np.flatnonzero(shared))
        diagnostics[survey] = dict(known_results=int(arrays['known_result'][:, si].sum()),
            shared_targets=int(shared.sum()), multiple_targets=int(multiple.sum()),
            unknown_counts=int(((arrays['association_count'][:, si] < 0) & arrays['known_result'][:, si]).sum()))
        print('assembled QSO', survey, diagnostics[survey], flush=True)
    report = dict(rows=n, bands=list(bands), roles=_summary(arrays, bands, batch_size),
                  association=diagnostics, sources=sources, no_database_queries=True)
    for a in arrays.values(): a.flush()
    return report


def assemble_stars(root: Path, out: Path, *, batch_size: int) -> dict:
    """Reuse the cleaned stellar photometry and its frozen cone roles."""
    manifest = json.loads((root/'spatial_roles.json').read_text())
    cleaning = json.loads((root/'cleaning_report.json').read_text())
    n = cleaning['totals']['retained']; bands = band_labels()
    arrays = _new_arrays(out, n, bands)
    keys = np.lib.format.open_memmap(out/'catalogue_key.npy', mode='w+', dtype='i8', shape=(n, 3))
    sources = {}; offset = 0
    for region in manifest['regions']:
        path = root/'photometry'/f"cone_{region['cone']:03d}.npz"
        raw = load_npz(path); sources[str(path)] = file_hash(path)
        if tuple(raw['bands']) != bands:
            raise ValueError('stellar band order changed')
        hi = offset+len(raw['ra']); sl = slice(offset, hi)
        for k in ('flux', 'variance', 'ra', 'dec', 'association_ambiguous'):
            arrays[k][sl] = raw[k]
        arrays['role'][sl] = ROLES.index(region['role']); arrays['field'][sl] = region['cone']
        arrays['l'][sl], arrays['b'][sl] = galactic_from_equatorial(raw['ra'], raw['dec'])
        arrays['known_result'][sl] = True
        keys[sl] = np.column_stack([raw[k] for k in ('release', 'brickid', 'objid')])
        offset = hi
    if offset != n or len(np.unique(keys, axis=0)) != n:
        raise ValueError('stellar rows missing or repeated')
    report = dict(rows=n, bands=list(bands), roles=_summary(arrays, bands, batch_size), sources=sources)
    for a in arrays.values(): a.flush()
    keys.flush()
    return report


class TrainingRows:
    """Read-only memory maps in native flux units; covariance created per batch."""
    def __init__(self, root: Path, bands: tuple[str, ...]):
        self.root, self.bands = Path(root), tuple(bands)
        self.arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
                       for p in self.root.glob('*.npy')}

    def select(self, roles: tuple[str, ...], z_range: tuple[float, float] | None = None) -> np.ndarray:
        keep = np.isin(self.arrays['role'], [ROLES.index(r) for r in roles]) & self.arrays['eligible']
        if z_range is not None:
            z = self.arrays['zspec']; keep &= (z >= z_range[0]) & (z < z_range[1])
        return np.flatnonzero(keep)

    def batches(self, indices: np.ndarray, transform, batch_size: int):
        """Yield every selected row once, in fixed order, with unit weight."""
        if batch_size < 1 or len(indices) == 0:
            raise ValueError('positive batch size and nonempty selection required')
        for lo in range(0, len(indices), batch_size):
            rows = indices[lo:lo+batch_size]
            phot = Photometry(self.arrays['flux'][rows], self.arrays['variance'][rows], self.bands)
            features = transform(phot)
            yield features.x, features.cov, features.observed, np.ones(len(rows))
