"""Selection and geometry for full-sample preparation; no model fitting."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .background import galactic_healpix
from .data import galactic_from_equatorial
from .legacy import assign_bricks, is_north, random_in_cone


def unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """ICRS degrees to Cartesian unit vectors, including RA wrap."""
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)))


def clean_stellar_rows(rows: dict, known_tree: cKDTree, *, radius_arcsec: float,
                       min_abs_b_deg: float) -> tuple[np.ndarray, dict]:
    """Keep mask-clean PSF rows in their own hemisphere, excluding known QSOs.

    Coordinates are ICRS degrees; the known-quasar tree contains unit vectors.
    Duplicate catalogue keys (release, brickid, objid) count once. No flux,
    magnitude, S/N or band-availability cut is made.
    """
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError('positive finite QSO-removal radius required')
    ra, dec = np.asarray(rows['ra']), np.asarray(rows['dec'])
    valid = np.isfinite(ra + dec) & (np.abs(dec) <= 90)
    keep = valid & (np.char.strip(np.asarray(rows['type']).astype(str)) == 'PSF')
    keep &= np.asarray(rows['maskbits']) == 0
    counts = dict(raw=len(ra), psf_mask_clean=int(keep.sum()))
    b = np.full(len(ra), np.nan)
    _, b[valid] = galactic_from_equatorial(ra[valid], dec[valid])
    north = is_north(ra, dec, b)
    release = np.asarray(rows['release'])
    keep &= np.where(north, release == 9011, np.isin(release, [9010, 9012]))
    keep &= np.abs(b) >= min_abs_b_deg
    counts['own_hemisphere_and_latitude'] = int(keep.sum())
    idx = np.flatnonzero(keep)
    keys = np.rec.fromarrays([np.asarray(rows[k])[idx] for k in ('release', 'brickid', 'objid')])
    _, first = np.unique(keys, return_index=True)
    counts['duplicate_catalogue_keys'] = len(idx) - len(first)
    keep[:] = False; keep[idx[first]] = True
    idx = np.flatnonzero(keep)
    chord = 2*np.sin(np.deg2rad(radius_arcsec/3600)/2)
    distance, _ = known_tree.query(unit_vectors(ra[idx], dec[idx]), distance_upper_bound=chord)
    known = np.isfinite(distance)
    keep[idx[known]] = False
    counts.update(known_qso_removed=int(known.sum()), retained=int(keep.sum()))
    return keep, counts


def spatial_role_manifest(regions: list[dict], historical_test_cells: list[int], *,
                          seed: int, northern_test_fraction: float,
                          selection_fraction: float, calibration_fraction: float) -> dict:
    """Assign disjoint region roles without looking at densities or scores.

    Final test cells use Galactic NESTED HEALPix IDs already carried by the
    region design. Remaining roles use whole cones, allowing independent
    selection and calibration within a fitted sky cell. Preserve at least one
    fitting cone per non-test (hemisphere, cell) stratum.
    """
    if not (0 < northern_test_fraction < 1 and 0 < selection_fraction < 1 and
            0 < calibration_fraction < 1 and selection_fraction+calibration_fraction < 1):
        raise ValueError('invalid spatial partition fractions')
    rng = np.random.default_rng(seed)
    old_test = set(map(int, historical_test_cells))
    north = sorted({int(c['cell']) for c in regions if c['hemisphere'] == 'north'} - old_test)
    if len(north) < 2:
        raise ValueError('need independent northern training and test cells')
    ntest = min(len(north)-1, max(1, round(northern_test_fraction*len(north))))
    added = sorted(map(int, rng.choice(north, ntest, replace=False)))
    test = old_test | set(added)
    by_stratum = {}
    result = {}
    for c in sorted(regions, key=lambda x:x['cone']):
        if int(c['cell']) in test:
            result[int(c['cone'])] = dict(c, role='test')
        else:
            by_stratum.setdefault((c['hemisphere'], int(c['cell'])), []).append(c)
    leftovers = {'north': [], 'south': []}
    for (hemi, _), cones in sorted(by_stratum.items()):
        order = rng.permutation(len(cones))
        for j, i in enumerate(order):
            c = cones[i]
            result[int(c['cone'])] = dict(c, role='fit')
            if j:
                leftovers[hemi].append(int(c['cone']))
    # Fractions of all non-test cones; capacity is limited by preserving fit cones.
    for hemi, extra in leftovers.items():
        extra = list(rng.permutation(sorted(extra)))
        total = sum(c['hemisphere'] == hemi and c['role'] != 'test' for c in result.values())
        ns = min(len(extra), round(selection_fraction*total))
        nc = min(len(extra)-ns, round(calibration_fraction*total))
        for cid in extra[:ns]: result[int(cid)]['role'] = 'select'
        for cid in extra[ns:ns+nc]: result[int(cid)]['role'] = 'calib'
        if ns == 0 or nc == 0:
            raise ValueError(f'insufficient independent {hemi} selection/calibration cones')
    return dict(seed=seed, historical_test_cells=sorted(old_test),
                additional_northern_test_cells=added, test_cells=sorted(test),
                regions=[result[k] for k in sorted(result)])


def assign_object_roles(ra: np.ndarray, dec: np.ndarray, manifest: dict, *,
                        nside: int, fine_nside: int, selection_fraction: float,
                        calibration_fraction: float, radius_deg: float,
                        historical_test_cones: list[dict]) -> np.ndarray:
    """Apply the same sky roles to QSO objects and stellar-region sources.

    Outside the stellar cones, deterministic fine-cell roles reserve QSO
    selection/calibration blocks. Test cells and historical test cones always
    take precedence. Angles/radii are degrees.
    """
    import healpy as hp
    l, b = galactic_from_equatorial(ra, dec)
    fine = galactic_healpix(l, b, fine_nside)
    draws = np.random.default_rng(manifest['seed']).uniform(size=hp.nside2npix(fine_nside))
    roles = np.full(len(ra), 'fit', dtype='<U6')
    roles[draws[fine] < selection_fraction] = 'select'
    roles[(draws[fine] >= selection_fraction) &
          (draws[fine] < selection_fraction+calibration_fraction)] = 'calib'
    tree = cKDTree(unit_vectors(ra, dec))
    chord = 2*np.sin(np.deg2rad(radius_deg)/2)
    for c in manifest['regions']:
        idx = tree.query_ball_point(unit_vectors([c['ra']], [c['dec']])[0], chord)
        roles[idx] = c['role']
    roles[np.isin(galactic_healpix(l, b, nside), manifest['test_cells'])] = 'test'
    for c in historical_test_cones:
        idx = tree.query_ball_point(unit_vectors([c['ra']], [c['dec']])[0],
            2*np.sin(np.deg2rad(c['radius_deg'])/2))
        roles[idx] = 'test'
    return roles


def cone_selection_area(ra: float, dec: float, radius_deg: float, hemisphere: str,
                         bricks: dict, fetch, *, bands: tuple[str, ...],
                         n_points: int, seed: int, min_abs_b_deg: float) -> dict:
    """Mask-clean area with any selected optical exposure; area in square degrees.

    Samples uniform spherical-cap positions. Each exposure map uses its own
    WCS. No source counts or measured fluxes enter this geometric calculation.
    Also report each band's area and intersection with r coverage when present.
    """
    from astropy.wcs import WCS
    if not bands or n_points < 1:
        raise ValueError('exposure bands and positive random count required')
    pr, pd = random_in_cone(ra, dec, radius_deg, n_points, np.random.default_rng(seed))
    _, b = galactic_from_equatorial(pr, pd)
    own = (is_north(pr, pd, b) == (hemisphere == 'north')) & (np.abs(b) >= min_abs_b_deg)
    which = assign_bricks(bricks, pr, pd)
    observed = np.zeros((n_points, len(bands)), bool)
    missing = []

    def sample(image, rr, dd):
        x, y = WCS(image[1]).all_world2pix(rr, dd, 0)
        ix, iy = np.round(x).astype(int), np.round(y).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < image[0].shape[1]) & (iy < image[0].shape[0])
        values = np.full(len(rr), np.nan)
        values[inside] = image[0][iy[inside], ix[inside]]
        return values

    for k in np.unique(which[(which >= 0) & own]):
        name = str(bricks['brickname'][k]); idx = np.flatnonzero((which == k) & own)
        mask = fetch(name, hemisphere, 'maskbits')
        if mask is None:
            missing.append(name + ':maskbits'); continue
        clean = sample(mask, pr[idx], pd[idx]) == 0
        for j, band in enumerate(bands):
            exposure = fetch(name, hemisphere, 'nexp-' + band)
            if exposure is None:
                missing.append(name + ':nexp-' + band); continue
            observed[idx, j] = clean & (sample(exposure, pr[idx], pd[idx]) > 0)
    total_area = 2*np.pi*(1-np.cos(np.deg2rad(radius_deg)))*(180/np.pi)**2
    union = observed.any(axis=1); frac = float(union.mean())
    return dict(area_deg2=total_area*frac, fraction=frac, n_points=n_points, seed=seed,
        area_error_deg2=total_area*np.sqrt(frac*(1-frac)/n_points),
        band_area_deg2={band:float(total_area*observed[:,j].mean()) for j,band in enumerate(bands)},
        no_image_records=missing, geometry='maskbits=0, own hemisphere, latitude domain, any optical exposure',
        external_survey_area_status='not measured by this Legacy geometry calculation')
