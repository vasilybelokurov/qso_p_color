"""Local reuse must preserve identities, photometric systems and missing bands."""
import numpy as np
import pytest

from qso_pcolor.legacy import BANDS
from qso_pcolor.legacy_acquisition import legacy_rows_from_sources, compare_legacy_rows
from qso_pcolor.multisurvey_data import catalogue_photometry


def sources():
    ids = np.array([2**55 + 3, 2**55 + 1], dtype=np.int64)
    objects = dict(object_id=np.array(['d0', 's0', 'd1']),
        preferred_catalogue=np.array(['desi', 'sdss', 'desi']),
        preferred_input_row=np.array([0, 0, 1]))
    order = np.array([2, 0, 1])
    targets = dict(master_index=order, object_id=objects['object_id'][order],
        ra=np.array([180., 180., 160.]), dec=np.array([40., 0., 0.]))
    desi = dict(targetid=ids[::-1].copy(), ra=np.array([180., 180.]), dec=np.array([40., 0.]),
        release=np.array([9011, 9010]), maskbits=np.zeros(2, int), type=np.array(['PSF', 'PSF']))
    sdss = dict(ra=np.array([160.]), dec=np.array([0.]), ls_ra=np.array([160.+.1/3600]),
        ls_dec=np.array([0.]), release=np.array([9012]), maskbits=np.zeros(1, int), type=np.array(['PSF']))
    for band in BANDS:
        desi.update({f'flux_{band}': np.array([-3., 2.]), f'flux_ivar_{band}':np.ones(2),
                     f'nobs_{band}':np.ones(2, int)})
        sdss.update({f'flux_{band}':np.array([4.]), f'flux_ivar_{band}':np.ones(1),
                     f'nobs_{band}':np.ones(1, int)})
    return targets, objects, dict(targetid=ids), desi, sdss


def test_exact_ids_and_hemispheres_preserve_negative_flux_and_sdss_counterpart_position():
    inputs = sources()
    rows = legacy_rows_from_sources(*inputs, radius_arcsec=1.)
    assert rows['legacy_source_row'].tolist() == [0, 1, 0]
    assert rows['value_g'].tolist() == [-3., 2., 4.]
    assert rows['ra'][2] == inputs[-1]['ls_ra'][0]
    assert rows['match_sep_arcsec'][2] == pytest.approx(.1)
    assert rows['legacy_nobs_wise_is_proxy'].tolist() == [True, True, False]
    phot = catalogue_photometry('decals', rows, clean=True, vhs_bad_bits=0)
    assert phot.observed.sum(axis=1).tolist() == [5, 5, 5]
    assert np.all(phot.flux[0, 5:] == -3.)
    assert not phot.observed[0, :5].any()
    assert not phot.observed[1:, 5:].any()


@pytest.mark.parametrize('failure', ['missing_id', 'wrong_hemisphere', 'negative_row', 'wrong_identity'])
def test_invalid_sources_fail_without_query_fallback(failure):
    inputs = sources()
    if failure == 'missing_id':
        inputs[3]['targetid'][0] += 100
    elif failure == 'wrong_hemisphere':
        inputs[3]['release'][0] = 9010
    elif failure == 'negative_row':
        inputs[1]['preferred_input_row'][0] = -1
    else:
        inputs[0]['object_id'][0] = 'bad'
    with pytest.raises(ValueError):
        legacy_rows_from_sources(*inputs, radius_arcsec=1.)


def test_wise_nobs_proxy_difference_is_harmless_only_when_final_masks_agree():
    rows = legacy_rows_from_sources(*sources(), radius_arcsec=1.)
    for band in ('w1', 'w2'):
        rows[f'error_{band}'][0] = 0
        rows[f'nobs_{band}'][0] = 0
    query = {k:v.copy() for k,v in rows.items()}
    for band in ('w1', 'w2'):
        query[f'nobs_{band}'][0] = 180
    report = compare_legacy_rows(rows, query)
    assert report['same_release_quality_flag_differences'] == 1
    assert report['same_release_band_mask_differences'] == 0
    rows['error_w1'][0] = query['error_w1'][0] = 1
    with pytest.raises(ValueError, match='masks disagree'):
        compare_legacy_rows(rows, query)


def test_same_release_flux_difference_is_rejected():
    rows = legacy_rows_from_sources(*sources(), radius_arcsec=1.)
    query = {k:v.copy() for k,v in rows.items()}
    query['value_g'][0] += 1
    with pytest.raises(ValueError, match='measurements'):
        compare_legacy_rows(rows, query)
    query['release'][0] = 9010
    assert compare_legacy_rows(rows, query)['different_release'] == 1
