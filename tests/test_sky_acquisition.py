"""Acquisition restart and sky boundaries must not change row identities."""
import json

import healpy as hp
import numpy as np
import pytest

from qso_pcolor.sky_acquisition import sky_blocks, fetch_sky_blocks, acquisition_lock


CONFIG = dict(max_targets=3, parent_nside=1, order_nside=1024)


def test_partition_covers_wrap_poles_duplicates_and_parent_boundaries():
    ra = np.array([359.999, 0., .001, 30., 180., 20., 20., 50., 190.])
    dec = np.array([0., 0., 0., 90., -90., 20., 20., 30., -50.])
    blocks = sky_blocks(ra, dec, **CONFIG)
    np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), np.arange(len(ra)))
    for rows in blocks:
        assert 0 < len(rows) <= 3
        assert len(np.unique(hp.ang2pix(1, ra[rows], dec[rows], nest=True, lonlat=True))) == 1
    assert sky_blocks(np.array([]), np.array([]), **CONFIG) == []
    with pytest.raises(ValueError):
        sky_blocks([0], [91], **CONFIG)
    with pytest.raises(ValueError):
        sky_blocks([0], [0], **dict(CONFIG, max_targets=0))


def test_interrupted_run_reuses_completed_blocks_and_restores_original_order(tmp_path):
    # Every point belongs to one parent; incoming order deliberately reversed.
    inputs = dict(ra=np.arange(10., 20.)[::-1], dec=np.zeros(10),
                  objid=np.arange(1237662962244125491, 1237662962244125501, dtype=np.int64))
    calls = []
    fail = [True]
    def fetch(part, out):
        calls.append(part['objid'].copy())
        if len(calls) == 2 and fail[0]:
            raise ConnectionError('simulated interruption')
        rows = np.arange(len(part['ra']))[::-1]
        result = dict(idx=rows, ra=part['ra'][rows].copy(), dec=part['dec'][rows],
                      flux=-part['ra'][rows], objid=part['objid'][rows])
        result['ra'][result['objid'] == inputs['objid'][5]] = np.nan
        return result
    args = dict(identity={'query': 'ID join'}, config=CONFIG, fetch=fetch)
    with pytest.raises(ConnectionError):
        fetch_sky_blocks(inputs, tmp_path, **args)
    status = json.loads(next(tmp_path.glob('*/status.json')).read_text())
    assert status['stage'] == 'failed' and status['completed_targets'] == 3
    fail[0] = False
    result = fetch_sky_blocks(inputs, tmp_path, **args)
    assert len(calls) == 5  # first saved, failed second, three remaining
    np.testing.assert_array_equal(result['objid'], inputs['objid'])
    np.testing.assert_array_equal(result['flux'], -inputs['ra'])
    assert np.isnan(result['ra'][5])
    fetch_sky_blocks(inputs, tmp_path, **args)
    assert len(calls) == 5
    # Changed SQL cannot reuse previous measurements.
    fetch_sky_blocks(inputs, tmp_path, **dict(args, identity={'query': 'different'}))
    assert len(calls) == 9


def test_duplicate_or_missing_rows_never_get_checkpointed(tmp_path):
    def fetch(part, out):
        return dict(idx=np.array([0, 0]), ra=np.zeros(2), dec=np.zeros(2))
    with pytest.raises(ValueError, match='lost or duplicated'):
        fetch_sky_blocks(dict(ra=np.array([10., 11.]), dec=np.zeros(2)), tmp_path,
            identity={'query': 'bad'}, config=CONFIG, fetch=fetch)
    assert not list(tmp_path.glob('*/block_*.npz'))


def test_concurrent_worker_refused(tmp_path):
    with acquisition_lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError, match='already holds'):
            with acquisition_lock(tmp_path/'lock'):
                pass


def test_id_whole_result_is_reused_when_blocks_enabled(tmp_path, monkeypatch):
    from qso_pcolor import qso_acquisition as acq
    import sqlutilpy
    class Conn:
        def rollback(self): pass
        def close(self): pass
    monkeypatch.setattr(sqlutilpy, 'getConnection', lambda **kw: Conn())
    monkeypatch.setattr(sqlutilpy, 'upload', lambda *a, **kw: None)
    calls = []
    def get(query, **kw):
        calls.append(query)
        if query.startswith('EXPLAIN'):
            return dict(plan=np.array(['Index Scan using photoobjall_objid_idx']))
        return dict(idx=np.array([0]), ra=np.array([10.]), dec=np.array([0.]))
    monkeypatch.setattr(sqlutilpy, 'get', get)
    acq.fetch_sdss_by_id(np.array([10.]), np.array([0.]), np.array([123]), tmp_path)
    acq.fetch_sdss_by_id(np.array([10.]), np.array([0.]), np.array([123]), tmp_path, block_config=CONFIG)
    assert len(calls) == 2
