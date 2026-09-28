"""Catalogue identity reuse must preserve selection, missing rows and caches."""
import numpy as np
import pytest

from qso_pcolor import qso_acquisition as acq


def test_ids_use_all_memberships_without_float_roundoff():
    big = 1237662962244125491
    catalogue = dict(sdss_name=np.array(['a','b','c','d','e']),
                     objid=np.array([str(big),str(big+1),'0','bad',str(big+2)]))
    members = dict(catalogue=np.array(['desi','sdss','sdss','sdss','sdss','sdss']),
                   source_id=np.array(['101','a','b','c','d','e']),
                   object_index=np.array([0,0,1,2,3,1]))
    ids, conflict = acq.sdss_photometric_ids(members,np.arange(5),catalogue)
    assert ids.tolist() == [big,-1,-1,-1,-1]
    assert conflict.tolist() == [False,True,False,False,False]
    with pytest.raises(ValueError,match='missing'):
        acq.sdss_photometric_ids(members,np.arange(5),
                                {k:v[1:] for k,v in catalogue.items()})


def raw_rows(ra):
    ra = np.asarray(ra,float)
    raw = dict(idx=np.arange(len(ra)),ra=ra,dec=np.zeros(len(ra)),
               clean=np.ones(len(ra),int),match_sep_arcsec=np.full(len(ra),.2))
    for band in 'ugriz':
        raw[f'value_{band}'] = -ra  # negative measurements must survive
        raw[f'error_{band}'] = np.ones(len(ra))
    return raw


def test_hybrid_fallback_order_negative_flux_and_no_repeat_query(tmp_path,monkeypatch):
    targets = dict(ra=np.arange(10.,16.),dec=np.zeros(6))
    ids = raw_rows([15,12,13,11])
    ids.update(target_index=np.array([5,2,3,1]),
               sdss_photometric_objid=np.array([105,102,-1,101]))
    ids['match_sep_arcsec'][1] = 2.  # ID points outside the existing radius
    calls = []
    def match(name,ra,dec,cache,**kw):
        calls.append(ra.tolist())
        return raw_rows(ra)
    monkeypatch.setattr(acq,'match_catalogue',match)
    result = acq.acquire_sdss_batch(tmp_path,targets,0,6,ids,1.)
    assert calls == [[10.,12.,13.,14.]]
    np.testing.assert_array_equal(result['ra'],targets['ra'])
    np.testing.assert_array_equal(result['value_u'],-targets['ra'])
    assert result['sdss_id_link_used'].tolist() == [False,True,False,False,False,True]
    assert result['sdss_photometric_objid'].tolist() == [-1,101,-1,-1,-1,105]
    reread = acq.read_acquired_batch(tmp_path,'sdss',targets,0,6,1.)
    repeated = acq.acquire_sdss_batch(tmp_path,targets,0,6,ids,1.)
    for key in result:
        np.testing.assert_array_equal(reread[key],result[key])
        np.testing.assert_array_equal(repeated[key],result[key])
    assert len(calls) == 1


def test_id_query_uses_exact_identifiers_primary_selection_and_restores_rows(tmp_path,monkeypatch):
    import sqlutilpy
    class Connection:
        def rollback(self): pass
        def close(self): pass
    monkeypatch.setattr(sqlutilpy,'getConnection',lambda **kw:Connection())
    uploaded = []
    monkeypatch.setattr(sqlutilpy,'upload',lambda *a,**kw:uploaded.append((a,kw)))
    raw = raw_rows([20.,10.])
    raw['idx'] = np.array([1,0])
    raw['sdss_photometric_objid'] = np.array([-1,1237662962244125491])
    queries = []
    def get(query,**kw):
        queries.append(query)
        if query.startswith('EXPLAIN'):
            return {'plan':np.array(['Index Scan using photoobjall_objid_idx on photoobjall c'])}
        return raw
    monkeypatch.setattr(sqlutilpy,'get',get)
    ids = np.array([1237662962244125491,1237662962244125492],np.int64)
    result = acq.fetch_sdss_by_id(np.array([10.,20.]),np.zeros(2),ids,tmp_path)
    assert 'c.objid=m.objid AND c.mode=1' in queries[1]
    assert 'q3c_join' not in queries[1]
    assert uploaded[0][1]['analyze']
    np.testing.assert_array_equal(uploaded[0][0][1][3],ids)
    assert result['sdss_photometric_objid'].tolist() == [1237662962244125491,-1]
    acq.fetch_sdss_by_id(np.array([10.,20.]),np.zeros(2),ids,tmp_path)
    assert len(queries) == 2


def test_association_counts_reused_across_auditor_versions(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import sqlutilpy
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('audit_for_test',scripts/'audit_qso_associations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, new = tmp_path/'old'/'counts', tmp_path/'new'/'counts'
    old.mkdir(parents=True); new.mkdir(parents=True)
    calls = []
    def join(*args,**kw):
        calls.append(args)
        return dict(idx=np.array([1,0]),match_count=np.array([2,0]))
    monkeypatch.setattr(sqlutilpy,'local_join',join)
    ra,dec = np.array([10.,20.]),np.zeros(2)
    first = module.match_counts('sdss',ra,dec,1.,old)
    second = module.match_counts('sdss',ra,dec,1.,new)
    assert first.tolist() == second.tolist() == [0,2]
    assert len(calls) == 1


def test_independent_download_reused_including_unmatched_targets(tmp_path,monkeypatch):
    targets = dict(ra=np.arange(10.,16.),dec=np.zeros(6))
    ids = raw_rows([11.,12.])
    ids.update(target_index=np.array([1,2]),sdss_photometric_objid=np.array([101,102]))
    ids['match_sep_arcsec'][1] = 2.  # failed ID requires a later positional lookup
    saved = raw_rows([10.,13.,14.,15.])
    saved['target_index'] = np.array([0,3,4,5])
    saved['ra'][1] = np.nan  # already searched, with no counterpart; do not requery
    calls = []
    def match(name,ra,dec,cache,**kw):
        calls.append(ra.tolist())
        return raw_rows(ra)
    monkeypatch.setattr(acq,'match_catalogue',match)
    raw = acq.acquire_sdss_batch(tmp_path,targets,0,6,ids,1.,position_rows=saved)
    assert calls == [[12.]]
    assert np.isnan(raw['ra'][3])
    assert raw['sdss_id_link_used'].tolist() == [False,True,False,False,False,False]
    np.testing.assert_array_equal(raw['value_u'],-targets['ra'])
