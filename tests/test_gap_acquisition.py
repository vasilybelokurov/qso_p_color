"""No repeat download for cached results, including cached non-detections."""
import json
import numpy as np
import pytest
from qso_pcolor import gap_acquisition as gap
from qso_pcolor.data import _save_npz, _load_npz


def test_simple_join_reduction_preserves_nulls_negatives_and_multiple_counts():
    raw=dict(idx=np.array([1,0,1,2]),ra=np.array([3.,np.nan,4.,5.]),
        dec=np.array([0.,np.nan,0.,0.]),match_sep_arcsec=np.array([.8,np.nan,.2,.1]),
        value_g=np.array([8.,np.nan,-2.,1.]))
    r=gap.nearest_rows(raw,3,spatial_counts=True)
    assert r['idx'].tolist()==[0,1,2]
    assert r['match_count'].tolist()==[0,2,1]
    assert np.isnan(r['ra'][0]) and r['value_g'][1]==-2
    r=gap.nearest_rows(raw,4,spatial_counts=True)
    assert np.isnan(r['ra'][3]) and r['match_count'][3]==0


def test_query_uses_correct_q3c_order_and_exact_ps1_id_join():
    q=gap.query_sql('skymapper',1.)
    assert 'q3c_join(m.ra,m.dec,c.raj2000,c.dej2000,1.0/3600.)' in q
    assert 'JOIN skymapper_dr4.main' in q and 'LEFT JOIN' not in q and 'LATERAL' not in q
    q=gap.query_sql('ps1',1.,by_id=True)
    assert 'c.objid=m.objid' in q and 'primarydetection=1' in q and 'q3c_join' not in q
    positional=gap.query_sql('ps1',1.)
    assert 'CROSS JOIN LATERAL' in positional and 'OFFSET 0' in positional
    with pytest.raises(ValueError):gap.query_sql('allwise',2.,by_id=True)


def test_cached_nonmatches_not_queried_and_bad_id_gets_positional_fallback(tmp_path,monkeypatch):
    n=5;targets=dict(ra=np.arange(n,dtype=float)+10,dec=np.zeros(n),object_id=np.array(list('abcde')))
    out=tmp_path/'gap_acquisition/ps1';out.mkdir(parents=True)
    inv=tmp_path/'inventory';inv.mkdir()
    links=inv/'ps1_identifier_candidates.npz'
    _save_npz(links,target_index=np.array([1,2]),objid=np.array([2**55+1,2**55+3]),within_radius=np.ones(2,bool))
    (inv/'summary.json').write_text(json.dumps(dict(identifier_links=dict(candidate_file_sha256={links.name:gap.file_hash(links)}))))
    def raw(ra,kind):
        count=len(ra);r=dict(idx=np.arange(count),ra=np.array(ra),dec=np.zeros(count),
            match_sep_arcsec=np.full(count,.1),match_count=np.full(count,1),lookup_kind=np.full(count,kind))
        for b in 'grizy':r.update({f'value_{b}':np.full(count,20.),f'error_{b}':np.ones(count),f'{b}nframes':np.ones(count,int)})
        return r
    reused=raw([np.nan,13.],0);reused['dec'][0]=np.nan;reused['target_index']=np.array([0,3])
    monkeypatch.setattr(gap,'prepare_reuse',lambda *a,**kw:reused)
    calls=[]
    def fetch(survey,inputs,cache,*,radius,by_id=False):
        calls.append((by_id,inputs['ra'].tolist()))
        r=raw(inputs['ra'],2 if by_id else 1)
        if by_id:r['match_sep_arcsec'][inputs['ra']==11.]=2. # failed ID association
        return r
    monkeypatch.setattr(gap,'fetch_native',fetch)
    cfg=dict(batch_size=5,match_radius_arcsec={'ps1':1.},clean=True,vhs_bad_bits=0)
    gap.acquire_survey('ps1',tmp_path,inv,targets,cfg,query_size=2)
    assert calls==[(True,[11.,12.]),(False,[11.]),(False,[14.])]
    saved=_load_npz(tmp_path/'acquired/ps1_0000000.npz')
    assert saved['object_id'].tolist()==list('abcde')
    assert saved['lookup_kind'].tolist()==[0,1,2,0,1]
    assert np.isnan(saved['ra'][0])
    calls.clear()
    gap.acquire_survey('ps1',tmp_path,inv,targets,cfg,query_size=2)
    assert calls==[]


def test_auditor_queries_only_unknown_counts(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys,types
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('gap_audit',scripts/'audit_qso_associations.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    calls=[]
    def query(sql,name,arrays,names,**kw):
        calls.append(arrays[1].tolist())
        return dict(idx=np.arange(len(arrays[1])),match_count=np.full(len(arrays[1]),2))
    monkeypatch.setitem(sys.modules,'sqlutilpy',types.SimpleNamespace(local_join=query))
    ra=np.array([10.,11.,12.]);dec=np.zeros(3)
    counts=m.match_counts('ps1',ra,dec,1.,tmp_path,known_counts=np.array([0,-1,3]))
    assert counts.tolist()==[0,2,3]
    assert calls==[[11.]]
    again=m.match_counts('ps1',ra,dec,1.,tmp_path)
    assert np.array_equal(again,counts) and calls==[[11.]]


def test_empty_inner_join_restores_every_target_as_a_nonmatch():
    raw={k:np.array([],float) for k in ['idx','ra','dec','match_sep_arcsec','value_g']}
    result=gap.nearest_rows(raw,7,spatial_counts=True)
    assert np.array_equal(result['idx'],np.arange(7))
    assert np.all(result['match_count']==0)
    assert np.isnan(result['ra']).all() and np.isnan(result['value_g']).all()
