"""Local inventory distinguishes certified nonmatches from unqueried targets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree


def module():
    p=Path(__file__).parents[1]/'scripts/inventory_qso_photometry.py'
    spec=importlib.util.spec_from_file_location('inventory',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_nearby_coordinates_cannot_reuse_a_cached_nonmatch():
    m=module()
    targets=dict(ra=np.array([0.,1.]),dec=np.array([0.,0.]))
    old=dict(ra=np.array([0.,1.+1e-9]),dec=np.array([0.,0.]))
    src,dst=m.exact_target_rows(cKDTree(m.xyz(**targets)),targets,old)
    assert src.tolist()==dst.tolist()==[0]


def test_cone_reuse_requires_whole_aperture_inside_and_handles_ra_wrap():
    m=module()
    tree=cKDTree(m.xyz(np.array([359.99,.0999,.099]),np.zeros(3)))
    assert m.cone_targets(tree,0.,0.,.1,1.).tolist()==[0,2]


def test_inventory_keeps_negative_measurements_and_cached_nonmatches(tmp_path,monkeypatch):
    m=module();out=tmp_path/'inventory';out.mkdir()
    old=tmp_path/'old';(old/'queries').mkdir(parents=True)
    np.savez(tmp_path/'targets.npz',ra=[10.,11.,12.],dec=[0.,0.,0.],object_id=['a','b','c'])
    cfg=dict(match_radius_arcsec={s:2. if s=='allwise' else 1. for s in m.SURVEY_NAMES},clean=True,vhs_bad_bits=0)
    (tmp_path/'provenance.json').write_text(json.dumps(dict(config=cfg)))
    ra=np.array([10.,11.]);dec=np.zeros(2)
    np.savez(old/'targets.npz',ra=ra,dec=dec)
    query='SELECT m.idx, x.* FROM mytmptable m LEFT JOIN LATERAL (SELECT c.ra,c.dec FROM allwise.main c WHERE q3c_join(m.ra,m.dec,c.ra,c.dec,2.0/3600.) AND (TRUE)) x ON TRUE'
    key=hashlib.sha256(query.encode()+ra.tobytes()+dec.tobytes()).hexdigest()[:16]
    path=old/'queries'/f'allwise_{key}.npz'
    rows=dict(idx=np.arange(2),ra=[10.,np.nan],dec=[0.,np.nan],cc_flags=['0000','0000'])
    for b in ('w1','w2','w3','w4'):rows.update({f'value_{b}':[-1.,np.nan],f'error_{b}':[1.,np.nan]})
    np.savez(path,**rows);path.with_suffix('.json').write_text(json.dumps(dict(query=query)))
    (out/'discovered_native.json').write_text(json.dumps(dict(npz_headers_scanned=1,files=[dict(path=str(path),survey='allwise')],errors=[])))
    monkeypatch.setattr(sys,'argv',['inventory','--cache',str(tmp_path),'--inventory',str(out)])
    m.main()
    s=json.loads((out/'summary.json').read_text())['surveys']['allwise']
    assert s['reusable_query_results']==2
    assert s['known_no_counterpart']==1
    assert s['usable_any_band']==1
    assert s['unresolved_targets']==1
    with np.load(out/'allwise_target_inventory.npz') as z:
        assert z['unresolved_target_index'].tolist()==[2]
        assert z['counterpart'].tolist()==[True,False,False]
        assert z['observed'][0].all()


def test_rounded_identifiers_are_not_used_for_catalogue_joins():
    p=Path(__file__).parents[1]/'scripts/inventory_qso_identifier_links.py'
    spec=importlib.util.spec_from_file_location('id_inventory',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.exact_integer_id('454137801351027801')==454137801351027801
    for v in ['4.541378013510278e+17',None,'nan','-1','123.0']:
        assert m.exact_integer_id(v)==-1
