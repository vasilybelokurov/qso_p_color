import numpy as np
from scipy.spatial import cKDTree

from qso_pcolor.preparation import (unit_vectors,clean_stellar_rows,spatial_role_manifest,
                                    assign_object_roles,cone_selection_area)


def test_cleaning_preserves_negative_flux_and_removes_alias_qso_and_duplicate_ids():
    # Southern hemisphere and a known QSO across the RA=0 boundary.
    rows=dict(ra=np.array([359.99995,20.,20.,40.,50.,60.]),dec=np.full(6,-30.),
        type=np.array(['PSF']*5+['EXP']),maskbits=np.array([0,0,0,0,1,0]),
        release=np.array([9010,9010,9010,9011,9010,9010]),
        brickid=np.ones(6,int),objid=np.array([1,2,2,3,4,5]),flux_r=np.full(6,-10.))
    known=cKDTree(unit_vectors(np.array([.00005]),np.array([-30.])))
    keep,count=clean_stellar_rows(rows,known,radius_arcsec=1,min_abs_b_deg=0)
    assert np.flatnonzero(keep).tolist()==[1]
    assert count['known_qso_removed']==1 and count['duplicate_catalogue_keys']==1
    assert rows['flux_r'][keep][0]<0


def test_spatial_roles_retain_historical_tests_add_north_and_leave_fit_in_each_cell():
    regions=[]
    for hemi,base in [('north',0),('south',50)]:
        for cell in range(base,base+10):
            for _ in range(3): regions.append(dict(cone=len(regions),hemisphere=hemi,cell=cell))
    kwargs=dict(seed=123,northern_test_fraction=.2,selection_fraction=.15,calibration_fraction=.2)
    a=spatial_role_manifest(regions,[50],**kwargs)
    b=spatial_role_manifest(regions[::-1],[50],**kwargs)
    assert a==b
    assert 50 in a['test_cells'] and len(a['additional_northern_test_cells'])==2
    for hemi in ('north','south'):
        assert {c['role'] for c in a['regions'] if c['hemisphere']==hemi}=={'fit','select','calib','test'}
    for cell in {c['cell'] for c in regions}-set(a['test_cells']):
        assert any(c['cell']==cell and c['role']=='fit' for c in a['regions'])


def test_historical_test_cones_override_any_new_role():
    manifest=dict(seed=1,test_cells=[],regions=[dict(ra=150.,dec=40.,role='calib')])
    roles=assign_object_roles(np.array([150.,150.01]),np.array([40.,40.]),manifest,
        nside=4,fine_nside=16,selection_fraction=.15,calibration_fraction=.2,radius_deg=.3,
        historical_test_cones=[dict(ra=150.,dec=40.,radius_deg=.15)])
    assert roles.tolist()==['test','test']


def test_area_uses_any_optical_band_instead_of_requiring_r():
    from astropy.wcs import WCS
    w=WCS(naxis=2);w.wcs.crpix=[50,50];w.wcs.cdelt=[-.01,.01]
    w.wcs.crval=[150.,-30.];w.wcs.ctype=['RA---TAN','DEC--TAN']
    header=w.to_header()
    bricks=dict(ra1=np.array([149.]),ra2=np.array([151.]),
                dec1=np.array([-31.]),dec2=np.array([-29.]),brickname=np.array(['mock']))
    def fetch(brick,hemisphere,kind):
        # No r or z exposure; valid mask-clean g observations everywhere.
        return np.full((100,100),1 if kind=='nexp-g' else 0),header
    area=cone_selection_area(150.,-30.,.1,'south',bricks,fetch,bands=('g','r','z'),
                            n_points=1000,seed=3,min_abs_b_deg=0)
    expected=2*np.pi*(1-np.cos(np.deg2rad(.1)))*(180/np.pi)**2
    assert area['area_deg2']==expected
    assert area['band_area_deg2']['g']==expected and area['band_area_deg2']['r']==0


def test_match_counts_have_a_separate_cache_and_keep_unmatched_rows(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from qso_pcolor.multisurvey_data import match_catalogue
    queries=[]
    def join(query,table,data,names,**kwargs):
        queries.append(query)
        result=dict(idx=np.array([1,0]),ra=np.array([np.nan,10.]),dec=np.array([np.nan,0.]))
        if 'count(*) OVER ()' in query: result['match_count']=np.array([-1,2])
        return result
    monkeypatch.setitem(sys.modules,'sqlutilpy',SimpleNamespace(local_join=join))
    args=('sdss',np.array([10.,20.]),np.array([0.,0.]),tmp_path)
    match_catalogue(*args,radius_arcsec=1)
    result=match_catalogue(*args,radius_arcsec=1,include_match_count=True)
    assert result['match_count'].tolist()==[2,-1] and np.isnan(result['ra'][1])
    match_catalogue(*args,radius_arcsec=1,include_match_count=True)
    assert len(queries)==2


def test_qso_association_audit_preserves_zero_counts_and_detects_global_sharing(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('qso_association_audit',scripts/'audit_qso_associations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    queries=[]
    def join(query,*args,**kwargs):
        queries.append(query)
        return dict(idx=np.array([2,0,1]),match_count=np.array([2,1,0]))
    monkeypatch.setitem(sys.modules,'sqlutilpy',SimpleNamespace(local_join=join))
    counts=module.match_counts('sdss',np.array([10.,20.,30.]),np.zeros(3),1.,tmp_path)
    assert counts.tolist()==[1,0,2]
    again=module.match_counts('sdss',np.array([10.,20.,30.]),np.zeros(3),1.,tmp_path)
    assert np.array_equal(counts,again) and len(queries)==1
    assert module.shared_source_mask(np.array([10.,np.nan,20.,10.]),np.array([0.,np.nan,0.,0.])).tolist()==[True,False,False,True]
