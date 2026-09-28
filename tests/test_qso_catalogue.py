import numpy as np
import pytest

from qso_pcolor.qso_catalogue import combine_qso_catalogues


def desi(ra,z=None,ids=None):
    n=len(ra)
    return dict(ra=np.array(ra),dec=np.zeros(n),zspec=np.ones(n) if z is None else np.array(z),
                targetid=np.arange(n)+10 if ids is None else np.array(ids))


def sdss(ra,z=None,warning=None):
    n=len(ra)
    return dict(ra=np.array(ra),dec=np.zeros(n),zspec=np.ones(n) if z is None else np.array(z),
        sdss_name=np.array([f"sdss{i}" for i in range(n)]),
        zwarning=np.zeros(n,int) if warning is None else np.array(warning),is_qso_final=np.ones(n,int))


def merge(d,s):
    return combine_qso_catalogues(d,s,radius_arcsec=1.,redshift_conflict_tolerance=.001)


def test_transitive_positional_groups_keep_every_original_id_and_flag_extent():
    # Endpoints are >1 arcsec apart, connected by the middle member.
    o,m,r=merge(desi([10.,10.+.8/3600,10.+1.6/3600]),sdss([30.]))
    assert r["unique_objects"]==2 and r["input_rows"]==4
    assert sorted(o["n_members"].tolist())==[1,3]
    assert o["extended_duplicate_group"].sum()==1
    assert len(m["source_id"])==4
    assert np.array_equal(np.bincount(m["object_index"]),o["n_members"])


def test_ra_wrap_and_redshift_disagreement_merge_but_do_not_hide_the_conflict():
    o,m,r=merge(desi([359.99995],[1.]),sdss([.00005],[2.]))
    assert r["unique_objects"]==1 and r["both_catalogues"]==1
    assert o["redshift_conflict"].tolist()==[True]
    assert o["preferred_catalogue"].tolist()==["desi"]
    assert sorted(m["zspec"].tolist())==[1.,2.]


def test_invalid_redshifts_retained_and_valid_clean_source_preferred():
    o,m,r=merge(desi([10.],[-999.]),sdss([10.,30.],[1.,-999.]))
    assert r["input_rows"]==3 and r["invalid_adopted_redshifts"]==1
    assert o["preferred_catalogue"].tolist()==["sdss","sdss"]
    assert len(m["source_id"])==3


def test_input_permutations_preserve_object_ids_and_source_row_mapping():
    d=desi([10.,20.,40.],ids=[20,10,30]);s=sdss([10.,50.])
    a,_,_=merge(d,s)
    perm=np.array([2,0,1]);d2={k:v[perm] for k,v in d.items()}
    b,m,_=merge(d2,s)
    for k in ("object_id","ra","dec","zspec","n_members"):
        assert np.array_equal(a[k],b[k])
    use=m["catalogue"]=="desi"
    assert np.array_equal(d2["targetid"][m["input_row"][use]].astype(str),m["source_id"][use])


def test_no_redshift_magnitude_or_object_count_sampling():
    d=desi(np.arange(5000)/100,np.linspace(.01,8.,5000))
    o,m,r=merge(d,sdss([]))
    assert len(o["ra"])==5000 and len(m["ra"])==5000
    assert r["duplicate_memberships"]==0


def test_duplicate_catalogue_identifiers_are_refused_instead_of_silently_overwritten():
    with pytest.raises(ValueError,match="duplicate source IDs"):
        merge(desi([10.,20.],ids=[1,1]),sdss([]))
