"""Protect completion role separation, count normalization and abundance reuse."""
import numpy as np
import pytest
from qso_pcolor.full_population import population_rows, merge_empty_counts, count_prior, rebind_qso_prior


def test_roles_use_every_eligible_row_without_overlap():
    d=dict(eligible=np.array([True,True,True,True,False]),role=np.array([0,1,2,3,0]))
    np.testing.assert_array_equal(population_rows(d,('fit','select')),[0,1])
    np.testing.assert_array_equal(population_rows(d,('calib',)),[2])
    with pytest.raises(ValueError):population_rows(d,('bogus',))


def test_empty_bin_merging_conserves_counts_and_area_rate():
    counts=np.array([[0,4,0,3],[0,2,0,1]])
    merged,edges=merge_empty_counts(counts,np.arange(5.))
    assert merged.sum()==10
    assert np.all(merged.sum(axis=0)>0)
    p=count_prior(counts,np.array([1.,3.]),np.array([2,2]),np.arange(5.),
                  nside=4,nside_parent=2,n0=100,meta={})
    assert p.area=={2:4.}
    assert sum(p.counts.values())==10
    assert np.isclose(np.sum(p.global_density*np.diff(p.mag_edges))*4,10)


def test_count_without_area_fails_instead_of_inventing_density():
    with pytest.raises(ValueError,match='area'):
        count_prior(np.array([[1,2]]),np.array([0.]),np.array([2]),np.arange(3.),
                    nside=4,nside_parent=2,n0=100,meta={})
    with pytest.raises(ValueError,match='empty'):
        merge_empty_counts(np.zeros((2,3)),np.arange(4.))


def test_qso_reuse_changes_provenance_not_abundance_or_source():
    old=dict(sigma=[[1.,2.]],z_centres=[1.],mag_centres=[20.,21.],meta=dict(model_run_id='old',population='psf'))
    new=rebind_qso_prior(old,run_id='new',origin='source',origin_sha256='hash')
    assert new['sigma']==old['sigma'] and new['z_centres']==old['z_centres'] and new['mag_centres']==old['mag_centres']
    assert new['meta']['model_run_id']=='new' and old['meta']['model_run_id']=='old'
    assert new['meta']['abundance_source_model_run_id']=='old'
