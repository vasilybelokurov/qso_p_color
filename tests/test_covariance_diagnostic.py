import numpy as np
import pytest

from qso_pcolor.covariance_diagnostic import floor_covariance, constrained_step
from qso_pcolor.gaussmix import GaussianMixture


def test_floor_changes_only_small_eigenvalues_and_is_rotation_equivariant():
    angle=.4;u=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    s=u@np.diag([.0001,4.])@u.T
    c=floor_covariance(s,.001)
    np.testing.assert_allclose(c,u@np.diag([.001,4.])@u.T,atol=1e-14)
    np.testing.assert_allclose(floor_covariance(2*np.eye(2),.001),2*np.eye(2))


def test_constrained_xd_increases_likelihood_with_missing_bands_correlated_noise_and_weights():
    rng=np.random.default_rng(838)
    x=rng.normal(size=(79,3));obs=rng.uniform(size=x.shape)>.3;obs[:,0]=True;x[~obs]=np.nan
    a=rng.normal(size=(len(x),3,3));cov=.02*(a@a.swapaxes(1,2));weights=rng.uniform(.2,2,len(x))
    mix=GaussianMixture(np.array([.4,.6]),np.array([[-1.,0.,0.],[1.,0.,0.]]),np.tile(np.eye(3),(2,1,1)),('a','b','c'))
    def batches():
        for lo in range(0,len(x),11):yield x[lo:lo+11],cov[lo:lo+11],obs[lo:lo+11],weights[lo:lo+11]
    for _ in range(20):
        mix,result=constrained_step(batches,mix,floor=.01,expected_rows=len(x))
        assert result['change']>=-1e-12 and result['min_eigenvalue']>=.01-1e-12
    with pytest.raises(ValueError,match='constraint'):
        constrained_step(batches,mix,floor=10.,expected_rows=len(x))


def test_floor_replaces_repeated_inflation_when_one_dimension_is_missing():
    x=np.array([[-1.,np.nan],[1.,np.nan]])
    mix=GaussianMixture(np.ones(1),np.zeros((1,2)),np.array([np.diag([1.,2.])]),('a','b'))
    def source():yield x,np.zeros((2,2,2)),np.array([[True,False],[True,False]]),np.ones(2)
    for _ in range(4):mix,_=constrained_step(source,mix,floor=.001,expected_rows=2)
    np.testing.assert_allclose(mix.covs[0],np.diag([1.,2.]),atol=1e-14)
