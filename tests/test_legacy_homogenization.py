"""Protect covariance propagation and missing-band behaviour in shared views."""
from itertools import combinations
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy_homogenization import affine_photometry, cubic_photometry, tied_native_mixture
from qso_pcolor.multisurvey import BandLuptitudeTransform
from qso_pcolor.multisurvey_data import Photometry


def test_affine_covariance_includes_induced_correlations():
    a=np.array([[1.,.1,0],[0,1.,.2],[.1,0,1.]])
    x=np.array([[20.,21.,22.]])
    s=np.array([np.diag([.1,.2,.3])])
    out,cov=affine_photometry(x,s,a,np.array([.1,.2,.3]))
    assert cov[0,0,1] != 0
    np.testing.assert_allclose(cov[0],a@s[0]@a.T)
    np.testing.assert_allclose(out[0],a@x[0]+[.1,.2,.3])
    with pytest.raises(ValueError,match='complete'):
        affine_photometry(x*np.nan,s,a,np.zeros(3))


def test_cubic_jacobian_matches_finite_difference_including_bounded_colour():
    x=np.array([[21.,20.,19.],[23.,20.,19.]])
    coef=np.array([[.01,.02,.03],[.02,-.01,.01],[.001,.002,-.001],[.0001,0,.0002]])
    _,jac=cubic_photometry(x,coef,(.3,3.3))
    for j in range(3):
        dx=np.zeros_like(x);dx[:,j]=1e-5
        yp,_=cubic_photometry(x+dx,coef,(.3,3.3));ym,_=cubic_photometry(x-dx,coef,(.3,3.3))
        np.testing.assert_allclose(jac[:,:,j],(yp-ym)/2e-5,atol=1e-9)


def test_shared_mixture_all_northern_subsets_match_direct_gaussian():
    # Three south variables and three views; one northern band needs no input colours.
    mix=GaussianMixture(np.ones(1),np.array([[1.,2.,3.,8.,9.,10.]]),np.eye(6)[None],labels=tuple('abcdef'))
    ni,si=np.arange(3,6),np.arange(3)
    a=np.array([[1.,.1,0],[0,1,.2],[.1,0,1]])
    b=np.array([.1,.2,.3]); residual=np.diag([.02,.03,.04])
    tied=tied_native_mixture(mix,ni,si,a,b,residual)
    np.linalg.cholesky(tied.covs)
    np.testing.assert_allclose(tied.covs[0,:3,:3],np.eye(3))
    for size in (1,2,3):
        for dims in combinations(range(3),size):
            cols=3+np.array(dims); obs=np.zeros((1,6),bool);obs[:,cols]=True
            x=np.full((1,6),np.nan);x[:,cols]=np.arange(size)+1.
            noise=np.eye(6)[None]*.1
            got=tied.log_prob(x,noise,observed=obs)[0]
            mean=a@mix.means[0,:3]+b; cov=a@a.T+residual+np.eye(3)*.1
            expected=multivariate_normal.logpdf(x[0,cols],mean[np.array(dims)],cov[np.ix_(dims,dims)])
            assert got==pytest.approx(expected)


def test_joint_north_and_south_are_correlated_views_not_duplicate_independent_bands():
    mix=GaussianMixture(np.ones(1),np.zeros((1,2)),np.eye(2)[None],labels=('s','n'))
    tied=tied_native_mixture(mix,np.array([1]),np.array([0]),np.ones((1,1)),np.zeros(1),np.array([[.1]]))
    np.testing.assert_allclose(tied.covs[0],[[1,1],[1,1.1]])


def test_negative_flux_single_band_remains_observed_in_shared_model():
    tr=BandLuptitudeTransform(('decals_dr9_south:g','decals_dr9_north:g'),np.ones(2))
    f=tr(Photometry(np.array([[-2.]]),np.array([[.3]]),('decals_dr9_north:g',)))
    assert f.observed[0,1] and not f.observed[0,0] and np.isfinite(f.x[0,1])
    mix=GaussianMixture(np.ones(1),np.full((1,2),23.),np.eye(2)[None],labels=tr.bands)
    tied=tied_native_mixture(mix,np.array([1]),np.array([0]),np.ones((1,1)),np.zeros(1),np.array([[.1]]))
    assert np.isfinite(tied.log_prob(f.x,f.cov,observed=f.observed)).all()


def test_shared_map_rejects_invalid_residual_covariance():
    mix=GaussianMixture(np.ones(1),np.zeros((1,2)),np.eye(2)[None],labels=('s','n'))
    with pytest.raises(np.linalg.LinAlgError):
        tied_native_mixture(mix,np.array([1]),np.array([0]),np.ones((1,1)),np.zeros(1),np.array([[-.1]]))
