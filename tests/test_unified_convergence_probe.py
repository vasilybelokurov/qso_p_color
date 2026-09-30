import importlib
from pathlib import Path
import numpy as np

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import accumulate_projected, fit_projected, native_mixture


def fixture():
    rng=np.random.default_rng(136)
    y=rng.normal(size=(111,3));obs=rng.random(y.shape)>.2;obs[:,0]=True;y[~obs]=np.nan
    a=rng.normal(size=(111,3,3));c=a@a.swapaxes(1,2)*.01
    h=np.array([[1.,0.],[0.,1.],[.3,1.]])
    op=(h,np.zeros(3),np.eye(3)*.02)
    mix=GaussianMixture(np.array([.3,.7]),np.array([[0.,0.],[1.,1.]]),np.tile(np.eye(2),(2,1,1)))
    def source():yield y,c,obs,np.zeros(111,int)
    return source,mix,op


def test_probe_additive_step_reproduces_production_kernel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    update=importlib.import_module('probe_unified_convergence').update_from_statistics
    source,mix,op=fixture();current=mix
    for _ in range(4):current=update(current,accumulate_projected(source,current,{0:op}),.001,'additive')
    expected=fit_projected(source,init=mix,operators={0:op},expected_rows=111,max_iter=4,tol=0.,regularization=.001)
    np.testing.assert_allclose(current.means,expected.mixture.means,atol=1e-12)
    np.testing.assert_allclose(current.covs,expected.mixture.covs,atol=1e-12)


def test_projected_constrained_updates_improve_likelihood_and_respect_floor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    update=importlib.import_module('probe_unified_convergence').update_from_statistics
    source,mix,op=fixture();history=[]
    for _ in range(10):
        stats=accumulate_projected(source,mix,{0:op});history.append(stats[3]/stats[4])
        mix=update(mix,stats,.01,'eigenvalue_floor')
        assert np.linalg.eigvalsh(mix.covs).min()>=.01-1e-12
    assert np.min(np.diff(history))>=-1e-10
