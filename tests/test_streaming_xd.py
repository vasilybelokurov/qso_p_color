import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.xd import fit_xd
from qso_pcolor.streaming_xd import fit_xd_batches


@pytest.mark.parametrize('batch_size', [1, 17, 80])
def test_streamed_em_matches_dense_with_missing_correlated_errors(batch_size):
    rng = np.random.default_rng(521)
    x = rng.normal(size=(73, 3)) + np.array([1e5, 0., 2.])
    obs = rng.uniform(size=x.shape) > .3
    obs[:, 0] = True
    x[~obs] = np.nan
    a = rng.normal(size=(73, 3, 3))
    cov = a @ a.swapaxes(1, 2) * .01
    weights = rng.uniform(.2, 2, len(x))
    init = GaussianMixture(np.array([.4, .6]), np.array([[1e5, -.5, 1.], [1e5+1, .5, 3.]]),
                           np.tile(np.eye(3), (2, 1, 1)), ('a', 'b', 'c'))
    visits = []
    def source():
        ids = []
        for lo in range(0, len(x), batch_size):
            hi = min(lo+batch_size, len(x)); ids.extend(range(lo, hi))
            yield x[lo:hi], cov[lo:hi], obs[lo:hi], weights[lo:hi]
        visits.append(ids)
    options = dict(init=init, max_iter=4, tol=0., regularization=.001)
    dense = fit_xd(x, cov, observed=obs, weights=weights, labels=init.labels, **options)
    stream = fit_xd_batches(source, expected_rows=len(x), **options)
    assert len(visits) == 5
    assert all(v == list(range(len(x))) for v in visits)
    for name in ['weights', 'means', 'covs']:
        np.testing.assert_allclose(getattr(stream.mixture, name), getattr(dense.mixture, name), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(stream.history, dense.history, rtol=1e-8)
    assert stream.mean_loglike == pytest.approx(dense.mean_loglike, abs=1e-8)


def test_stream_refuses_truncated_training_pass():
    mix = GaussianMixture(np.ones(1), np.zeros((1, 1)), np.ones((1, 1, 1)), ('x',))
    def source():
        yield np.zeros((3, 1)), np.zeros((3, 1, 1)), np.ones((3, 1), bool), np.ones(3)
    with pytest.raises(ValueError, match='accounting'):
        fit_xd_batches(source, init=mix, expected_rows=4, max_iter=1, tol=0., regularization=.01)


def test_interrupted_stream_fit_resumes_same_iterations_and_likelihood():
    rng=np.random.default_rng(311);x=rng.normal(size=(29,2));cov=np.tile(np.eye(2)*.03,(29,1,1));obs=np.ones_like(x,bool)
    mix=GaussianMixture(np.ones(1),np.zeros((1,2)),np.tile(np.eye(2),(1,1,1)),('a','b'))
    def source():
        for lo in range(0,len(x),7):yield x[lo:lo+7],cov[lo:lo+7],obs[lo:lo+7],np.ones(len(x[lo:lo+7]))
    kw=dict(expected_rows=29,max_iter=5,tol=0.,regularization=.001)
    full=fit_xd_batches(source,init=mix,**kw);saved={};history=[]
    def stop(it,model,ll,rows):
        history.append(ll);saved['mixture']=model
        if it==2:raise InterruptedError
    with pytest.raises(InterruptedError):fit_xd_batches(source,init=mix,progress=stop,**kw)
    resumed=fit_xd_batches(source,init=saved['mixture'],initial_history=tuple(history),**kw)
    np.testing.assert_allclose(resumed.mixture.covs,full.mixture.covs,rtol=1e-14)
    assert resumed.history==full.history and resumed.mean_loglike==full.mean_loglike and resumed.n_iter==5
