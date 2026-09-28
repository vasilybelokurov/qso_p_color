"""Pooled catch-all weights cannot disappear in bins without tail examples."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from qso_pcolor.outlier import fit_pooled_outlier_fraction


def test_empty_and_no_outlier_bins_borrow_support_without_a_probability_floor():
    assert fit_pooled_outlier_fraction([], [], prior_mean=.01, prior_strength=100) == .01
    # Every datum is certainly background: exactly the pseudo-count solution.
    for n in (10, 1000):
        result = fit_pooled_outlier_fraction(np.zeros(n), np.full(n, -np.inf),
                                           prior_mean=.01, prior_strength=100)
        assert result == pytest.approx(1 / (n + 100), rel=1e-12)
    result = fit_pooled_outlier_fraction([-np.inf], [0.], prior_mean=.01, prior_strength=100)
    assert result == pytest.approx(2 / 101)


def test_pooled_fit_matches_independent_penalised_likelihood_and_weight_replication():
    a, b = np.array([0., -8., -4.]), np.array([-6., -2., -3.])
    w = np.array([20, 3, 2])
    p, strength = .05, 15.
    def loss(eta):
        return -(w @ np.logaddexp(np.log1p(-eta) + a, np.log(eta) + b)
                 + strength * (p * np.log(eta) + (1-p) * np.log1p(-eta)))
    expected = minimize_scalar(loss, bounds=(1e-10, 1-1e-10), method="bounded",
                               options={"xatol": 1e-12}).x
    got = fit_pooled_outlier_fraction(a, b, weights=w, prior_mean=p, prior_strength=strength)
    repeated = fit_pooled_outlier_fraction(np.repeat(a, w), np.repeat(b, w),
                                          prior_mean=p, prior_strength=strength)
    assert got == pytest.approx(expected, abs=1e-8)
    assert got == pytest.approx(repeated, abs=1e-14)


@pytest.mark.parametrize("mean,strength", [(0., 1.), (1., 1.), (.1, 0.), (.1, np.inf)])
def test_invalid_pooling_is_refused(mean, strength):
    with pytest.raises(ValueError, match="pooling"):
        fit_pooled_outlier_fraction([0.], [-1.], prior_mean=mean, prior_strength=strength)


def test_invalid_densities_are_not_silently_dropped():
    with pytest.raises(ValueError, match="densities"):
        fit_pooled_outlier_fraction([-np.inf], [-np.inf], prior_mean=.1, prior_strength=10)


@pytest.mark.parametrize("family", ["gaussian", "student_t"])
def test_joint_catchall_marginalises_all_41_singletons_pairs_and_larger_subsets(family):
    from itertools import combinations
    from scipy.stats import multivariate_normal, multivariate_t
    from qso_pcolor.multisurvey import MultiSurveyOutlier
    from qso_pcolor.multisurvey_data import band_labels
    rng = np.random.default_rng(109)
    a = rng.normal(size=(41,41)) / np.sqrt(41)
    scale = a @ a.T + np.eye(41)
    mean = np.full(41,20.)
    outlier = MultiSurveyOutlier(mean,scale,2.,0.,band_labels(),"test",
        {"*":([10.,30.],[.01])},family=family,nu=2. if family=="student_t" else None,noise="exact")
    subsets = [(j,) for j in range(41)] + list(combinations(range(41),2))
    subsets += [tuple(sorted(rng.choice(41,size,replace=False))) for size in range(3,42)]
    x = mean + rng.normal(size=41)
    for dims in subsets:
        idx = np.array(dims); anchor=idx[0]
        observed = np.zeros((1,41),bool); observed[:,idx]=True
        got = outlier.conditional(anchor,band_labels()[anchor],"test").log_prob(x[None],observed=observed)[0]
        if family=="gaussian":
            expected = (multivariate_normal.logpdf(x[idx],mean[idx],scale[np.ix_(idx,idx)])
                - multivariate_normal.logpdf(x[anchor],mean[anchor],scale[anchor,anchor]))
        else:
            expected = (multivariate_t.logpdf(x[idx],mean[idx],scale[np.ix_(idx,idx)],df=2.)
                - multivariate_t.logpdf(x[anchor],mean[anchor],scale[anchor,anchor],df=2.))
        assert got == pytest.approx(expected,abs=1e-9)


def test_catchall_refuses_nan_weights_and_reversed_magnitude_edges():
    from qso_pcolor.multisurvey import MultiSurveyOutlier
    for edges,values in (([10.,30.],[np.nan]),([30.,10.],[.01])):
        with pytest.raises(ValueError,match="fraction"):
            MultiSurveyOutlier([20.],[[1.]],2.,0.,("test",),"test",{"*":(edges,values)})
