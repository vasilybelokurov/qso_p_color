"""Independent numerical checks of mixed-survey and infrared-only scoring."""
from itertools import combinations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import (BandLuptitudeTransform, MultiSurveyModel,
                                   conditional_log_prob)
from qso_pcolor.multisurvey_data import (Photometry, SURVEYS, band_labels,
                                        catalogue_photometry)
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel


def test_training_redshift_edges_have_no_roundoff_width_terminal_slice():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/"scripts/train_multisurvey_model.py"
    spec = importlib.util.spec_from_file_location("train_multisurvey_edges", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    edges = module.redshift_edges(.1, 4.4, .1)
    assert len(edges) == 44
    assert edges[0] == .1 and edges[-1] == 4.4
    assert np.allclose(np.diff(edges), .1, rtol=0, atol=1e-14)
    assert np.allclose(module.redshift_edges(.1, .45, .1), [.1, .2, .3, .4, .45])
    assert np.array_equal(module.redshift_edges(.1, .15, .2), [.1, .15])


def test_conditional_mixture_against_independent_gaussian_formula_and_quadrature():
    weights = np.array([.3, .7])
    means = np.array([[0., 1.], [2., -1.]])
    intrinsic = np.array([[[2., .7], [.7, 1.]], [[1., -.4], [-.4, 2.]]])
    noise = np.array([[.4, .12], [.12, .3]])
    mix = GaussianMixture(weights, means, intrinsic)
    a = .8
    total = intrinsic + noise
    w = weights * norm.pdf(a, means[:, 0], np.sqrt(total[:, 0, 0]))
    w /= w.sum()
    mu = means[:, 1] + total[:, 1, 0]/total[:, 0, 0]*(a-means[:, 0])
    var = total[:, 1, 1] - total[:, 1, 0]**2/total[:, 0, 0]

    def density(y):
        return np.exp(conditional_log_prob(mix, np.array([[a, y]]), noise[None],
                                          np.ones((1, 2), bool), 0)[0])
    for y in (-3., 0., 2., 4.):
        assert density(y) == pytest.approx(np.dot(w, norm.pdf(y, mu, np.sqrt(var))), rel=1e-12)
    assert quad(density, -np.inf, np.inf)[0] == pytest.approx(1., abs=1e-8)


def test_transform_preserves_negative_flux_and_full_covariance_against_monte_carlo():
    bands = ("allwise:w1", "allwise:w2", "vhs:j")
    f = np.array([-2., 3., 8.])
    covariance = np.array([[.0009, .0003, -.0001], [.0003, .0016, .0002], [-.0001, .0002, .0025]])
    p = Photometry(f, np.diag(covariance), bands)
    transform = BandLuptitudeTransform(bands, np.ones(3))
    result = transform(p, flux_covariance=covariance[None])
    assert result.observed.all()
    draws = np.random.default_rng(8).multivariate_normal(f, covariance, 120000)
    actual = 22.5 - 2.5/np.log(10.) * np.arcsinh(draws/2.)
    assert np.allclose(np.cov(actual.T), result.cov[0], rtol=.05, atol=1e-6)


def test_allwise_uses_raw_flux_even_when_negative():
    rows = {"ra": np.array([1.]), "dec": np.array([2.]), "cc_flags": np.array(["0000"])}
    for b in SURVEYS["allwise"].bands:
        rows[f"value_{b}"] = np.array([-2.])
        rows[f"error_{b}"] = np.array([3.])
    p = catalogue_photometry("allwise", rows, clean=True, vhs_bad_bits=0)
    assert p.observed.all() and (p.flux < 0).all()
    assert p.flux[0, 0] == pytest.approx(-2 * 10**.8)
    assert p.variance[0, 0] == pytest.approx(9 * 10**1.6)


def test_legacy_north_and_south_are_distinct_bands():
    rows = dict(ra=np.arange(3.), dec=np.zeros(3), release=np.array([9010, 9011, 9012]),
                maskbits=np.zeros(3, int))
    for b in "grz":
        rows[f"value_{b}"] = np.ones(3)
        rows[f"error_{b}"] = np.ones(3)
        rows[f"nobs_{b}"] = np.ones(3, int)
    p = catalogue_photometry("decals", rows, clean=True, vhs_bad_bits=0)
    assert np.array_equal(p.observed[:, 0], [True, False, True])
    assert np.array_equal(p.observed[:, 3], [False, True, False])


def synthetic_model():
    labels = band_labels(); d = len(labels)
    covariance = .2*np.eye(d) + .5*np.ones((d, d))
    mixtures = [GaussianMixture(np.ones(1), np.full((1, d), mean), covariance[None], labels)
                for mean in (20., 22.)]
    qso = SlicedColourRedshiftModel(np.array([.5, 2.5]), mixtures, np.array([1000, 1000]),
                                   "test_multisurvey", labels)
    background = GaussianMixture(np.ones(1), np.full((1, d), 21.),
                                 (covariance*2)[None], labels)
    transform = BandLuptitudeTransform(labels, np.full(d, .1))
    return MultiSurveyModel(qso, background, transform, labels,
                            np.tile([10., 30.], (d, 1)))


def test_background_marginal_routing_against_gaussian_conditioning():
    model = synthetic_model()
    labels = ('decals_dr9_south:r', 'decals_dr9_south:g')  # reverse input order
    marginal = GaussianMixture(np.ones(1), np.array([[20., 21.]]),
                               np.array([[[.5, .2], [.2, .8]]]), labels)
    model.background_marginals = (marginal,)
    d = len(model.transform.bands)
    r, g, extra = [model.transform.bands.index(b) for b in (*labels, 'allwise:w1')]
    x = np.full((2,d), np.nan); observed = np.zeros((2,d), bool)
    x[:,[r,g]] = [20.4,21.7]; observed[:,[r,g]] = True
    x[1,extra] = 19.; observed[1,extra] = True
    cov = np.broadcast_to(.1*np.eye(d),(2,d,d)).copy()
    result = model.background_log_prob(x,cov,observed,r)
    # Independent univariate Gaussian conditioning for the selected marginal.
    mean = 21. + .2/.6*(20.4-20.)
    variance = .9-.2**2/.6
    assert result[0] == pytest.approx(norm.logpdf(21.7,mean,np.sqrt(variance)))
    # An extra observed band must keep the original joint background exactly.
    baseline = conditional_log_prob(model.background,x[1:],cov[1:],observed[1:],r)[0]
    assert result[1] == pytest.approx(baseline,abs=1e-12)


def test_background_marginal_public_scorer_and_serialisation(tmp_path):
    model = synthetic_model()
    labels = ('decals_dr9_south:g','decals_dr9_south:r','decals_dr9_south:z')
    phot = Photometry([[3.,5.,7.]],[[.01,.01,.01]],labels)
    kwargs = dict(z_primary=np.array([1.5]), l_deg=np.array([180.]), b_deg=np.array([45.]),
                  match=RedshiftMatch(half_width_kms=2000.),min_bands=2)
    original = model.score(phot,**kwargs)[0]
    model.background_marginals = (GaussianMixture(np.ones(1),np.array([[21.,20.,19.]]),
                                                   np.diag([.3,.4,.5])[None],labels),)
    path = tmp_path/'model.json'; model.save(path); restored = MultiSurveyModel.load(path)
    result = restored.score(phot,**kwargs)[0]
    f = restored.transform(phot); g,r,z = [f.labels.index(b) for b in labels]
    expected = (norm.logpdf(f.x[0,r],20.,np.sqrt(.4+f.cov[0,r,r])) +
                norm.logpdf(f.x[0,z],19.,np.sqrt(.5+f.cov[0,z,z])))
    # This synthetic model's reference priority is input-schema order: g first.
    assert result.reference_band == labels[0]
    assert result.loglike_bkg == pytest.approx(expected,abs=1e-10)
    assert result.p_zmatch_given_qso == original.p_zmatch_given_qso
    assert result.loglike_qso_zprimary == original.loglike_qso_zprimary


def test_all_127_survey_combinations_and_two_band_ir_only():
    model = synthetic_model()
    p = Photometry(np.full((1, len(model.transform.bands)), 3.),
                   np.full((1, len(model.transform.bands)), .01), model.transform.bands)
    kw = dict(z_primary=np.array([1.5]), l_deg=np.array([180.]), b_deg=np.array([45.]),
              match=RedshiftMatch(half_width_kms=2000.), min_bands=2)
    count = 0
    for size in range(1, 8):
        for surveys in combinations(SURVEYS, size):
            result = model.score(p.keep_surveys(surveys), **kw)[0]
            assert np.isfinite(result.log_bayes_factor_qz_bkg)
            assert 0 <= result.p_zmatch_given_qso <= 1
            assert np.isnan(result.p_sameq)
            assert result.status == "no_prior_posterior_unavailable"
            assert set(result.surveys_used) == set(surveys)
            count += 1
    assert count == 127
    infrared = Photometry([[3., -1.]], [[.1, .1]], ("allwise:w1", "allwise:w2"))
    result = model.score(infrared, **kw)[0]
    assert result.n_bands_used == 2
    assert result.reference_band == "allwise:w1"
    assert np.isfinite(result.log_bayes_factor_qz_bkg)


def test_order_missing_anchor_batch_alignment_and_serialisation(tmp_path):
    model = synthetic_model()
    bands = ("allwise:w2", "sdss:r", "allwise:w1")
    p = Photometry([[2., np.nan, 3.], [3., 4., 5.]], [[.1, np.inf, .1], [.1, .1, .1]], bands)
    kw = dict(z_primary=np.array([1., 2.]), l_deg=np.array([180., 80.]),
              b_deg=np.array([45., 30.]), match=RedshiftMatch(half_width_kms=2000.), min_bands=2)
    result = model.score(p, candidate_id=np.array(["infrared", "optical"]), **kw)
    assert [r.candidate_id for r in result] == ["infrared", "optical"]
    for i in range(2):
        single = model.score(p.subset([i]), **{k: v[[i]] if isinstance(v, np.ndarray) else v for k, v in kw.items()})[0]
        assert result[i].log_bayes_factor_qz_bkg == pytest.approx(single.log_bayes_factor_qz_bkg)
    path = tmp_path / "model.json"; model.save(path)
    loaded = MultiSurveyModel.load(path)
    assert loaded.transform_id == model.transform_id
    again = loaded.score(p, **kw)
    assert np.allclose([r.log_bayes_factor_qz_bkg for r in result],
                       [r.log_bayes_factor_qz_bkg for r in again])


def test_unknown_or_duplicate_filters_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        Photometry([[1]], [[1]], ("g",))
    with pytest.raises(ValueError, match="duplicate"):
        Photometry([[1, 1]], [[1, 1]], ("sdss:g", "sdss:g"))


def test_absent_or_invalid_measurements_return_explicit_status():
    model=synthetic_model()
    p=Photometry([[np.nan,2.],[1.,2.]],[[np.inf,0.],[-1.,np.inf]],("sdss:u","allwise:w1"))
    rows=model.score(p,z_primary=np.array([1.5,1.5]),l_deg=np.zeros(2),b_deg=np.full(2,45.),
                     match=RedshiftMatch(half_width_kms=2000.),min_bands=2)
    assert all(r.status=="insufficient_photometry" for r in rows)


def test_pattern_sampling_weights_restore_original_band_fractions():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/"scripts/train_multisurvey_model.py"
    spec=importlib.util.spec_from_file_location("multisurvey_trainer_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observed=np.array([[True,True,False]]*1000+[[False,True,True]]*12)
    idx,weights=module.sample_patterns(observed,np.ones(1012,bool),50,10,np.random.default_rng(7))
    assert len(idx)==50
    assert weights.sum()==pytest.approx(1012)
    assert np.allclose(np.average(observed[idx],axis=0,weights=weights[idx]),
                       np.array([1000,1012,12])/1012)


def test_ir_posteriors_keep_the_field_quasar_and_require_matching_priors():
    from qso_pcolor.priors import GridQSOPrior

    labels=("allwise:w1", "allwise:w2")
    z=np.linspace(.5,2.5,11)
    mixtures=[GaussianMixture(np.ones(1),np.array([[20.,20.+zz]]),
                              np.diag([.2,.04])[None],labels) for zz in z]
    qso=SlicedColourRedshiftModel(z,mixtures,np.full(len(z),1000),"ir_test",labels)
    bkg=GaussianMixture(np.ones(1),np.array([[20.,16.]]),np.diag([.3,.5])[None],labels)
    tr=BandLuptitudeTransform(labels,np.ones(2))
    model=MultiSurveyModel(qso,bkg,tr,labels,np.array([[10.,30.],[10.,30.]]))
    # Invert the independently specified luptitudes to construct test fluxes.
    flux=2*np.sinh((22.5-np.array([[20.,22.]]))/(2.5/np.log(10.)))
    phot=Photometry(flux,np.full((1,2),.001),labels)
    meta=dict(reference_band=labels[0],transform_id=model.transform_id)
    prior=GridQSOPrior(z,np.array([15.,25.]),np.ones((len(z),2)),meta=meta.copy())

    class Density:
        def __init__(self): self.meta=meta.copy()
        def __call__(self,m,l,b): return np.full(len(m),2.)
        def level(self,m,l,b): return np.full(len(m),2)

    args=dict(match=RedshiftMatch(half_width_kms=2000.),min_bands=2,
              l_deg=np.array([180.]),b_deg=np.array([45.]),
              priors={labels[0]:(prior,Density())})
    correct=model.score(phot,z_primary=np.array([2.]),**args)[0]
    wrong=model.score(phot,z_primary=np.array([1.]),**args)[0]
    assert wrong.log_bayes_factor_qz_bkg>0  # still quasar-like relative to the field
    assert wrong.log_r_per_unit_z<correct.log_r_per_unit_z-8
    for row in (correct,wrong):
        assert row.status=="ok"
        assert row.p_sameq==pytest.approx(np.exp(row.log_r_per_unit_z)*row.dz_match_eff)
        assert 0<row.p_sameq<1
    prior.meta["reference_band"]="sdss:r"
    with pytest.raises(ValueError,match="prior reference"):
        model.score(phot,z_primary=np.array([2.]),**args)


def test_validation_redshift_density_against_normalised_gaussian_interpolation():
    import importlib.util
    from pathlib import Path

    path=Path(__file__).parents[1]/"scripts/validate_multisurvey.py"
    spec=importlib.util.spec_from_file_location("multisurvey_validation_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    labels=("allwise:w1","allwise:w2")
    centres=np.array([.5,2.5])
    mixtures=[GaussianMixture(np.ones(1),np.array([[20.,20.+z]]),
                              np.diag([.2,.4])[None],labels) for z in centres]
    qso=SlicedColourRedshiftModel(centres,mixtures,np.array([1000,1000]),"ir_test",labels)
    transform=BandLuptitudeTransform(labels,np.ones(2))
    model=MultiSurveyModel(qso,mixtures[0],transform,labels,np.array([[10.,30.],[10.,30.]]))
    flux=2*np.sinh((22.5-np.array([[20.,21.]]))/(2.5/np.log(10.)))
    phot=Photometry(flux,np.full((1,2),.001),labels)
    features=transform(phot)
    endpoints=norm.pdf(features.x[0,1],20.+centres,np.sqrt(.4+features.cov[0,1,1]))
    integral=quad(lambda z: np.interp(z,centres,endpoints),*centres)[0]
    result=module.evaluate(model,phot,np.array([1.]),np.linspace(*centres,31),[.68],np.array([2.]))
    assert np.exp(result["log_redshift_density"][0])==pytest.approx(np.interp(1.,centres,endpoints)/integral)
    assert np.exp(result["alternative_log_redshift_density"][0])==pytest.approx(np.interp(2.,centres,endpoints)/integral)


def test_query_preserves_input_identity_and_reuses_completed_cache(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from qso_pcolor.multisurvey_data import match_catalogue

    calls=[]
    def fake_join(query,table,data,names,**kwargs):
        idx,ra,dec=data
        calls.append(ra.copy())
        # Return rows reversed, as an SQL result need not follow upload order.
        return dict(idx=idx[::-1],ra=ra[::-1],dec=dec[::-1],value_g=2*ra[::-1])
    monkeypatch.setitem(sys.modules,"sqlutilpy",SimpleNamespace(local_join=fake_join))
    ra=np.array([12.,3.,27.,11.,8.]);dec=np.arange(5.)
    result=match_catalogue("sdss",ra,dec,tmp_path,radius_arcsec=1.)
    assert [len(c) for c in calls]==[5]
    assert np.array_equal(result["idx"],np.arange(5))
    assert np.array_equal(result["ra"],ra)
    assert np.array_equal(result["value_g"],2*ra)
    again=match_catalogue("sdss",ra,dec,tmp_path,radius_arcsec=1.)
    assert len(calls)==1
    assert np.array_equal(again["ra"],ra)
