from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from scipy.special import logsumexp

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import BandLuptitudeTransform, MultiSurveyModel, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry, band_labels
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.unified import observation_layout, native_view, conditional_predictive_mixture, qso_support, catalogue_photometry


def fixture_model():
    labels = ('sdss:g', 'sdss:r', 'sdss:i')
    a = GaussianMixture(np.array([.3,.7]), np.array([[20.,19.,18.],[19.,19.,20.]]),
        np.array([[[1.,.4,.2],[.4,.5,.3],[.2,.3,1.]],np.eye(3)*.7]), labels)
    b = GaussianMixture(a.weights[::-1], a.means+.2, a.covs*1.2, labels)
    q = SlicedColourRedshiftModel([1.,2.], [a,b], [100,100], 'test', labels)
    return MultiSurveyModel(q,a,BandLuptitudeTransform(labels,np.ones(3)*.1),
        labels,np.array([[0.,40.]]*3),meta={'reference_min_snr':5.})


def test_reference_prefers_detection_without_discarding_negative_band():
    model=fixture_model();p=Photometry([[-1.,10.,2.],[.1,.2,.3]],np.ones((2,3)),model.transform.bands)
    np.testing.assert_array_equal(model.reference_indices(p),[1,0])
    assert p.observed.all()
    assert np.isfinite(model.transform(p).x).all()


def test_catalogue_coordinates_provenance_and_ab_magnitude_conversion():
    mags=np.array([[20.,19.],[21.,22.]])
    p=catalogue_photometry(mags,np.full_like(mags,.1),['legacy:g','sdss:r'],
        ra_deg=[180.,180.],dec_deg=[50.,0.],measurement='abmag')
    np.testing.assert_array_equal(p.observed,[[False,True,True],[True,False,True]])
    assert p.flux[0,1]==pytest.approx(10.)
    override=catalogue_photometry([[-1.]],[[.2]],['legacy:g'],ra_deg=180.,dec_deg=50.,legacy_hemisphere='south')
    assert override.flux[0,0]==-1 and override.observed[0,0]
    assert not override.observed[0,1]


def test_support_conditional_matches_scorer_with_correlated_errors_and_masks():
    model=fixture_model();x=np.array([20.4,19.2,18.8]);c=np.array([[.3,.08,0],[.08,.2,.03],[0,.03,.1]])
    for bits in itertools.product([False,True],repeat=3):
        obs=np.array(bits)
        if obs.sum()<2:continue
        anchor=np.flatnonzero(obs)[0]
        mix,ix=conditional_predictive_mixture(model.qso,1.3,x,c,obs,anchor)
        expected=logsumexp([np.log(.7)+conditional_log_prob(model.qso.mixtures[0],x[None],c[None],obs[None],anchor)[0],
                           np.log(.3)+conditional_log_prob(model.qso.mixtures[1],x[None],c[None],obs[None],anchor)[0]])
        assert mix.log_prob(x[ix])[0]==pytest.approx(expected,abs=1e-11)


def test_support_rejects_remote_tail_and_is_batch_order_invariant():
    model=fixture_model();x=np.array([[20.,19.,18.],[10.,19.,30.]])
    soft=model.transform.softening;factor=2.5/np.log(10)
    flux=2*soft*np.sinh((22.5-x)/factor-np.log(soft))
    variance=(flux*flux+4*soft*soft)*.01/factor**2
    p=Photometry(flux,variance,model.transform.bands)
    result=qso_support(model,p,[1.3,1.3],draws=511,seed=42)
    assert result['percentile'][0]>.02
    assert result['percentile'][1]==1/512
    reverse=qso_support(model,p.subset([1,0]),[1.3,1.3],draws=511,seed=42)
    np.testing.assert_array_equal(reverse['percentile'],result['percentile'][::-1])
    single=qso_support(model,p.keep_bands(('sdss:r',)),1.3,draws=20,seed=42)
    assert np.isnan(single['percentile']).all()
    assert (single['status']=='no_colour_information').all()


def test_full_native_layout_preserves_other_surveys_and_joint_roundtrip():
    labels=band_labels();tr=BandLuptitudeTransform(labels,np.linspace(.1,1.,len(labels)))
    cal={'matrix':[[1.05,-.05,0],[0,1.02,-.02],[0,-.01,1.01]],
        'populations':{k:{'offset':[.01,.02,.03],'extra_covariance':(np.eye(3)*.001).tolist()} for k in ('qso','stars')}}
    layout=observation_layout(tr,cal,native_variance_floor=1e-6)
    assert len(layout['latent_labels'])==36
    mix=GaussianMixture(np.ones(1),np.arange(36.)[None],np.eye(36)[None],tuple(layout['latent_labels']))
    view=native_view(mix,layout,'qso');restored=GaussianMixture.from_dict(view.to_dict())
    for label in labels:
        if not label.startswith('decals_dr9_north:'):
            k=labels.index(label);j=layout['canonical_indices'].index(k)
            assert view.means[0,k]==mix.means[0,j]
    np.testing.assert_allclose(view.covs,restored.covs)
    for j in range(41):
        one=np.zeros((1,41),bool);one[0,j]=True
        expected=-.5*np.log(2*np.pi*(view.covs[0,j,j]+.01))
        assert view.log_prob(view.means,np.eye(41)*.01,observed=one)[0]==pytest.approx(expected)
    obs=np.zeros((1,41),bool);obs[0,[0,10,13,15,36]]=True
    assert np.isfinite(view.log_prob(view.means,np.eye(41)*.01,observed=obs)).all()
