"""Restart, separation, and full-row accounting for the bounded experiment."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qso_pcolor.full_sample import write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import accumulate_projected
from qso_pcolor.projected_parallel import ProjectedBatchFactory


@pytest.fixture
def trial(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('trial_unified_covariance')


def test_development_excludes_training_and_calibration_and_is_reproducible(trial):
    n=240; rng=np.random.default_rng(9)
    data=dict(role=np.repeat([0,1,2,3],60), zspec=np.ones(n), observed=np.ones((n,3),bool),
        flux=np.full((n,3),10.), variance=np.ones((n,3)), ra=np.full(n,180.),
        dec=np.tile([0.,60.],n//2), b=np.full(n,60.))
    data['flux'][180]=-1. # evaluation cut only
    opts=dict(seed=42, development_rows_per_hemisphere=8, score_rows_per_hemisphere=3,
              random_background_score_rows=10)
    cfg=dict(z_half_width=.1, reference_min_snr=5., min_abs_b_deg=25.)
    for kind in ('qso','stars'):
        task=dict(name=kind,kind=kind,z=1.)
        a=trial.select_development(data,task,cfg,opts);b=trial.select_development(data,task,cfg,opts)
        for key,rows in a.items():
            np.testing.assert_array_equal(rows,b[key]);assert (data['role'][rows]==3).all()
            assert 180 not in rows
        assert len(a['density'])==16 and len(a['score'])==6


def test_zero_prior_scores_remain_explicit_without_false_instability(trial):
    before=np.array([-np.inf,1.,2.,3.,-np.inf])
    after=np.array([-np.inf,1.01,2.01,-np.inf,4.])
    result=trial.shift_summary(before,after)
    assert result['stable_zero_weight']==1
    assert result['entered_zero_weight']==1 and result['left_zero_weight']==1
    assert result['nonfinite']==2
    assert result['p95_abs']==pytest.approx(.01)


@pytest.mark.parametrize('mode',['additive','eigenvalue_floor'])
def test_full_pass_checkpoint_resume_matches_uninterrupted_and_retains_weak_rows(trial,tmp_path,monkeypatch,mode):
    source=tmp_path/'source'; (source/'qso').mkdir(parents=True)
    rng=np.random.default_rng(88); n=37; d=3
    y=rng.normal(size=(n,d)); noise=np.full((n,d),.1);obs=rng.random((n,d))>.3;obs[:,0]=True
    for name,value in dict(y=y,noise=noise,observed=obs,role=np.zeros(n,int),source_row=np.arange(n),
        ra=np.full(n,180.),dec=np.tile([0.,60.],19)[:n],b=np.full(n,60.)).items():
        np.save(source/'qso'/(name+'.npy'),value)
    cfg=dict(batch_size=7,regularization=.001)
    write_json(source/'config.json',cfg)
    h=np.array([[1.,0.],[0.,1.],[.3,1.]])
    layout=dict(operators={'qso':dict(matrix=h.tolist(),offset=[0.,0.,0.],covariance=(np.eye(3)*.02).tolist())})
    write_json(source/'layout.json',layout)
    mix=GaussianMixture(np.array([.4,.6]),np.array([[0.,0.],[1.,1.]]),np.tile(np.eye(2),(2,1,1)))
    task=dict(name='qso_00',kind='qso',n=n,k=2,init=mix.to_dict())
    root=tmp_path/'trial';root.mkdir();(root/('qso_00_'+mode)).mkdir()
    write_json(root/'identity.json',{'fixture':True})
    write_json(root/'options.json',dict(prepared_run=str(source),nuisance_bundle='unused',
        max_iter=4,block_iterations=2,tolerance=1e-4,roundoff_tolerance=1e-9))
    np.savez(root/'qso_00_rows.npz',train=np.arange(n),density=np.array([0,1]))
    monkeypatch.setattr(trial.PSFMultiSurveyBaseline,'load',lambda _:SimpleNamespace(model=None))
    monkeypatch.setattr(trial,'conditional_values',lambda *args:np.array([0.,1.]))
    import time
    assert not trial.fit_block(task,mode,str(root),2,time.time()+60)['budget_exhausted']
    assert not trial.fit_block(task,mode,str(root),4,time.time()+60)['budget_exhausted']
    saved=trial.read(root/('qso_00_'+mode)/'checkpoint.json')
    factory=ProjectedBatchFactory(source/'qso',np.arange(n),7)
    expected=mix; history=[]
    for _ in range(4):
        stats=accumulate_projected(lambda:factory(0,n),expected,{0:trial.operator(layout,'qso')})
        assert stats[-1]==n
        history.append(stats[3]/n)
        expected=trial.update_from_statistics(expected,stats,.001,mode)
    fitted=GaussianMixture.from_dict(saved['mixture'])
    np.testing.assert_allclose(fitted.covs,expected.covs,atol=1e-12)
    np.testing.assert_allclose(fitted.means,expected.means,atol=1e-12)
    np.testing.assert_allclose(saved['history'],history,atol=1e-12)
    assert saved['iteration']==4
    # Reusing a finished block must neither fit again nor duplicate history.
    trial.fit_block(task,mode,str(root),4,time.time()+60)
    assert trial.read(root/('qso_00_'+mode)/'checkpoint.json')==saved
    saved['fit_source_hash']='bad';write_json(root/('qso_00_'+mode)/'checkpoint.json',saved)
    with pytest.raises(ValueError,match='identity mismatch'):
        trial.fit_block(task,mode,str(root),4,time.time()+60)
