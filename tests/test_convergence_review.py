import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qso_pcolor.convergence_review import (
    audit_fit, checked_rows, compare_probes, diagnose_step, _run_fit,
    progress_report, verify_parent_identity,
)
from qso_pcolor.full_sample import TrainingRows, write_json, file_hash
from qso_pcolor.full_training import make_source
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import BandLuptitudeTransform
from qso_pcolor.streaming_xd import fit_xd_batches


def test_audit_distinguishes_decline_slow_progress_and_numerical_collapse():
    mix = GaussianMixture(np.ones(1), np.zeros((1,2)), np.eye(2)[None], ('g','r'))
    r = dict(history=np.linspace(-2,-1,50).tolist(), mixture=mix.to_dict(),
             n_fit=100, selected_k=1, n_iter=50, converged=False)
    assert audit_fit(r, 1e-5, 50)['action'] == 'continue'
    r['history'].reverse()
    a = audit_fit(r, 1e-5, 50)
    assert a['action'] == 'hold_for_diagnosis' and a['material_negative_steps'] == 49
    r['history'].reverse(); r['mixture']['covs'][0][0][0] = 1e-20
    assert audit_fit(r, 1e-5, 50)['numeric_ok'] is False


def test_fixed_rows_continue_without_refitting_or_reading_held_out_roles(tmp_path):
    rng = np.random.default_rng(483)
    root=tmp_path/'input'; root.mkdir(); parent=tmp_path/'parent'; parent.mkdir()
    out=tmp_path/'out'; out.mkdir(); bands=('sdss:g','sdss:r'); n=55
    role=np.array([0]*27+[1]*12+[2]*8+[3]*8,dtype='u1')
    flux=rng.normal(10,1,(n,2)); variance=np.full((n,2),.1)
    # Held-out values are deliberately unusable; reading them would fail fitting.
    flux[role>=2]=np.nan
    for key,a in dict(flux=flux,variance=variance,role=role,eligible=np.ones(n,bool),zspec=np.ones(n)).items():
        np.save(root/(key+'.npy'),a)
    data=TrainingRows(root,bands); transform=BandLuptitudeTransform(bands,np.ones(2))
    source=SimpleNamespace(transform=transform,reference_priority=bands)
    cfg=dict(final_shape_roles=['fit','select'],fit_batch_size=7,tol=0.,regularization=.001)
    rows=data.select(('fit','select')); batches=make_source(data,rows,source,cfg)
    x=np.concatenate([b[0] for b in batches()])
    init=GaussianMixture(np.ones(1),x.mean(axis=0)[None],np.eye(2)[None],bands)
    first=fit_xd_batches(batches,init=init,expected_rows=len(rows),max_iter=2,tol=0.,regularization=.001)
    record=dict(selected_k=1,n_fit=len(rows),history=first.history,n_iter=first.n_iter,
        converged=False,mixture=first.mixture.to_dict(),mean_training_log_density=first.mean_loglike,
        fit_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    write_json(parent/'qso_00.json',record); digest=file_hash(parent/'qso_00.json')
    # Use qso directory expected by worker without copying arrays.
    (tmp_path/'qso').symlink_to(root,target_is_directory=True)
    options=dict(parent=str(parent),additional_iterations=3,workers=2,task_rows=14,
        diagnostic_seed=51,diagnostic_rows=19,magnitude_quantiles=3,tail_fraction=.1)
    _run_fit('qso_00',(0.,2.),tmp_path,source,cfg,options,out,'continue')
    saved=json.loads((out/'qso_00.json').read_text())
    uninterrupted=fit_xd_batches(batches,init=init,expected_rows=len(rows),max_iter=5,tol=0.,regularization=.001)
    np.testing.assert_allclose(saved['mixture']['means'],uninterrupted.mixture.means)
    np.testing.assert_allclose(saved['history'],uninterrupted.history)
    assert saved['history'][:2]==record['history'] and saved['n_iter']==5
    assert file_hash(parent/'qso_00.json')==digest
    with np.load(out/'qso_00.probe.npz') as probe:
        assert np.isin(probe['rows'],rows).all()
    result_hash=file_hash(out/'qso_00.json')
    _run_fit('qso_00',(0.,2.),tmp_path,source,cfg,options,out,'continue')
    assert file_hash(out/'qso_00.json')==result_hash
    bad=dict(record,fit_rows_sha256='wrong')
    with pytest.raises(ValueError,match='identities'):
        checked_rows(data,(0.,2.),cfg,bad)


def test_counterfactual_identifies_additive_ridge_likelihood_loss():
    x=np.array([[-1.],[-.5],[.5],[1.]])
    cov=np.zeros((4,1,1));obs=np.ones((4,1),bool);w=np.ones(4)
    mix=GaussianMixture(np.ones(1),np.zeros((1,1)),np.array([[[.625]]]),('x',))
    before=mix.to_dict()
    result=diagnose_step(lambda:iter([(x,cov,obs,w)]),mix,{'regularization':.1},4)
    assert result['updates']['configured']['change']<0
    assert result['updates']['zero_ridge_counterfactual']['change']==pytest.approx(0,abs=1e-12)
    assert mix.to_dict()==before


def test_probe_strata_and_weighted_additional_work(tmp_path):
    before=dict(conditional=np.array([-8.,-1.,-20.,-3.]),joint=np.array([-9.,-2.,-23.,-5.]),
        anchor=np.array([0,0,1,1]),mask=np.array([3,3,6,6]),magnitude=np.array([19.,21.,18.,22.]))
    after={k:v.copy() for k,v in before.items()};after['conditional']+=np.array([2.,0.,1.,0.])
    r=compare_probes(before,after,dict(magnitude_quantiles=2,tail_fraction=.1))
    assert r['all']['max_abs_change']==2 and set(r['by_band_mask'])=={'3','6'}
    after['mask'][0]=7
    with pytest.raises(ValueError,match='changed'):
        compare_probes(before,after,dict(magnitude_quantiles=2,tail_fraction=.1))
    fits={'qso_00':dict(action='continue',rows=100,k=2,iteration=300),
          'background':dict(action='continue',rows=200,k=4,iteration=300),
          'qso_01':dict(action='hold_for_diagnosis',rows=999,k=20,iteration=300)}
    write_json(tmp_path/'qso_00.status.json',dict(state='completed',iteration=350))
    write_json(tmp_path/'background.status.json',dict(state='running',iteration=325))
    p=progress_report(tmp_path,dict(fits=fits),dict(additional_iterations=100))
    assert p['overall_percent']==pytest.approx(100*(100*2*50+200*4*25)/(100*2*50+200*4*100))
    with pytest.raises(ValueError,match='identity'):
        verify_parent_identity({'config':{'tol':1e-5}},{'config':{'tol':1e-4}})
