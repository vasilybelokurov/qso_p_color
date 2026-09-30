import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from qso_pcolor.full_sample import write_json, file_hash
from qso_pcolor.gaussmix import GaussianMixture


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('run_unified_pilot')


def fixture_run(root):
    root.mkdir(exist_ok=True)
    for folder in ('qso','stars','fits','progress'):
        (root/folder).mkdir(exist_ok=True)
    cfg=dict(training_cells=None,test_cells=[],fit_roles=[0,1],z_half_width=.1,
        batch_size=13,max_iter=4,tol=0.,regularization=.001,qso_workers=4,
        stellar_workers=4,stellar_task_rows=23,pilot=False)
    rng=np.random.default_rng(855)
    n=91
    for kind in ('qso','stars'):
        y=rng.normal(size=(n,2));o=rng.random((n,2))>.2;o[:,0]=True;y[~o]=np.nan
        for key,value in dict(y=y,noise=np.full((n,2),.02),observed=o,source_row=np.arange(n),
            cell=np.arange(n)%9,role=np.arange(n)%4,zspec=np.ones(n)).items():
            np.save(root/kind/(key+'.npy'),value)
    init=GaussianMixture(np.array([.4,.6]),np.array([[0.,0.],[1.,1.]]),np.tile(np.eye(2),(2,1,1)))
    op=dict(matrix=np.eye(2).tolist(),offset=[0.,0.],covariance=(np.eye(2)*.01).tolist())
    write_json(root/'layout.json',dict(operators=dict(stars=op,qso=op)))
    write_json(root/'identity.json',dict(test='fixed identity'))
    pointer=root/'active';pointer.write_text('unchanged');cfg['active_pointer']=str(pointer)
    write_json(root/'prepared.json',dict(active_pointer_hash=file_hash(pointer)))
    n_fit=int(np.isin(np.arange(n)%4,[0,1]).sum())
    tasks=[dict(name='stars_00',kind='stars',n=n_fit,k=2,init=init.to_dict())]
    tasks += [dict(name=f'qso_{i:02d}',kind='qso',n=n_fit,k=2,z=1.,init=init.to_dict()) for i in range(4)]
    write_json(root/'tasks.json',tasks)
    return cfg,tasks


def test_disk_checkpoint_resume_matches_uninterrupted_and_rejects_changed_identity(tmp_path,runner,monkeypatch):
    cfg,tasks=fixture_run(tmp_path/'interrupted');task=tasks[1];root=tmp_path/'interrupted'
    original=runner.fit_projected
    def interrupted(*args,**kwargs):
        progress=kwargs['progress']
        def stop(it,*values):
            progress(it,*values)
            if it==2:raise InterruptedError
        kwargs['progress']=stop
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'fit_projected',interrupted)
    with pytest.raises(InterruptedError):runner.fit_task(task,cfg,root)
    saved=json.loads((root/'progress/qso_00_checkpoint.json').read_text())
    assert saved['iteration']==len(saved['history'])==2
    monkeypatch.setattr(runner,'fit_projected',original)
    runner.fit_task(task,cfg,root)
    got=json.loads((root/'fits/qso_00.json').read_text())
    assert got['resumed_iteration']==2 and got['n_iter']==4
    other_cfg,other_tasks=fixture_run(tmp_path/'continuous')
    runner.fit_task(other_tasks[1],other_cfg,tmp_path/'continuous')
    expected=json.loads((tmp_path/'continuous/fits/qso_00.json').read_text())
    assert got['mixture']==expected['mixture'] and got['history']==expected['history']
    (root/'fits/qso_00.json').unlink()
    write_json(root/'identity.json',dict(test='changed'))
    with pytest.raises(ValueError,match='identity'):runner.fit_task(task,cfg,root)


def test_four_plus_four_launcher_completes_both_populations_and_preserves_pointer(tmp_path,runner):
    full=importlib.import_module('train_unified_full')
    cfg,tasks=fixture_run(tmp_path)
    full.validate_config(cfg)
    full.run_fits(tmp_path,cfg)
    report=json.loads((tmp_path/'training_complete.json').read_text())
    assert report['tasks']==5
    for task in tasks:
        result=json.loads((tmp_path/'fits'/(task['name']+'.json')).read_text())
        assert result['n']==task['n'] and result['n_iter']==4
    assert (tmp_path/'active').read_text()=='unchanged'
    execution=json.loads((tmp_path/'execution.json').read_text())
    assert execution['qso_workers']==execution['stellar_workers']==4
    assert execution['state']=='completed'
    hashes={p.name:file_hash(p) for p in (tmp_path/'fits').glob('*.json')}
    full.run_fits(tmp_path,cfg)
    assert hashes=={p.name:file_hash(p) for p in (tmp_path/'fits').glob('*.json')}


def test_full_launcher_refuses_pilot_footprint(tmp_path,runner):
    full=importlib.import_module('train_unified_full')
    cfg,_=fixture_run(tmp_path);cfg['training_cells']=[22,63,169]
    with pytest.raises(ValueError,match='all sky'):full.validate_config(cfg)


def stopping_run(root, min_gain=.02):
    cfg, tasks = fixture_run(root)
    cfg.update(max_iter=8, covariance_update='map', predictive_stopping=dict(block_iterations=2, min_gain=min_gain))
    (root/'stopping').mkdir(exist_ok=True)
    rows = np.flatnonzero(np.arange(91) % 4 == 3)          # role 3 only; never training rows
    for task in tasks:
        np.savez(root/'stopping'/(task['name']+'.npz'), rows=rows, anchors=np.zeros(len(rows), int))
    return cfg, tasks


@pytest.mark.parametrize('values,expected_iteration,reason,fitted', [
    ([1., .5, .4], 0, 'predictive_plateau', 2),                 # worse at once: keep the warm start
    ([1., 2., 3., 3.01, 9.], 6, 'predictive_plateau', 6),       # small gain stops; best so far wins
    ([1., 2., 3., 4., 5.], 8, 'iteration_limit', 8)])
def test_predictive_stopping_decisions_and_best_checkpoint(tmp_path, runner, monkeypatch, values, expected_iteration, reason, fitted):
    cfg, tasks = stopping_run(tmp_path)
    scripted = iter(values)
    monkeypatch.setattr(runner, 'stopping_density', lambda *a: next(scripted))
    runner.fit_task(tasks[1], cfg, tmp_path)
    got = json.loads((tmp_path/'fits/qso_00.json').read_text())
    assert (got['selected_iteration'], got['stop_reason'], got['n_iter']) == (expected_iteration, reason, fitted)
    assert [e['iteration'] for e in got['stopping_evaluations']] == list(range(0, fitted+1, 2))
    if expected_iteration == 0:
        assert got['mixture'] == tasks[1]['init']
    assert np.isfinite(got['mean_loglike'])


def test_predictive_stopping_resume_mid_block_and_after_decision_match_uninterrupted(tmp_path, runner, monkeypatch):
    whole_cfg, whole_tasks = stopping_run(tmp_path/'whole', min_gain=-1e9)
    runner.fit_task(whole_tasks[1], whole_cfg, tmp_path/'whole')
    expected = json.loads((tmp_path/'whole/fits/qso_00.json').read_text())
    cfg, tasks = stopping_run(tmp_path/'cut', min_gain=-1e9); root = tmp_path/'cut'
    original = runner.fit_projected

    def interrupted(*args, **kwargs):
        progress = kwargs['progress']
        def stop(it, *values):
            progress(it, *values)
            if it == 3: raise InterruptedError
        return original(*args, **{**kwargs, 'progress': stop})
    monkeypatch.setattr(runner, 'fit_projected', interrupted)
    with pytest.raises(InterruptedError): runner.fit_task(tasks[1], cfg, root)
    monkeypatch.setattr(runner, 'fit_projected', original)
    runner.fit_task(tasks[1], cfg, root)
    got = json.loads((root/'fits/qso_00.json').read_text())
    for key in ('mixture', 'history', 'stopping_evaluations', 'selected_iteration', 'stop_reason', 'n_iter'):
        assert got[key] == expected[key], key
    # A crash after the decision is recorded but before the result is written.
    (root/'fits/qso_00.json').unlink()
    runner.fit_task(tasks[1], cfg, root)
    again = json.loads((root/'fits/qso_00.json').read_text())
    for key in ('mixture', 'history', 'stopping_evaluations', 'selected_iteration', 'stop_reason', 'n_iter', 'mean_loglike'):
        assert again[key] == got[key], key


def test_stopping_density_matches_direct_conditional_density(tmp_path, runner):
    from qso_pcolor.multisurvey import conditional_log_prob
    from qso_pcolor.projected_xd import native_mixture
    cfg, tasks = stopping_run(tmp_path)
    data = runner.arrays(tmp_path, 'qso'); layout = json.loads((tmp_path/'layout.json').read_text())
    panel = dict(np.load(tmp_path/'stopping/qso_00.npz')); mix = GaussianMixture.from_dict(tasks[1]['init'])
    native = native_mixture(mix, *runner.operator(layout, 'qso'))
    values = []
    for r in panel['rows']:
        cov = np.diag(data['noise'][r])[None]
        values.append(conditional_log_prob(native, data['y'][r][None], cov, data['observed'][r][None], 0)[0])
    assert runner.stopping_density(mix, layout, 'qso', data, panel, 5) == pytest.approx(np.mean(values), rel=1e-12)


def test_pause_file_stops_at_a_stopping_check_and_resume_matches_uninterrupted(tmp_path, runner):
    full = importlib.import_module('train_unified_full')
    cfg, tasks = stopping_run(tmp_path/'whole', min_gain=-1e9); full.run_fits(tmp_path/'whole', cfg)
    cfg, tasks = stopping_run(tmp_path/'cut', min_gain=-1e9); root = tmp_path/'cut'
    (root/'PAUSE').write_text('pause')
    full.run_fits(root, cfg)
    execution = json.loads((root/'execution.json').read_text())
    assert execution['state'] == 'paused' and len(execution['paused_tasks']) == len(tasks)
    assert not list((root/'fits').glob('*.json')) and not (root/'training_complete.json').exists()
    for task in tasks:                                   # each paused right after its warm-start evaluation
        trace = json.loads((root/'progress'/(task['name']+'_stopping.json')).read_text())
        assert [e['iteration'] for e in trace['evaluations']] == [0] and trace['stop_reason'] is None
    (root/'PAUSE').unlink()
    full.run_fits(root, cfg)
    assert json.loads((root/'execution.json').read_text())['state'] == 'completed'
    for task in tasks:
        a = json.loads((root/'fits'/(task['name']+'.json')).read_text())
        b = json.loads((tmp_path/'whole/fits'/(task['name']+'.json')).read_text())
        for key in ('mixture', 'history', 'stopping_evaluations', 'selected_iteration', 'stop_reason', 'mean_loglike'):
            assert a[key] == b[key], (task['name'], key)
