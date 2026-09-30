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
