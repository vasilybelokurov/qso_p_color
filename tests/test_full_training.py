import json
from pathlib import Path
import numpy as np
import pytest

from qso_pcolor.full_sample import assemble_qso, TrainingRows, ROLES, write_json
from qso_pcolor.full_training import validate_roles, fit_population
from qso_pcolor.multisurvey import BandLuptitudeTransform
from qso_pcolor.multisurvey_data import SURVEYS, band_labels, match_catalogue
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.streaming_xd import initialise_batches


def native(survey, ids):
    ids = np.array(ids); n = len(ids); spec = SURVEYS[survey]
    raw = dict(idx=np.arange(n), object_id=np.array(['a','b','c','d'])[ids],
               ra=10.+ids, dec=np.zeros(n), match_count=np.ones(n, int),
               match_sep_arcsec=np.zeros(n))
    for b in spec.bands:
        raw['value_'+b] = np.full(n, 20. if spec.kind=='mag' else -1.)
        raw['error_'+b] = np.ones(n)
    for key in spec.extra:
        raw[key] = np.zeros(n, int)
        if 'nobs' in key or 'nframes' in key or 'ngood' in key or key=='clean': raw[key][:]=1
    if survey=='decals': raw['release'][:]=9010; raw['match_count'][:]=2
    if survey=='allwise': raw['cc_flags']=np.full(n,'0000')
    return raw


def test_local_assembly_preserves_partial_results_roles_and_negative_flux(tmp_path, monkeypatch):
    q=tmp_path/'q';s=tmp_path/'s';q.mkdir();s.mkdir()
    targets=dict(ra=10.+np.arange(4),dec=np.zeros(4),object_id=np.array(list('abcd')),
        master_index=np.arange(4),zspec=np.ones(4),redshift_conflict=np.zeros(4,bool),extended_duplicate_group=np.zeros(4,bool))
    np.savez(q/'targets.npz',**targets)
    np.savez(s/'qso_roles.npz',object_id=targets['object_id'],master_index=targets['master_index'],role=np.array(ROLES))
    cfg=dict(surveys=list(SURVEYS),batch_size=2,clean=True,vhs_bad_bits=0,match_radius_arcsec={k:1. for k in SURVEYS})
    write_json(q/'provenance.json',{'config':cfg});calls=[]
    def read(root,survey,targets,lo,hi,radius,**kwargs):
        assert kwargs=={'cache_only':True}
        calls.append((survey,lo));return native(survey,np.arange(lo,hi))
    monkeypatch.setattr('qso_pcolor.full_sample.read_acquired_batch',read)
    for survey in SURVEYS:
        partial=survey in ('nsc','skymapper','vhs')
        write_json(q/f'progress_{survey}.json',dict(processed=2 if partial else 4))
        if partial:
            p=q/'gap_acquisition'/survey;p.mkdir(parents=True)
            raw=native(survey,[1,2]);raw['target_index']=np.array([1,2])
            np.savez(p/'reuse.npz',**raw)
    r=assemble_qso(q,s,tmp_path/'out',batch_size=2)
    d=TrainingRows(tmp_path/'out',band_labels())
    assert len(d.arrays['flux'])==4
    assert r['association']['decals']['multiple_targets']==0  # never use cross-release counts
    assert r['association']['nsc']['known_results']==3
    assert ('nsc',2) not in calls
    col=band_labels().index('nsc:g')
    assert np.isfinite(d.arrays['flux'][2,col]) and np.isnan(d.arrays['flux'][3,col])
    assert (d.arrays['flux'][:,band_labels().index('sdss:g')]<0).all()
    for i,role in enumerate(ROLES):assert d.select((role,)).tolist()==[i]


def test_cache_only_does_not_query_missing_data(tmp_path,monkeypatch):
    import sqlutilpy
    monkeypatch.setattr(sqlutilpy,'local_join',lambda *a,**k:pytest.fail('unexpected WSDB query'))
    with pytest.raises(FileNotFoundError,match='local photometry'):
        match_catalogue('sdss',np.array([10.]),np.array([0.]),tmp_path,radius_arcsec=1.,cache_only=True)


def test_frozen_roles_reject_calibration_leakage():
    cfg=json.loads(Path('configs/full_sample_training.json').read_text())
    validate_roles(cfg)
    cfg['final_shape_roles'].append('calib')
    with pytest.raises(ValueError,match='frozen'):validate_roles(cfg)


def test_full_population_fit_uses_all_rows_and_excludes_calibration_test(tmp_path):
    from types import SimpleNamespace
    rng=np.random.default_rng(440)
    bands=('sdss:g','sdss:r');n=53
    role=np.array([0]*21+[1]*12+[2]*9+[3]*11,dtype='u1')
    for key,a in dict(role=role,eligible=np.ones(n,bool),flux=rng.normal(10,1,(n,2)),variance=np.full((n,2),.1),zspec=np.ones(n)).items():np.save(tmp_path/(key+'.npy'),a)
    data=TrainingRows(tmp_path,bands);transform=BandLuptitudeTransform(bands,np.ones(2))
    source=SimpleNamespace(transform=transform,reference_priority=bands)
    cfg=dict(fit_roles=['fit'],selection_roles=['select'],final_shape_roles=['fit','select'],
        background_k_candidates=[1,2],fit_batch_size=7,selection_max_iter=2,max_iter=3,tol=0.,regularization=.001)
    initial={k:initialise_batches(lambda:data.batches(data.select(('fit',)),transform,7),k,labels=bands,seed=55+k) for k in [1,2]}
    out=tmp_path/'fit';out.mkdir()
    record=fit_population('background',data,None,source,cfg,out,initial)
    assert record['n_fit']==33 and record['role_counts']==dict(fit=21,select=12,calib=0,test=0)
    assert all(r['fit_rows']==21 and r['selection_rows']==12 for r in record['selection_records'].values())
    assert json.loads((out/'background.final.checkpoint.json').read_text())['rows']==33


def test_spatial_precomputed_log_prob_matches_full_covariance_path():
    from qso_pcolor.joint_spatial import fit_joint_spatial_weights,fit_joint_spatial_log_prob
    from qso_pcolor.spatial import component_log_prob
    rng=np.random.default_rng(443);x=rng.normal(size=(30,2));cov=np.tile(np.eye(2)*.03,(30,1,1));obs=np.ones_like(x,bool);obs[::3,1]=False
    mix=GaussianMixture(np.array([.3,.7]),np.array([[-1.,0.],[1.,0.]]),np.tile(np.eye(2),(2,1,1)),('x','y'))
    l=np.repeat([0.,180.],15);b=np.full(30,45.);w=np.ones(30)
    kw=dict(nside=4,nside_parent=2,n0=10,max_iter=30,tol=1e-7,meta={})
    a=fit_joint_spatial_weights(mix,x,cov,obs,l,b,w,**kw)
    lp=np.concatenate([component_log_prob(mix,x[i:i+7],cov[i:i+7],obs[i:i+7]) for i in range(0,30,7)])
    c=fit_joint_spatial_log_prob(mix,lp,l,b,w,**kw)
    np.testing.assert_allclose(a.evaluate(l,b)[0],c.evaluate(l,b)[0],rtol=1e-13)


def test_candidate_training_roundtrip_and_resume_preserves_active_pointer(tmp_path):
    from qso_pcolor.full_sample import file_hash
    from qso_pcolor.full_training import train,preflight
    from qso_pcolor.multisurvey import MultiSurveyModel
    from qso_pcolor.qso_model import SlicedColourRedshiftModel
    cfg=json.loads(Path('configs/full_sample_training.json').read_text())
    cfg.update(input_root=str(tmp_path/'inputs'),output_root=str(tmp_path/'fits'),source_pointer=str(tmp_path/'model/current'),
        background_k_candidates=[1,2],qso_k_candidates=[1,2],fit_batch_size=7,selection_max_iter=1,max_iter=2)
    cfg['spatial'].update(max_iter=5)
    bands=('sdss:g','sdss:r');rng=np.random.default_rng(41);source_dir=tmp_path/'model/base';source_dir.mkdir(parents=True)
    mix=GaussianMixture(np.ones(1),np.full((1,2),20.),np.tile(np.eye(2),(1,1,1)),bands)
    qso=SlicedColourRedshiftModel(np.array([.15,.25]),[mix,mix],np.array([10,10]),'test',bands)
    model=MultiSurveyModel(qso,mix,BandLuptitudeTransform(bands,np.ones(2)),bands,np.array([[0.,30.],[0.,30.]]))
    model.save(source_dir/'model.json');write_json(Path(cfg['source_pointer']),dict(bundle='base'))
    pointer_hash=file_hash(Path(cfg['source_pointer']))
    root=tmp_path/'inputs/assembled';root.mkdir(parents=True);report={}
    for pop in ['qso','stars']:
        p=root/pop;p.mkdir();n=40
        role=np.repeat(np.arange(4),10).astype('u1')
        for key,a in dict(role=role,eligible=np.ones(n,bool),flux=rng.normal(10,1,(n,2)),variance=np.full((n,2),.1),
                          zspec=np.full(n,.2),l=np.tile([0.,180.],20),b=np.full(n,45.)).items():np.save(p/(key+'.npy'),a)
        report[pop]=dict(rows=n,bands=list(bands),roles={r:dict(eligible=10) for r in ROLES})
    write_json(root/'report.json',report)
    files={str(p.relative_to(root)):file_hash(p) for p in root.glob('*/*.npy')};files['report.json']=file_hash(root/'report.json')
    write_json(root/'manifest.json',dict(identity=dict(config=cfg),complete=True,files=files))
    write_json(tmp_path/'inputs/current.json',dict(directory=str(root),manifest_sha256=file_hash(root/'manifest.json')))
    check=preflight(cfg);assert check['unique_qso_final_rows']==20 and check['slice_row_sum']==40
    out=train(cfg)
    result=MultiSurveyModel.load(out/'model.json')
    assert result.spatial_background is not None
    assert result.qso.n_train.tolist()==[20,20]
    assert result.meta['background']['n_fit']==20
    assert result.meta['caps'] is None
    assert not json.loads((out/'completion.json').read_text())['release_ready']
    assert file_hash(Path(cfg['source_pointer']))==pointer_hash
    previous=file_hash(out/'background.json')
    assert train(cfg)==out and file_hash(out/'background.json')==previous
    # Refit both synthetic slices concurrently from identical initial mixtures.
    # This exercises real spawned workers and the existing final assembly path.
    from qso_pcolor.parallel_training import train_parallel
    expected = [m.to_dict() for m in result.qso.mixtures]
    for p in out.glob('qso_[0-9][0-9]*.json'):
        p.unlink()
    assert train_parallel(cfg, workers=2) == out
    parallel = MultiSurveyModel.load(out/'model.json')
    assert [m.to_dict() for m in parallel.qso.mixtures] == expected
    assert file_hash(out/'background.json') == previous
    assert file_hash(Path(cfg['source_pointer'])) == pointer_hash
    execution = json.loads((out/'parallel_execution.json').read_text())
    assert execution['state'] == 'completed'
    assert sorted(execution['slice_order']) == [0, 1]
    from qso_pcolor.sky_acquisition import acquisition_lock
    with acquisition_lock(out/'worker.lock'):
        with pytest.raises(RuntimeError, match='already holds'):
            train_parallel(cfg, workers=2)
