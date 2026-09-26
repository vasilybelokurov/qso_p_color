#!/usr/bin/env python
"""Fit a declared background marginal using existing training fields only.

The joint quasar fit, transform, and full-survey background stay unchanged.
Component selection uses the original internal background selection fields.
The final spatial holdout and spectroscopic comparison never enter fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train_multisurvey_model import sample_patterns
from validate_multisurvey import auc
from qso_pcolor.background import BackgroundColourModel
from qso_pcolor.features import RelativeFluxTransform, deredden
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import MultiSurveyModel, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.xd import _init_mixture, fit_xd


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def diagnose_background(cfg, model):
    """Exchange field terms in a common density coordinate, without refitting.

    Convert the new conditional luptitude density to the original two flux
    ratios using the measured-flux Jacobian at fixed r. This is a diagnostic
    exchange, not an alternative shipped probability model.
    """
    saved = dict(np.load(cfg['comparison_sample']))
    manifest = json.loads(str(saved['signature']))
    if manifest['hashes']['new_model'] != digest(cfg['base_model']):
        raise ValueError('diagnostic scores do not belong to the joint-only base model')
    background = BackgroundColourModel.load(cfg['original_background'])
    if manifest['hashes']['old_background'] != digest(cfg['original_background']):
        raise ValueError('original background differs from the diagnostic score cache')
    n = len(saved['z'])
    flux, var = deredden(saved['flux'],saved['ivar'],saved['transmission'])
    old_features = RelativeFluxTransform()(
        np.column_stack([flux,np.full((n,2),np.nan)]),
        np.column_stack([var,np.full((n,2),np.inf)]),('g','r','z','w1','w2'))
    old_b = background.log_prob(old_features.x,old_features.cov,old_features.ref_mag,
                               saved['l'],saved['b'],observed=old_features.observed)
    labels = tuple(cfg['bands'])
    if labels != ('decals_dr9_south:g','decals_dr9_south:r','decals_dr9_south:z'):
        raise ValueError('this original-model diagnostic is defined for southern grz only')
    features = model.transform(Photometry(saved['flux'],1/saved['ivar'],labels))
    indices = [features.labels.index(b) for b in labels]
    new_b = model.background_log_prob(features.x,features.cov,features.observed,indices[1])
    du_df = (2.5/np.log(10))/np.hypot(saved['flux'][:,[0,2]],2*model.transform.softening[np.array(indices)[[0,2]]])
    df_dratio = saved['flux'][:,1,None]*saved['transmission'][:,[0,2]]/saved['transmission'][:,1,None]
    new_b += np.log(du_df*df_dratio).sum(axis=1)
    old_q = saved['old_log_bf']+old_b
    new_q = saved['new_log_bf']+new_b
    scores = {'original_Q_original_B':old_q-old_b,'joint_Q_joint_B':new_q-new_b,
              'original_Q_joint_B':old_q-new_b,'joint_Q_original_B':new_q-old_b}
    q = saved['kind']=='QSO'
    result = {name:{kind:auc(score[q],score[saved['kind']==kind]) for kind in ['STAR','GALAXY']}
              for name,score in scores.items()}
    result['hashes'] = {key:digest(cfg[key]) for key in ['base_model','original_background','comparison_sample']}
    result['interpretation'] = ('Diagnostic exchange after converting new conditional luptitude densities into original dereddened flux-ratio coordinates at fixed measured r; excludes WISE. Models retain their different first-order noise approximations. The exchange locates the discrepancy; mixture size is chosen by separate internal field likelihood, and swapped models are never deployed.')
    Path(cfg['diagnostic_report']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/background_marginal_lsw_grz.json'))
    parser.add_argument('--diagnose-only',action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    model = MultiSurveyModel.load(cfg['base_model'])
    labels = tuple(cfg['bands'])
    if any(tuple(m.labels) == labels for m in model.background_marginals):
        raise ValueError('the base model already has a marginal for these bands')
    if model.background_marginals and not cfg.get('append', False):
        raise ValueError("base model already has marginals; set \"append\": true to add another")
    if args.diagnose_only:
        diagnose_background(cfg,model)
        return
    settings = model.meta['settings']; original = settings['config']
    hashes = {key:digest(cfg[key]) for key in ['base_model','background_sample']}
    base_run = model.meta.get('joint_run_id', model.meta['run_id'])
    if hashes['background_sample'] != settings['background_sha256']:
        raise ValueError('background cache differs from the base-model training data')
    identity = dict(config=cfg, hashes=hashes)
    run = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(cfg['cache_dir'])/run; root.mkdir(parents=True, exist_ok=True)
    (root/'settings.json').write_text(json.dumps(identity,indent=2))
    data = dict(np.load(cfg['background_sample']))
    phot = Photometry(data['flux'],data['variance'],tuple(data['bands']))
    rows, weights = sample_patterns(phot.observed,
        ~data['held'] & (phot.observed.sum(1)>=original['min_bands']),
        settings['max_background_fit'], original['min_band_training'],
        np.random.default_rng(original['seed']+10))
    indices = np.array([phot.bands.index(b) for b in labels])
    rows = rows[phot.observed[rows][:,indices].all(axis=1)]
    features = model.transform(phot.subset(rows))
    x = features.x[:,indices]; cov = features.cov[:,indices,:][:,:,indices]
    weights = weights[rows]
    validation = np.isin(data['field'][rows],model.meta['background_selection_fields'])
    train = ~validation
    anchor = labels.index(cfg['reference_band'])
    records = []
    for k in cfg['k_candidates']:
        path = root/f'k_{k}.json'
        if path.exists():
            record = json.loads(path.read_text())
        else:
            init = _init_mixture(x[train],np.ones_like(x[train],bool),k,np.random.default_rng(cfg['seed']+k))
            fit = fit_xd(x[train],cov[train],weights=weights[train],init=init,labels=labels,
                         seed=cfg['seed'],max_iter=cfg['max_iter'],tol=cfg['tol'],regularization=cfg['regularization'])
            lp = conditional_log_prob(fit.mixture,x[validation],cov[validation],
                                      np.ones_like(x[validation],bool),anchor)
            record = dict(k=k,selection_log_density=float(np.average(lp,weights=weights[validation])),
                          n_iter=fit.n_iter,converged=fit.converged,mixture=fit.mixture.to_dict())
            path.write_text(json.dumps(record))
        records.append(record)
        print({key:v for key,v in record.items() if key!='mixture'},flush=True)
    selected = max(records,key=lambda r:r['selection_log_density'])
    path = root/'final.json'
    if path.exists():
        final = json.loads(path.read_text())
    else:
        fit = fit_xd(x,cov,weights=weights,init=GaussianMixture.from_dict(selected['mixture']),
                     labels=labels,seed=cfg['seed'],max_iter=cfg['max_iter'],tol=cfg['tol'],
                     regularization=cfg['regularization'])
        final = dict(mixture=fit.mixture.to_dict(),n_iter=fit.n_iter,converged=fit.converged,
                     convergence_tail=fit.history[-5:])
        path.write_text(json.dumps(final))
    manifest = dict(run_id=run,**identity,n_fit=len(rows),n_selection=int(validation.sum()),
                    selected_k=selected['k'],selection=[{k:v for k,v in r.items() if k!='mixture'} for r in records],
                    final={k:v for k,v in final.items() if k!='mixture'},
                    selection_fields=model.meta['background_selection_fields'],
                    routing='Use only when all observed bands are contained in the fitted band set; otherwise use the joint background.')
    model.background_marginals = model.background_marginals + (GaussianMixture.from_dict(final['mixture']),)
    fitted = list(model.meta.get('background_marginals_fitted',
                                 [model.meta['background_marginal']] if 'background_marginal' in model.meta else []))
    model.meta = {**model.meta,'joint_run_id':base_run,'run_id':run,'background_marginal':manifest,
                  'background_marginals_fitted':fitted + [manifest]}
    model.save(cfg['output_model'])
    print('Saved candidate',cfg['output_model'],flush=True)


if __name__ == '__main__':
    main()
