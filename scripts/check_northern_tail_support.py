#!/usr/bin/env python
"""Check the failed colour point against actual northern shape-training rows."""
from pathlib import Path
import json,numpy as np
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.full_sample import write_json, ROLES
diagnostic=json.loads(Path('configs/northern_tail_diagnostic.json').read_text())
cfg=json.loads(Path(diagnostic['release_config']).read_text());model=MultiSurveyModel.load(Path(cfg['candidate'])/'model.json')
bands=tuple(diagnostic['probe_bands']);idx=[model.transform.bands.index(b) for b in bands]
probe=np.array(diagnostic['probe_luptitudes']);colour=np.array([probe[0]-probe[1],probe[1]-probe[2]])
report=dict(probe_luptitudes=probe.tolist(),bands=bands,reference_window=diagnostic['reference_window'],populations={})
for kind in ('qso','stars'):
 root=Path(cfg['inputs'])/kind;flux=np.load(root/'flux.npy',mmap_mode='r');variance=np.load(root/'variance.npy',mmap_mode='r');role=np.load(root/'role.npy',mmap_mode='r')
 n=0;nearest=None;distance=np.inf
 for lo in range(0,len(flux),diagnostic['support_batch_size']):
  phot=Photometry(flux[lo:lo+diagnostic['support_batch_size'],idx],variance[lo:lo+diagnostic['support_batch_size'],idx],bands);obs=phot.observed
  x=22.5-2.5/np.log(10)*(np.arcsinh(phot.flux/(2*model.transform.softening[idx]))+np.log(model.transform.softening[idx]))
  use=obs.all(axis=1)&np.isin(role[lo:lo+len(x)],[ROLES.index(r) for r in diagnostic['roles']])&(x[:,1]>=diagnostic['reference_window'][0])&(x[:,1]<diagnostic['reference_window'][1])
  rr=np.flatnonzero(use);n+=len(rr)
  if len(rr):
   c=np.column_stack((x[rr,0]-x[rr,1],x[rr,1]-x[rr,2]));ds=np.linalg.norm(c-colour,axis=1);j=np.argmin(ds)
   if ds[j]<distance:distance=float(ds[j]);nearest=dict(row=int(lo+rr[j]),luptitudes=x[rr[j]].tolist(),colours=c[j].tolist())
 report['populations'][kind]=dict(bright_training_rows=n,nearest_colour_distance_mag=distance,nearest=nearest)
print(json.dumps(report,indent=2));write_json(Path('docs/NORTHERN_TAIL_SUPPORT_2026-09-30.json'),report)
