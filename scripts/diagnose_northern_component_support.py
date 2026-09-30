#!/usr/bin/env python
"""Trace responsibility support of the components admitting failed grid points."""
import json,numpy as np
from pathlib import Path
from scipy.special import softmax
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.spatial import component_log_prob
from qso_pcolor.full_sample import write_json, ROLES
diagnostic=json.loads(Path('configs/northern_tail_diagnostic.json').read_text())
cfg=json.loads(Path(diagnostic['release_config']).read_text());m=MultiSurveyModel.load(Path(cfg['candidate'])/'model.json')
data={p.stem:np.load(p,mmap_mode='r') for p in (Path(cfg['inputs'])/'qso').glob('*.npy')}
idx=[m.transform.bands.index(b) for b in diagnostic['probe_bands']];res={}
k=diagnostic['component_index'];lo_mag,hi_mag=diagnostic['reference_window'];width=diagnostic['z_half_width']
for j in diagnostic['slice_indices']:
 z=m.qso.z_centres[j];rows=np.flatnonzero(np.isin(data['role'],[ROLES.index(r) for r in diagnostic['roles']])&data['eligible']&(data['zspec']>=z-width)&(data['zspec']<z+width))
 total=np.zeros(len(m.qso.mixtures[j].weights));north=np.zeros(len(m.qso.mixtures[j].weights));nbright=np.zeros(len(m.qso.mixtures[j].weights));nn=0;nb=0
 for lo in range(0,len(rows),diagnostic['batch_size']):
  rr=rows[lo:lo+diagnostic['batch_size']];f=m.transform(Photometry(data['flux'][rr],data['variance'][rr],m.transform.bands));r=softmax(component_log_prob(m.qso.mixtures[j],f.x,f.cov,f.observed)+np.log(m.qso.mixtures[j].weights),axis=1)
  use=f.observed[:,idx].all(axis=1);bright=use&(f.x[:,idx[1]]>=lo_mag)&(f.x[:,idx[1]]<hi_mag)
  total+=r.sum(axis=0);north+=r[use].sum(axis=0);nbright+=r[bright].sum(axis=0);nn+=int(use.sum());nb+=int(bright.sum())
 res[str(j)]=dict(z=float(z),rows=len(rows),northern_rows=nn,northern_bright_rows=nb,component=dict(index=k,weight=float(m.qso.mixtures[j].weights[k]),total_responsibility=float(total[k]),northern_responsibility=float(north[k]),northern_bright_responsibility=float(nbright[k])))
 print(j,res[str(j)],flush=True)
write_json(Path('docs/NORTHERN_COMPONENT_SUPPORT_2026-09-30.json'),res)
