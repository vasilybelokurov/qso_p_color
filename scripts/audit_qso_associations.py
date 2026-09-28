#!/usr/bin/env python
"""Audit completed QSO photometry batches while acquisition continues; never fit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from build_qso_master import sha256
from fetch_full_qso_photometry import write_json
from qso_pcolor.data import _save_npz
from qso_pcolor.legacy import is_north
from qso_pcolor.multisurvey_data import SURVEYS, match_catalogue, catalogue_photometry
from qso_pcolor.qso_acquisition import read_acquired_batch


def match_counts(survey,ra,dec,radius,cache):
    """Count all eligible associations inside the declared radius, including zero."""
    import sqlutilpy as sqlutil
    spec=SURVEYS[survey]
    sql=f"""SELECT m.idx, x.match_count FROM mytmptable m CROSS JOIN LATERAL (
        SELECT count(*) AS match_count FROM {spec.table} c
        WHERE q3c_join(m.ra,m.dec,c.{spec.ra},c.{spec.dec},{radius}/3600.)
          AND ({spec.where})) x"""
    digest=hashlib.sha256(sql.encode()+ra.tobytes()+dec.tobytes()).hexdigest()[:16]
    path=cache/f'{survey}_{digest}.npz'
    # SQL + target-coordinate digest is independent of auditor implementation.
    # Reuse these exact counts when the photometry acquisition strategy changes.
    candidates=[path] if path.exists() else sorted(cache.parent.parent.glob(f'*/counts/{path.name}'))
    for previous in candidates:
        with np.load(previous) as d:
            if str(d['query'])==sql and d['match_count'].shape==ra.shape:
                return d['match_count']
    result=sqlutil.local_join(sql,'mytmptable',(np.arange(len(ra)),ra,dec),('idx','ra','dec'),
        asDict=True,intNullVal=-1,preamb="SET jit=off; SET statement_timeout='7200s'")
    order=np.argsort(result['idx'])
    if not np.array_equal(result['idx'][order],np.arange(len(ra))):
        raise ValueError('association-count query lost input identities')
    counts=result['match_count'][order]
    if (counts<0).any(): raise ValueError('missing association counts')
    _save_npz(path,match_count=counts,query=np.array(sql))
    return counts


def shared_source_mask(ra,dec):
    matched=np.isfinite(ra+dec);shared=np.zeros(len(ra),bool)
    keys=np.rec.fromarrays([ra[matched],dec[matched]])
    _,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    shared[matched]=counts[inverse]>1
    return shared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args()
    if not 0<args.poll_seconds<=60: raise ValueError('poll interval must be in (0,60] seconds')
    root=args.cache;cfg=json.loads((root/'provenance.json').read_text())['config']
    targets=dict(np.load(root/'targets.npz'));n=len(targets['ra']);batch=cfg['batch_size']
    identity=dict(script_sha256=sha256(Path(__file__)),targets_sha256=sha256(root/'targets.npz'),
        source_provenance_sha256=sha256(root/'provenance.json'),model_fitted=False)
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'association_audit'/version;out.mkdir(parents=True,exist_ok=True)
    counts_cache=out/'counts';counts_cache.mkdir(exist_ok=True)
    write_json(out/'provenance.json',identity)
    print('Association audit cache:',out,flush=True)
    while True:
        complete=True;summary={}
        for survey in cfg['surveys']:
            finished=out/f'{survey}_complete.json'
            if finished.exists():
                summary[survey]=json.loads(finished.read_text());continue
            progress=root/f'progress_{survey}.json'
            if not progress.exists(): complete=False;continue
            done=json.loads(progress.read_text())['processed']
            if done<n: complete=False
            stats=Counter();positions=[]
            for lo in range(0,done,batch):
                hi=min(n,lo+batch)
                stamp=out/f'{survey}_{lo:07d}.json'
                # Acquisition updates progress only after this complete raw batch is cached.
                if not stamp.exists() or done==n:
                    raw=read_acquired_batch(root,survey,targets,lo,hi,cfg['match_radius_arcsec'][survey])
                    if done==n: positions.append((raw['ra'],raw['dec']))
                if not stamp.exists():
                    count=match_counts(survey,targets['ra'][lo:hi],targets['dec'][lo:hi],
                                       cfg['match_radius_arcsec'][survey],counts_cache)
                    matched=np.isfinite(raw['ra']+raw['dec'])
                    if not np.array_equal(count>0,matched):
                        raise ValueError(f'{survey}: catalogue changed between acquisition and association audit')
                    sep=raw['match_sep_arcsec'];radius=cfg['match_radius_arcsec'][survey]
                    if np.any(matched&(~np.isfinite(sep)|(sep>radius*(1+1e-8))|(sep<0))):
                        raise ValueError('nearest match outside its declared radius')
                    wrong=np.zeros(hi-lo,bool)
                    if survey=='decals':
                        north=is_north(targets['ra'][lo:hi],targets['dec'][lo:hi])
                        wrong=matched&~np.where(north,raw['release']==9011,np.isin(raw['release'],[9010,9012]))
                    p=catalogue_photometry(survey,raw,clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits'])
                    _save_npz(out/f'{survey}_{lo:07d}.npz',multiple_matches=count>1,wrong_hemisphere=wrong)
                    record=dict(targets=hi-lo,matched=int(matched.sum()),multiple_matches=int((count>1).sum()),
                        wrong_hemisphere=int(wrong.sum()),negative_flux_measurements=int((p.observed&(p.flux<0)).sum()))
                    write_json(stamp,record)
                    print(f'audited {survey} {hi:,}/{n:,}: {record}',flush=True)
                stats.update(json.loads(stamp.read_text()))
            if done==n:
                shared=shared_source_mask(np.concatenate([x[0] for x in positions]),
                                          np.concatenate([x[1] for x in positions]))
                _save_npz(out/f'{survey}_shared_sources.npz',shared_source=shared)
                stats['targets_sharing_catalogue_source']=int(shared.sum())
            summary[survey]=dict(stats,complete=done==n)
            if done==n: write_json(finished,summary[survey])
        write_json(out/'report.json',dict(surveys=summary,complete=complete,model_fitted=False,
            policy='Multiple matches, shared catalogue sources and wrong-hemisphere Legacy matches are masks for final photometry assembly; raw measurements remain intact.'))
        if complete or not args.watch: return
        time.sleep(args.poll_seconds)


if __name__=='__main__': main()
