#!/usr/bin/env python
"""Reuse inventoried measurements and acquire only remaining five-survey results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from qso_pcolor.data import _load_npz
from qso_pcolor.gap_acquisition import prepare_reuse, acquire_survey
from qso_pcolor.sky_acquisition import acquisition_lock


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--inventory',required=True,type=Path)
    p.add_argument('--surveys',nargs='+',required=True,choices=('ps1','allwise','nsc','skymapper','vhs'))
    p.add_argument('--block-config',type=Path,default=Path('configs/photometry_sky_blocks.json'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--first-checkpoints',action='store_true',
                   help='Save one initial batch in every survey, then continue all remaining batches')
    p.add_argument('--max-batches',type=int,help='Operational pause after this many new aligned batches per survey')
    a=p.parse_args();cfg=json.loads((a.cache/'provenance.json').read_text())['config']
    operational=json.loads(a.block_config.read_text());targets=_load_npz(a.cache/'targets.npz')
    print('Starting cache-first acquisition for '+', '.join(a.surveys),flush=True)
    with acquisition_lock(a.cache/'gap_acquisition'/'worker.lock'):
        for s in a.surveys:prepare_reuse(s,a.cache,a.inventory,targets,cfg)
        if not a.prepare_only:
            if a.first_checkpoints:
                for s in a.surveys:
                    progress=a.cache/f'progress_{s}.json'
                    if progress.exists() and json.loads(progress.read_text())['processed']>=min(cfg['batch_size'],len(targets['ra'])):
                        continue
                    acquire_survey(s,a.cache,a.inventory,targets,cfg,
                                   query_size=operational['max_targets'],max_batches=1)
            for s in a.surveys:
                acquire_survey(s,a.cache,a.inventory,targets,cfg,query_size=operational['max_targets'],max_batches=a.max_batches)


if __name__=='__main__':main()
