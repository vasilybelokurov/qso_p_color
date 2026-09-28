#!/usr/bin/env python
"""Clean, partition, match and measure stellar regions without fitting a model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from scipy.spatial import cKDTree

from build_qso_master import sha256
from fetch_full_qso_photometry import write_json
from qso_pcolor.data import _save_npz
from qso_pcolor.legacy import brick_image_fetcher, load_bricks
from qso_pcolor.multisurvey_data import band_labels, catalogue_photometry, match_catalogue
from qso_pcolor.preparation import (unit_vectors, clean_stellar_rows, spatial_role_manifest,
                                    assign_object_roles, cone_selection_area)


def clean(cfg, root, regions, raw_root):
    with np.load(Path(cfg['qso_members']).expanduser()) as q:
        known = cKDTree(unit_vectors(q['ra'], q['dec']))
    progress = json.loads((raw_root/'stellar_source_progress.json').read_text())
    records = {r['cone']:r for r in progress['regions']}
    counts = []
    for c in regions:
        cid = c['cone']; out = root/'clean'/f'cone_{cid:03d}.npz'
        report = out.with_suffix('.json')
        if not out.exists() or not report.exists():
            sql_hash = hashlib.sha1(records[cid]['query'].encode()).hexdigest()[:10]
            source = raw_root/'stellar_sources'/f'cone_{cid:03d}_{sql_hash}.npz'
            with np.load(source) as d:
                rows = {k:d[k] for k in d.files if not k.startswith('_')}
            keep, count = clean_stellar_rows(rows, known,
                radius_arcsec=cfg['known_qso_radius_arcsec'], min_abs_b_deg=cfg['min_abs_b_deg'])
            _save_npz(out, raw_input_row=np.flatnonzero(keep), **{k:v[keep] for k,v in rows.items()})
            write_json(report, dict(cone=cid, raw_sha256=sha256(source), **count))
        counts.append(json.loads(report.read_text()))
        print(f"clean {len(counts)}/{len(regions)}: cone {cid}, {counts[-1]['retained']:,} retained", flush=True)
    # Cross-region duplicates would invalidate independent area/count accounting.
    keys = []
    for c in regions:
        with np.load(root/'clean'/f"cone_{c['cone']:03d}.npz") as d:
            keys.append(np.rec.fromarrays([d[k] for k in ('release','brickid','objid')]))
    joined = np.concatenate(keys)
    if len(np.unique(joined)) != len(joined):
        raise ValueError('catalogue duplicates across regions: resolve before area accounting')
    report = dict(regions=counts, totals={key:sum(c[key] for c in counts) for key in
        ('raw','psf_mask_clean','own_hemisphere_and_latitude','duplicate_catalogue_keys','known_qso_removed','retained')},
        cross_region_duplicate_keys=0)
    write_json(root/'cleaning_report.json', report)
    print(json.dumps(report['totals']), flush=True)


def roles(cfg, root, regions):
    old = json.loads(Path(cfg['historical_holdout_model']).read_text())['meta']
    if old['holdout_nside'] != cfg['nside']:
        raise ValueError('historical and new coarse cell resolutions differ')
    manifest = spatial_role_manifest(regions, old['holdout_blocks'], seed=cfg['seed'],
        northern_test_fraction=cfg['northern_test_fraction'],
        selection_fraction=cfg['selection_fraction'], calibration_fraction=cfg['calibration_fraction'])
    historical = [dict(c, radius_deg=cfg['historical_field_radius_deg']) for c in
                  json.loads(Path(cfg['historical_fields']).read_text()) if c['held']]
    historical += json.loads(Path(cfg['extra_validation_config']).read_text())['extra_background_fields']
    # A cone intersecting a historical test cone is entirely assigned to test.
    for c in manifest['regions']:
        for h in historical:
            chord = np.linalg.norm(unit_vectors([c['ra']],[c['dec']])[0]-unit_vectors([h['ra']],[h['dec']])[0])
            if chord <= 2*np.sin(np.deg2rad(cfg['cone_radius_deg']+h['radius_deg'])/2):
                c['role'] = 'test'
    manifest.update(coordinate_frame='Galactic', ordering='NESTED', nside=cfg['nside'],
        fine_nside=cfg['fine_nside'], historical_test_cones=historical,
        geometry_frozen_before_fits=True, coverage_certified=False,
        region_counts=dict(Counter(c['hemisphere']+':'+c['role'] for c in manifest['regions'])))
    for hemi in ('north','south'):
        for role in ('fit','select','calib','test'):
            if not manifest['region_counts'].get(hemi+':'+role):
                raise ValueError(f'no {hemi} {role} regions')
    with np.load(cfg['qso_targets']) as q:
        qroles = assign_object_roles(q['ra'],q['dec'],manifest,nside=cfg['nside'],
            fine_nside=cfg['fine_nside'],selection_fraction=cfg['selection_fraction'],
            calibration_fraction=cfg['calibration_fraction'],radius_deg=cfg['cone_radius_deg'],
            historical_test_cones=historical)
        _save_npz(root/'qso_roles.npz',object_id=q['object_id'],master_index=q['master_index'],role=qroles)
    manifest['qso_role_counts'] = dict(Counter(qroles.tolist()))
    write_json(root/'spatial_roles.json',manifest)
    print('Spatial roles:',json.dumps(manifest['region_counts']),flush=True)


def photometry(cfg, root, regions):
    def one(c):
        cid=c['cone'];out=root/'photometry'/f'cone_{cid:03d}.npz';report=out.with_suffix('.json')
        if out.exists() and report.exists(): return json.loads(report.read_text())
        rows=dict(np.load(root/'clean'/f'cone_{cid:03d}.npz'));n=len(rows['ra'])
        flux=[];variance=[];ambiguities=[];separations=[];survey_reports={}
        for survey in cfg['surveys']:
            if survey == 'decals':
                raw={k:rows[k] for k in ('ra','dec','release','maskbits')}
                for band in ('g','r','z','w1','w2'):
                    raw['value_'+band]=rows['flux_'+band]
                    raw['error_'+band]=rows['flux_ivar_'+band]
                    raw['nobs_'+band]=rows['nobs_'+band]
                ambiguous=np.zeros(n,bool);sep=np.zeros(n)
            else:
                parts=[]
                for lo in range(0,n,cfg['batch_size']):
                    hi=min(n,lo+cfg['batch_size'])
                    parts.append(match_catalogue(survey,rows['ra'][lo:hi],rows['dec'][lo:hi],
                        root/'queries',radius_arcsec=cfg['match_radius_arcsec'][survey],include_match_count=True))
                if not parts: raise ValueError(f'empty region {cid}')
                raw={k:np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'idx'}
                ambiguous=raw['match_count']>1
                matched=np.isfinite(raw['ra']+raw['dec'])
                key=np.rec.fromarrays([raw['ra'][matched],raw['dec'][matched]])
                _,inverse,count=np.unique(key,return_inverse=True,return_counts=True)
                shared=np.zeros(n,bool);shared[matched]=count[inverse]>1
                ambiguous |= shared
                sep=raw['match_sep_arcsec']
            p=catalogue_photometry(survey,raw,clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits'])
            flux.append(p.flux);variance.append(p.variance);ambiguities.append(ambiguous);separations.append(sep)
            survey_reports[survey]=dict(ambiguous_sources=int(ambiguous.sum()),
                matched_sources=int(np.isfinite(raw['ra']).sum()),
                bands=list(p.bands),raw_observed=p.observed.sum(axis=0).tolist(),
                unambiguous_observed=(p.observed & ~ambiguous[:,None]).sum(axis=0).tolist())
            print(f"photometry cone {cid}: {survey} ({n:,} sources)",flush=True)
        f=np.concatenate(flux,axis=1);v=np.concatenate(variance,axis=1)
        clean_f=f.copy();clean_v=v.copy();offset=0
        for survey,ambiguous in zip(cfg['surveys'],ambiguities):
            width=len(band_labels((survey,)))
            clean_f[ambiguous,offset:offset+width]=np.nan;clean_v[ambiguous,offset:offset+width]=np.inf
            offset+=width
        _save_npz(out,ra=rows['ra'],dec=rows['dec'],release=rows['release'],brickid=rows['brickid'],objid=rows['objid'],
            raw_input_row=rows['raw_input_row'],flux=clean_f,variance=clean_v,raw_flux=f,raw_variance=v,
            bands=np.array(band_labels(tuple(cfg['surveys']))),surveys=np.array(cfg['surveys']),
            association_ambiguous=np.column_stack(ambiguities),match_sep_arcsec=np.column_stack(separations))
        record=dict(cone=cid,rows=n,surveys=survey_reports,
            usable_any_band=int((np.isfinite(clean_f)&np.isfinite(clean_v)&(clean_v>0)).any(axis=1).sum()),
            association_policy='Mask survey bands for multiple matches or a catalogue source shared by distinct targets; preserve raw measurements.')
        write_json(report,record);return record
    done=[];failed=[]
    with ThreadPoolExecutor(cfg['photometry_workers']) as pool:
        jobs={pool.submit(one,c):c['cone'] for c in regions}
        for f in as_completed(jobs):
            try: done.append(f.result())
            except Exception as error:
                failed.append(dict(cone=jobs[f],error=f'{type(error).__name__}: {error}'))
                print('PHOTOMETRY ERROR:',failed[-1],flush=True)
            write_json(root/'photometry_progress.json',dict(completed=len(done),total=len(regions),regions=done,failed=failed))


def areas(cfg, root, regions):
    old=Path(cfg['existing_image_cache']);old_fetch=brick_image_fetcher(old)
    new_fetch=brick_image_fetcher(root/'brick_images')
    # Regions can touch the same brick; serialise each fetch, not whole regions.
    locks={};guard=threading.Lock()
    def fetch(brick,hemi,kind):
        with guard: lock=locks.setdefault((brick,hemi,kind),threading.Lock())
        with lock:
            path=old/hemi/f'{brick}-{kind}.fits.fz'
            return old_fetch(brick,hemi,kind) if path.exists() else new_fetch(brick,hemi,kind)
    def one(c):
        path=root/'areas'/f"cone_{c['cone']:03d}.json"
        if path.exists(): return json.loads(path.read_text())
        value=cone_selection_area(c['ra'],c['dec'],cfg['cone_radius_deg'],c['hemisphere'],
            load_bricks(cfg['bricks']),fetch,bands=tuple(cfg['exposure_bands']),
            n_points=cfg['area_points'],seed=cfg['seed']+c['cone'],min_abs_b_deg=cfg['min_abs_b_deg'])
        record=dict(cone=c['cone'],**value);write_json(path,record)
        print(f"area cone {c['cone']}: {value['area_deg2']:.5f} deg2",flush=True);return record
    done=[];failed=[]
    with ThreadPoolExecutor(cfg['area_workers']) as pool:
        jobs={pool.submit(one,c):c['cone'] for c in regions}
        for f in as_completed(jobs):
            try: done.append(f.result())
            except Exception as error:
                failed.append(dict(cone=jobs[f],error=f'{type(error).__name__}: {error}'))
                print('AREA ERROR:',failed[-1],flush=True)
            write_json(root/'area_progress.json',dict(completed=len(done),total=len(regions),regions=done,failed=failed))


def coverage_report(cfg,root,regions):
    """Write coverage and association diagnostics even when some queries failed."""
    roles=json.loads((root/'spatial_roles.json').read_text())
    by_id={c['cone']:c for c in roles['regions']}
    labels=band_labels(tuple(cfg['surveys']))
    counts={h+':'+r:np.zeros(len(labels),np.int64) for h in ('north','south') for r in ('fit','select','calib','test')}
    missing_phot=[];missing_area=[];association_counts=Counter();area_total=0.
    for c in regions:
        cid=c['cone'];p=root/'photometry'/f'cone_{cid:03d}.json';a=root/'areas'/f'cone_{cid:03d}.json'
        if p.exists():
            record=json.loads(p.read_text());key=c['hemisphere']+':'+by_id[cid]['role'];offset=0
            for survey in cfg['surveys']:
                s=record['surveys'][survey];width=len(s['bands'])
                counts[key][offset:offset+width]+=s['unambiguous_observed'];offset+=width
                association_counts[survey]+=s['ambiguous_sources']
        else: missing_phot.append(cid)
        if a.exists(): area_total+=json.loads(a.read_text())['area_deg2']
        else: missing_area.append(cid)
    pooled={role:counts['north:'+role]+counts['south:'+role] for role in ('fit','select','calib','test')}
    shortfalls={role:[labels[j] for j in np.flatnonzero(values<cfg['minimum_band_per_role'])] for role,values in pooled.items()}
    write_json(root/'coverage_report.json',dict(bands=list(labels),
        hemisphere_role_band_counts={k:v.tolist() for k,v in counts.items()},
        pooled_role_band_counts={k:v.tolist() for k,v in pooled.items()},band_shortfalls=shortfalls,
        missing_photometry_regions=missing_phot,missing_area_regions=missing_area,
        ambiguous_associations_by_survey=dict(association_counts),base_area_deg2=area_total,
        base_preparation_complete=not missing_phot and not missing_area and not any(shortfalls.values()),
        external_survey_area_status='Legacy optical selection area measured; external survey footprints and band-specific completeness remain separate.',
        model_fitted=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/stellar_preparation.json'))
    parser.add_argument('--part',choices=('clean','roles','photometry','areas','report','all'),default='all')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());raw=Path(cfg['raw_root'])
    regions=json.loads((raw/'stellar_seed_regions.json').read_text())
    identity=dict(config=cfg,regions_sha256=sha256(raw/'stellar_seed_regions.json'),
        qso_members_sha256=sha256(Path(cfg['qso_members']).expanduser()),
        script_sha256=sha256(Path(__file__)),
        selection_code_sha256=sha256(Path(__file__).parents[1]/'src/qso_pcolor/preparation.py'),
        adapter_sha256=sha256(Path(__file__).parents[1]/'src/qso_pcolor/multisurvey_data.py'),
        legacy_code_sha256=sha256(Path(__file__).parents[1]/'src/qso_pcolor/legacy.py'),
        raw_progress_sha256=sha256(raw/'stellar_source_progress.json'),
        input_sha256={k:sha256(Path(cfg[k])) for k in
            ('historical_holdout_model','historical_fields','extra_validation_config','qso_targets','bricks')})
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    root=Path(cfg['output_root'])/version;root.mkdir(parents=True,exist_ok=True)
    for folder in ('clean','photometry','areas'): (root/folder).mkdir(exist_ok=True)
    write_json(root/'provenance.json',identity)
    print('Preparation cache:',root,flush=True)
    if args.part in ('all','clean'): clean(cfg,root,regions,raw)
    if args.part in ('all','roles'): roles(cfg,root,regions)
    if args.part == 'all':
        with ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(photometry,cfg,root,regions),pool.submit(areas,cfg,root,regions)]
            for f in as_completed(jobs): f.result()
        coverage_report(cfg,root,regions)
    elif args.part == 'photometry': photometry(cfg,root,regions)
    elif args.part == 'areas': areas(cfg,root,regions)
    elif args.part == 'report': coverage_report(cfg,root,regions)


if __name__ == '__main__': main()
