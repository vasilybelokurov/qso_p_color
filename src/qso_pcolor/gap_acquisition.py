"""Cache-first acquisition for the five remaining QSO surveys.

Every query result is durable before it enters a larger target-aligned batch.
The catalogue system, quality policy and scientific target list stay fixed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from .data import _load_npz, _save_npz
from .legacy_acquisition import file_hash
from .multisurvey_data import SURVEYS, catalogue_photometry
from .sky_acquisition import write_record


def query_sql(survey: str, radius: float, *, by_id: bool = False) -> str:
    """Native measurements; local targets first, indexed survey table second."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('positive finite radius required')
    if by_id and survey != 'ps1':
        raise ValueError('only PS1 has verified exact catalogue ID links')
    spec=SURVEYS[survey]
    join='c.objid=m.objid' if by_id else f'q3c_join(m.ra,m.dec,c.{spec.ra},c.{spec.dec},{radius}/3600.)'
    if survey=='ps1' and not by_id:
        # A flat join is reordered into a full stack-table scan on WSDB.
        # OFFSET 0 preserves a parameterised Q3C index lookup; keep all candidates.
        return f'''SELECT m.idx,x.* FROM mytmptable m CROSS JOIN LATERAL (
          SELECT {spec.columns}, q3c_dist(m.ra,m.dec,c.ra,c.dec)*3600 AS match_sep_arcsec
          FROM {spec.table} c WHERE {join} AND ({spec.where}) OFFSET 0) x'''
    join_type='LEFT JOIN' if by_id else 'JOIN'
    return f'''SELECT m.idx, {spec.columns},
      q3c_dist(m.ra,m.dec,c.{spec.ra},c.{spec.dec})*3600 AS match_sep_arcsec
      FROM mytmptable m {join_type} {spec.table} c
      ON {join} AND ({spec.where})'''


def nearest_rows(raw: dict, n: int, *, spatial_counts: bool) -> dict:
    """Reduce zero/multiple counterparts to one row per target, retaining counts."""
    ids=np.asarray(raw['idx'])
    if not len(ids):ids=ids.astype(np.int64)
    if not np.issubdtype(ids.dtype,np.integer) or np.any((ids<0)|(ids>=n)):
        raise ValueError('invalid returned target identities')
    finite=np.isfinite(raw['ra']+raw['dec'])
    sep=np.where(finite & np.isfinite(raw['match_sep_arcsec']),raw['match_sep_arcsec'],np.inf)
    order=np.lexsort((raw['dec'],raw['ra'],sep,ids))
    unique,first=np.unique(ids[order],return_index=True)
    rows=order[first]
    chosen={k:np.asarray(v)[rows].astype(str) if np.asarray(v).dtype.hasobject else np.asarray(v)[rows]
            for k,v in raw.items()}
    result=empty_like(chosen,n)
    assign_rows(result,unique,chosen)
    result['idx']=np.arange(n)
    result['match_count']=np.bincount(ids[finite],minlength=n) if spatial_counts else np.full(n,-1,np.int64)
    result['lookup_kind']=np.full(n,1 if spatial_counts else 2,np.int64)
    return result


def fetch_native(survey: str, inputs: dict, cache: Path, *, radius: float,
                 by_id: bool = False) -> dict:
    """Plan-checked ID or simple Q3C join, with atomic, query-keyed caching."""
    sql=query_sql(survey,radius,by_id=by_id)
    h=hashlib.sha256(sql.encode())
    for k,v in sorted(inputs.items()):h.update(k.encode()+v.dtype.str.encode()+v.tobytes())
    cache.mkdir(parents=True,exist_ok=True);path=cache/(h.hexdigest()[:24]+'.npz')
    if path.exists():return _load_npz(path)
    import sqlutilpy as sqlutil
    started=time.monotonic();conn=sqlutil.getConnection(db='wsdb',driver='psycopg')
    meta=path.with_suffix('.json')
    record=dict(query=sql,targets=len(inputs['ra']),by_id=by_id,stage='querying',
                database_pid=int(conn.info.backend_pid),started_utc=datetime.now(timezone.utc).isoformat())
    try:
        names=['idx',*inputs];arrays=[np.arange(len(inputs['ra'])),*inputs.values()]
        sqlutil.upload('mytmptable',arrays,names,conn=conn,noCommit=True,temp=True,analyze=True)
        plan=sqlutil.get('EXPLAIN '+sql,conn=conn,notNamed=True,asDict=True,strLength=30000,
            preamb="SET jit=off; SET cursor_tuple_fraction=1; SET statement_timeout='7200s'")
        lines=next(iter(plan.values())).tolist();record['plan']=lines
        write_record(meta,record)
        table=SURVEYS[survey].table.split('.')[1]
        index_token='objid' if by_id else 'q3c'
        if any(f'Seq Scan on {table}' in line for line in lines) or not any(
                'Index' in line and index_token in line for line in lines):
            raise RuntimeError(f'{survey}: required {index_token} index not used')
        raw=sqlutil.get(sql,conn=conn,asDict=True,intNullVal=-1)
        if not len(raw['idx']):
            # sqlutilpy returns float-typed empty columns regardless of SQL type.
            raw['idx']=np.empty(0,np.int64)
            for key in SURVEYS[survey].extra:
                raw[key]=np.empty(0,dtype='<U4' if key=='cc_flags' else np.int64)
        result=nearest_rows(raw,len(inputs['ra']),spatial_counts=not by_id)
        matched=np.isfinite(result['ra']+result['dec'])
        if not by_id and np.any(matched & (~np.isfinite(result['match_sep_arcsec']) |
                                          (result['match_sep_arcsec']>radius*(1+1e-8)))):
            raise ValueError('positional result outside association radius')
        _save_npz(path,**result)
        record.update(stage='complete',elapsed_seconds=time.monotonic()-started,
                      matched=int(matched.sum()),sha256=file_hash(path))
        write_record(meta,record)
        print(f'{survey} {"ID" if by_id else "position"}: {len(inputs["ra"]):,} targets saved in '
              f'{record["elapsed_seconds"]:.1f}s; {record["matched"]:,} counterparts',flush=True)
        return result
    except Exception as error:
        write_record(meta,dict(record,stage='failed',error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        conn.rollback();conn.close()


def empty_like(template: dict, n: int) -> dict:
    """Explicit sentinels, never uninitialised measurements."""
    result={}
    for k,v in template.items():
        dtype=np.asarray(v).dtype
        fill=np.nan if np.issubdtype(dtype,np.inexact) else ('' if dtype.kind in 'US' else -1)
        result[k]=np.full(n,fill,dtype=dtype)
    return result


def assign_rows(output: dict, take: np.ndarray, values: dict) -> None:
    for k in output:
        if k not in values:continue
        dtype=np.promote_types(output[k].dtype,values[k].dtype)
        if dtype!=output[k].dtype:output[k]=output[k].astype(dtype)
        output[k][take]=values[k]


def prepare_reuse(survey: str, root: Path, inventory: Path, targets: dict, cfg: dict) -> dict:
    """Assemble all verified local results, including nonmatches, with hashes."""
    out=root/'gap_acquisition'/survey;out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((inventory/'summary.json').read_text())
    if file_hash(root/'targets.npz')!=summary['target_sha256']:
        raise ValueError('target inventory changed')
    p=inventory/f'{survey}_target_inventory.npz'
    if file_hash(p)!=summary['output_sha256'][p.name] or file_hash(inventory/'native_files.json')!=summary['native_file_manifest_sha256']:
        raise ValueError('inventory references changed')
    request=dict(targets_sha256=summary['target_sha256'],inventory_sha256=file_hash(p),
                 sources_sha256=summary['native_file_manifest_sha256'],radius=cfg['match_radius_arcsec'][survey],
                 clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits'])
    stamp=out/'reuse.json';saved=out/'reuse.npz'
    if saved.exists():
        record=json.loads(stamp.read_text())
        if record['request']!=request or file_hash(saved)!=record['sha256']:
            raise ValueError('saved reuse provenance differs')
        return _load_npz(saved)
    inv=_load_npz(p)
    if not np.array_equal(inv['object_id'],targets['object_id']):raise ValueError('inventory object identities differ')
    selected=np.flatnonzero(inv['reusable_source_file']>=0)
    records=json.loads((inventory/'native_files.json').read_text());result={}
    spec=SURVEYS[survey]
    keys=['ra','dec',*spec.extra,*[f'{x}_{b}' for b in spec.bands for x in ('value','error')]]
    for fi in np.unique(inv['reusable_source_file'][selected]):
        source=records[fi];path=Path(source['path'])
        if file_hash(path)!=source['sha256']:raise ValueError('source cache changed')
        raw=_load_npz(path);local=np.flatnonzero(inv['reusable_source_file'][selected]==fi)
        rr=inv['reusable_source_row'][selected[local]];have=rr>=0
        subset={k:raw[k][rr[have]] for k in keys}
        if not result:result=empty_like({k:raw[k] for k in keys},len(selected))
        assign_rows(result,local[have],subset)
    matched=np.isfinite(result['ra']+result['dec']);sep=np.full(len(selected),np.nan)
    t=selected[matched]
    sep[matched]=SkyCoord(targets['ra'][t]*u.deg,targets['dec'][t]*u.deg).separation(
        SkyCoord(result['ra'][matched]*u.deg,result['dec'][matched]*u.deg)).arcsec
    if np.any(sep[matched]>request['radius']*(1+1e-8)):raise ValueError('reused association outside radius')
    result.update(target_index=selected,idx=np.arange(len(selected)),match_sep_arcsec=sep,
                  match_count=np.where(matched,-1,0).astype(np.int64),lookup_kind=np.zeros(len(selected),np.int64))
    phot=catalogue_photometry(survey,result,clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits'])
    if not np.array_equal(phot.observed,inv['observed'][selected]):raise ValueError('reused band masks differ')
    _save_npz(saved,**result)
    verified=_load_npz(saved)
    if not all(np.array_equal(v,verified[k],equal_nan=np.issubdtype(v.dtype,np.inexact)) for k,v in result.items()):
        raise ValueError('saved reuse differs')
    write_record(stamp,dict(request=request,sha256=file_hash(saved),targets=len(selected),
                            known_nonmatches=int((~matched).sum())))
    print(f'{survey}: {len(selected):,} existing results assembled locally',flush=True)
    return result


def acquire_survey(survey: str, root: Path, inventory: Path, targets: dict, cfg: dict,
                   *, query_size: int, max_batches: int | None = None) -> None:
    """Fill only gaps; publish full batches to the existing reader and auditor."""
    if query_size<1:raise ValueError('positive operational query size required')
    reusable=prepare_reuse(survey,root,inventory,targets,cfg)
    out=root/'gap_acquisition'/survey;n=len(targets['ra']);batch=cfg['batch_size']
    early=out/'verified_early_id_results.npz'
    if early.exists():
        stamp=json.loads(early.with_suffix('.json').read_text())
        if file_hash(early)!=stamp['sha256'] or file_hash(root/'targets.npz')!=stamp['targets_sha256']:
            raise ValueError('early verified ID results changed')
        extra=_load_npz(early)
        keep=~np.isin(extra['target_index'],reusable['target_index'])
        reusable={k:np.concatenate((v,extra[k][keep])) for k,v in reusable.items()}

    ref=np.full(n,-1,np.int64);ref[reusable['target_index']]=np.arange(len(reusable['target_index']))
    ids=np.full(n,-1,np.int64)
    if survey=='ps1':
        p=inventory/'ps1_identifier_candidates.npz'
        summary=json.loads((inventory/'summary.json').read_text())
        if file_hash(p)!=summary['identifier_links']['candidate_file_sha256'][p.name]:raise ValueError('PS1 ID candidates changed')
        links=_load_npz(p);use=links['within_radius'];ti=links['target_index'][use];values=links['objid'][use]
        # Membership aliases can repeat, but different IDs must not be hidden.
        pairs=np.unique(np.column_stack((ti,values)),axis=0)
        unique,counts=np.unique(pairs[:,0],return_counts=True)
        if np.any(counts>1):raise ValueError('conflicting PS1 IDs')
        ids[pairs[:,0]]=pairs[:,1]
    template={k:v for k,v in reusable.items() if k!='target_index'}
    bands=np.zeros(len(catalogue_photometry(survey,template,clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits']).bands),np.int64)
    completed=0;new_batches=0;started=time.monotonic()
    progress=root/f'progress_{survey}.json'
    previous_done=json.loads(progress.read_text())['processed'] if progress.exists() else 0
    for lo in range(0,n,batch):
        hi=min(n,lo+batch);path=root/'acquired'/f'{survey}_{lo:07d}.npz'
        if path.exists():
            raw=_load_npz(path)
            if not np.array_equal(raw['object_id'],targets['object_id'][lo:hi]):raise ValueError('saved acquired target identities differ')
        else:
            raw=empty_like(template,hi-lo);local=np.flatnonzero(ref[lo:hi]>=0)
            assign_rows(raw,local,{k:v[ref[lo:hi][local]] for k,v in reusable.items() if k in raw})
            gaps=np.flatnonzero(ref[lo:hi]<0)+lo
            for offset in range(0,len(gaps),query_size):
                part=gaps[offset:offset+query_size];done=np.zeros(len(part),bool)
                write_record(out/'status.json',dict(stage='querying',batch_start=lo,query_targets=len(part),
                    completed_aligned_targets=completed,reused_targets=len(reusable['idx']),total=n,
                    updated_utc=datetime.now(timezone.utc).isoformat()))
                idsel=np.flatnonzero(ids[part]>0)
                if len(idsel):
                    t=part[idsel]
                    fetched=fetch_native(survey,dict(ra=targets['ra'][t],dec=targets['dec'][t],objid=ids[t]),
                                         out/'id_queries',radius=cfg['match_radius_arcsec'][survey],by_id=True)
                    ok=np.isfinite(fetched['ra']+fetched['dec']+fetched['match_sep_arcsec']) & (
                        fetched['match_sep_arcsec']<=cfg['match_radius_arcsec'][survey]) & (fetched['match_sep_arcsec']>=0)
                    assign_rows(raw,t[ok]-lo,{k:v[ok] for k,v in fetched.items()});done[idsel[ok]]=True
                t=part[~done]
                if len(t):
                    fetched=fetch_native(survey,dict(ra=targets['ra'][t],dec=targets['dec'][t]),out/'position_queries',
                                         radius=cfg['match_radius_arcsec'][survey])
                    assign_rows(raw,t-lo,fetched)
            raw.update(idx=np.arange(hi-lo),target_index=np.arange(lo,hi),object_id=targets['object_id'][lo:hi])
            if np.any(raw['lookup_kind']<0):raise ValueError('unfilled acquired targets')
            _save_npz(path,**raw);new_batches+=1
        phot=catalogue_photometry(survey,raw,clean=cfg['clean'],vhs_bad_bits=cfg['vhs_bad_bits'])
        bands+=phot.observed.sum(0);completed=hi
        record=dict(survey=survey,processed=hi,total=n,complete=hi==n,bands=list(phot.bands),
                    observed_counts=bands.tolist(),elapsed_seconds=time.monotonic()-started,
                    status='Cache-first raw photometry; independent association audit pending. No model fit.')
        if hi>=previous_done:write_record(progress,record)
        write_record(out/'status.json',dict(record,stage='complete' if hi==n else 'saved',
                     updated_utc=datetime.now(timezone.utc).isoformat()))
        print(f'{survey}: {hi:,}/{n:,} target-aligned rows saved',flush=True)
        if max_batches is not None and new_batches>=max_batches:return
