#!/usr/bin/env python
"""Inventory existing PS1/AllWISE identifier links, without downloading fluxes.

These are candidates for checked catalogue-ID lookups, not certified photometry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u
from qso_pcolor.data import _load_npz, _save_npz
from qso_pcolor.legacy_acquisition import file_hash


def exact_integer_id(value) -> int:
    """Reject rounded/scientific-notation identifiers instead of guessing digits."""
    s='' if value is None else str(value).strip()
    return int(s) if s.isascii() and s.isdigit() and 0<int(s)<=np.iinfo(np.int64).max else -1


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out
    targets=_load_npz(a.cache/'targets.npz')
    cfg=json.loads((a.cache/'provenance.json').read_text())['config']
    master=Path(cfg['master_manifest']).expanduser()
    members=_load_npz(master.parent/'members.npz')
    reverse=np.full(members['object_index'].max()+1,-1,np.int64)
    reverse[targets['master_index']]=np.arange(len(targets['ra']))
    selected=(members['catalogue']=='sdss') & (reverse[members['object_index']]>=0)
    names=members['source_id'][selected].astype(str)
    indexes=reverse[members['object_index'][selected]]
    order=np.argsort(names);names,indexes=names[order],indexes[order]
    if len(np.unique(names))!=len(names):raise ValueError('SDSS membership names are not unique')
    def align(values):
        pos=np.searchsorted(names,values)
        good=pos<len(names)
        good[good]&=names[pos[good]]==values[good]
        target=np.full(len(values),-1,np.int64);target[good]=indexes[pos[good]]
        return target
    ps1=Path.home()/'data/qso/ps1/qso_ps1_object_index.parquet'
    rows=pq.read_table(ps1).to_pydict()
    ti=align(np.array(rows['sdss_name']))
    ids=np.array(rows['objID'])
    if not np.issubdtype(ids.dtype,np.integer):raise ValueError('PS1 IDs are not exact integers')
    good=(ti>=0)&(ids>0)
    row=np.flatnonzero(good);sel=ti[good]
    sep=SkyCoord(targets['ra'][sel]*u.deg,targets['dec'][sel]*u.deg).separation(
        SkyCoord(np.array(rows['ps1_ra'])[good]*u.deg,np.array(rows['ps1_dec'])[good]*u.deg)).arcsec
    within=np.isfinite(sep)&(sep<=cfg['match_radius_arcsec']['ps1'])
    _save_npz(out/'ps1_identifier_candidates.npz',target_index=sel,source_row=row,
              objid=ids[good],separation_arcsec=sep,within_radius=within)
    report=dict(ps1=dict(path=str(ps1),sha256=file_hash(ps1),rows=len(ids),
        eligible_target_identities=len(np.unique(sel)),within_radius_targets=len(np.unique(sel[within])),
        status='Exact integer object IDs; stack primary-detection status and measurement columns still require verification.'))
    wise=Path.home()/'data/qso/wise/snapshots/qso_wise_full_events.parquet'
    parts=[];f=pq.ParquetFile(wise)
    for batch in f.iter_batches(batch_size=1000000,columns=['sdss_name','allwise_cntr']):
        parts.append(pa.Table.from_batches([batch]).group_by(['sdss_name','allwise_cntr']).aggregate([]))
    unique=pa.concat_tables(parts).group_by(['sdss_name','allwise_cntr']).aggregate([]).to_pydict()
    ti=align(np.array(unique['sdss_name']));ids=np.full(len(ti),-1,np.int64)
    bad=0
    for j,v in enumerate(unique['allwise_cntr']):
        ids[j]=exact_integer_id(v)
        if ids[j]<0:bad+=1
    good=(ti>=0)&(ids>0)
    pairs=np.unique(np.column_stack((ti[good],ids[good])),axis=0)
    ut,counts=np.unique(pairs[:,0],return_counts=True)
    single=ut[counts==1];take=np.isin(pairs[:,0],single)
    _save_npz(out/'allwise_identifier_candidates.npz',target_index=pairs[:,0],cntr=pairs[:,1],
              single_id_for_target=take)
    report['allwise']=dict(path=str(wise),sha256=file_hash(wise),bytes=wise.stat().st_size,mtime_ns=wise.stat().st_mtime_ns,
        total_events=f.metadata.num_rows,distinct_name_id_pairs=len(ti),
        invalid_or_missing_id_pairs=bad,eligible_targets_with_id=len(ut),
        eligible_targets_with_single_id=len(single),conflicting_target_ids=int((counts>1).sum()),
        status='No exact AllWISE IDs recovered: stored cntr strings are rounded scientific notation or missing. Do not cast these to integers for catalogue joins. W1/W2 time-series photometry is not the required AllWISE W1-W4 catalogue photometry.')
    report['candidate_file_sha256']={p.name:file_hash(p) for p in out.glob('*_identifier_candidates.npz')}
    report.update(target_sha256=file_hash(a.cache/'targets.npz'),members_sha256=file_hash(master.parent/'members.npz'),
                  implementation_sha256=file_hash(Path(__file__)),new_database_queries=0)
    (out/'identifier_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
