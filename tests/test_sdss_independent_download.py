"""The independent query excludes completed/ID-linked rows without losing misses."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('options', [[], ['--whole-list']])
def test_whole_list_download_identity_and_no_duplicate_work(tmp_path,monkeypatch,options):
    import sqlutilpy
    scripts=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('independent_fetch',scripts/'fetch_sdss_without_ids.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'cache';root.mkdir()
    np.savez(root/'targets.npz',ra=np.arange(10.,15.),dec=np.zeros(5),master_index=np.arange(5))
    members=dict(catalogue=np.array(['sdss','sdss']),source_id=np.array(['a','b']),object_index=np.array([1,3]))
    np.savez(tmp_path/'members.npz',**members)
    master=tmp_path/'manifest.json';master.write_text(json.dumps(dict(files={'members.npz':module.sha256(tmp_path/'members.npz')})))
    (root/'provenance.json').write_text(json.dumps(dict(master_sha256=module.sha256(master),config=dict(master_manifest=str(master),match_radius_arcsec={'sdss':1.}))))
    (root/'progress_sdss.json').write_text(json.dumps(dict(processed=1)))
    monkeypatch.setattr(module,'cached_query',lambda *a,**kw:dict(sdss_name=np.array(['a','b']),objid=np.array(['1001','1003'])))
    class Connection:
        info=SimpleNamespace(backend_pid=123)
        def rollback(self):pass
        def close(self):pass
    monkeypatch.setattr(sqlutilpy,'getConnection',lambda **kw:Connection())
    uploads=[]
    monkeypatch.setattr(sqlutilpy,'upload',lambda *a,**kw:uploads.append(a))
    queries=[]
    def get(query,**kw):
        queries.append(query)
        if query.startswith('EXPLAIN'):
            return dict(plan=np.array(['Bitmap Index Scan on photoobjall_q3c_ang2ipix_idx']))
        ra=uploads[-1][1][1][::-1]
        return dict(idx=np.array([1,0]),ra=np.where(ra==12,12.,np.nan),
                    dec=np.where(ra==12,0.,np.nan),match_sep_arcsec=np.where(ra==12,.1,np.nan),
                    value_u=np.where(ra==12,-2.,np.nan))
    monkeypatch.setattr(sqlutilpy,'get',get)
    monkeypatch.setattr(sys,'argv',['fetch','--cache',str(root)]+options)
    module.main()
    assert len(uploads)==1
    np.testing.assert_array_equal(np.sort(uploads[0][1][1]),[12.,14.])
    with np.load(root/'sdss_without_ids'/'photometry.npz') as d:
        assert d['target_index'].tolist()==[2,4]
        assert d['value_u'][0]==-2. and np.isnan(d['ra'][1])
    request=json.loads((root/'sdss_without_ids'/'request.json').read_text())
    assert request['targets']==2 and request['row_cap'] is None
    module.main()
    assert len(uploads)==1 and len(queries)==2
