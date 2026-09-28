"""Acquisition batches must preserve all eligible master objects."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def test_photometry_batches_keep_every_eligible_object_and_original_identity(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('full_sample_fetch', scripts / 'fetch_full_qso_photometry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = 7
    source = tmp_path / 'source.npz'
    np.savez(source, type=np.array(['PSF'] * 6 + ['EXP']),
             maskbits=np.array([0] * 5 + [1, 0]), release=np.full(n, 9010))
    objects = dict(ra=np.arange(n, dtype=float)+150, dec=np.full(n, 40.),
        zspec=np.full(n, 1.5), preferred_catalogue=np.full(n, 'desi'),
        preferred_input_row=np.arange(n), object_id=np.array([f'desi:{i}' for i in range(n)]),
        redshift_conflict=np.zeros(n, bool), extended_duplicate_group=np.zeros(n, bool))
    np.savez(tmp_path / 'objects.npz', **objects)
    manifest = dict(files={'objects.npz': module.sha256(tmp_path / 'objects.npz')},
        sources={'desi': dict(path=str(source), sha256=module.sha256(source))})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    cfg = dict(master_manifest=str(tmp_path / 'manifest.json'), cache_dir=str(tmp_path / 'cache'),
        batch_size=2, surveys=['sdss'], z_min=.1, z_max=4.4, min_abs_b_deg=0,
        legacy_releases=[9010, 9011, 9012], query_order_nside=4,
        match_radius_arcsec={'sdss': 1.}, clean=True, vhs_bad_bits=0, sdss_use_object_ids=False)
    config = tmp_path / 'config.json'; config.write_text(json.dumps(cfg))
    calls = []

    def fake_match(survey, ra, dec, cache, *, radius_arcsec):
        calls.append(ra.copy())
        rows = dict(ra=ra, dec=dec, clean=np.ones(len(ra), int))
        for band in 'ugriz':
            rows[f'value_{band}'] = np.full(len(ra), -1.)
            rows[f'error_{band}'] = np.ones(len(ra))
        return rows

    monkeypatch.setattr(module, 'match_catalogue', fake_match)
    monkeypatch.setattr(sys, 'argv', ['fetch', '--config', str(config), '--surveys', 'sdss'])
    module.main()
    assert sorted(np.concatenate(calls).tolist()) == list(np.arange(5)+150.)
    assert [len(call) for call in calls] == [2, 2, 1]
    root = next((tmp_path / 'cache').iterdir())
    with np.load(root / 'targets.npz') as targets:
        assert set(targets['object_id']) == {f'desi:{i}' for i in range(5)}
        assert np.array_equal(targets['ra'], objects['ra'][targets['master_index']])
    progress = json.loads((root / 'progress_sdss.json').read_text())
    assert progress['complete'] and progress['processed'] == progress['total'] == 5
    assert progress['observed_counts'] == [5] * 5  # negative fluxes remain observed
    provenance = (root/'provenance.json').read_bytes()
    monkeypatch.setattr(sys, 'argv', ['fetch', '--config', str(config), '--surveys', 'sdss', '--resume-cache', str(root)])
    module.main()
    assert len(calls) == 3  # completed acquisition is not fetched again
    assert (root/'provenance.json').read_bytes() == provenance
    monkeypatch.setattr(module, 'match_catalogue', lambda *a, **kw: pytest.fail('unexpected network query'))
    assembled = []
    monkeypatch.setattr(module, 'assemble_legacy_from_cache', lambda root, **kw: assembled.append(root))
    monkeypatch.setattr(sys, 'argv', ['fetch', '--config', str(config), '--surveys', 'decals', '--resume-cache', str(root)])
    module.main()
    assert assembled == [root]
    cfg['z_max'] = 5.
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='differs'):
        module.main()
    monkeypatch.setattr(sys, 'argv', ['fetch', '--config', str(config)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, 'argv', ['fetch', '--config', str(config), '--surveys', 'ps1'])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2  # never bypass the verified local inventory
