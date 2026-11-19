"""Promoted Euclid pointers (19 Nov 2026): each loads its bundle (manifest hashes checked by load) and scores Euclid-only
photometry of a point source. Stage-1 pointers carry the Euclid point-source selection (selection 'euclid_point'):
the quasar colour density is the promoted one exactly, while the background density and the intensities are thinned
to that selection, so scores of Legacy-only input differ from ``current`` by design."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'models/multisurvey_psf'
POINTERS = ('current_euclid', 'current_euclid_magindep', 'current_euclid_joint', 'current_euclid_joint_magindep')
pytestmark = pytest.mark.skipif(not all((D/p).exists() for p in POINTERS), reason='Euclid pointers not present')


def _score(model, flux, var, bands):
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy
    n = len(flux)
    return model.score(Photometry(flux, var, bands), ra_deg=np.full(n, 52.), dec_deg=np.full(n, -45.),
                       z_primary=np.full(n, 1.5), morphology=['PSF']*n, match=RedshiftMatch(half_width_kms=2000.),
                       blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
                       separation_arcsec=np.full(n, 6.), fracflux=np.zeros(n), ood_flag_sigma=4.)


@pytest.mark.parametrize('name', POINTERS)
def test_pointer_loads_and_scores_euclid_only(name):
    import json
    from qso_pcolor.unified import UnifiedPSFModel
    bundle = json.loads((D/name).read_text())['bundle']
    assert (D/bundle).is_dir()
    m = UnifiedPSFModel.load(D/name)
    # a VIS ~ 20.5 point source with quasar-like flat Euclid colours, S/N ~ 50 per band (nanomaggies)
    f = np.array([[6.3, 6.0, 6.2, 6.5]]); v = (f/50.)**2
    s, d = _score(m, f, v, ('euclid:vis', 'euclid:y', 'euclid:j', 'euclid:h'))
    assert len(s) == 1 and s[0] is not None
    assert np.isfinite(m.quasar_probability(s)[0])


@pytest.mark.parametrize('name, parent', [('current_euclid', 'current'), ('current_euclid_magindep', 'current_magindep')])
def test_stage1_pointer_keeps_qso_density_and_euclid_selection(name, parent):
    import json
    from qso_pcolor.unified import UnifiedPSFModel
    a, b = UnifiedPSFModel.load(D/name), UnifiedPSFModel.load(D/parent)
    assert json.loads((D/json.loads((D/name).read_text())['bundle']/'manifest.json').read_text())['selection'] == 'euclid_point'
    bands = ('decals_dr9_south:g', 'decals_dr9_south:r', 'decals_dr9_south:z', 'decals_dr9_south:w1', 'decals_dr9_south:w2')
    f = np.array([[2.0, 2.4, 2.9, 4.5, 4.8], [10., 15., 19., 12., 8.]]); v = (f/30.)**2
    sa, _ = _score(a, f, v, bands); sb, _ = _score(b, f, v, bands)
    qa = np.array([s.loglike_qso_zprimary for s in sa]); qb = np.array([s.loglike_qso_zprimary for s in sb])
    assert np.allclose(qa, qb, atol=1e-9)                          # quasar colour model copied exactly
    ba = np.array([s.loglike_bkg for s in sa]); bb = np.array([s.loglike_bkg for s in sb])
    assert not np.allclose(ba, bb, atol=1e-6)                      # background thinned to the Euclid point selection
