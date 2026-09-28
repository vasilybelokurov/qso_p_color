"""The active saved bundle must expose the full requirement from a clean clone."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.multisurvey_data import band_labels

ROOT = Path(__file__).resolve().parents[1]


def test_active_bundle_has_one_psf_model_all_bands_priors_and_spatial_background():
    baseline = PSFMultiSurveyBaseline.load(ROOT / "models/multisurvey_psf/current")
    assert baseline.model.transform.bands == band_labels()
    assert set(baseline.priors) == set(band_labels())
    assert not baseline.model.background_marginals
    assert baseline.model.spatial_background.nside == 4
    assert baseline.model.spatial_background.nside_parent == 2
    assert baseline.model.qso.meta["n_fit"] == 55994
    assert baseline.model.qso.meta["n_holdout"] == 12120
    for label, (qp, density) in baseline.priors.items():
        assert qp.meta["reference_band"] == label == density.meta["reference_band"]
        assert density.nside == 4


def test_active_optical_only_example_and_every_single_band():
    baseline = PSFMultiSurveyBaseline.load(ROOT / "models/multisurvey_psf/current")
    args = dict(morphology=["PSF"], z_primary=1.8, l_deg=276.337, b_deg=60.189,
        match=RedshiftMatch(half_width_kms=2000.),
        blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
        separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.)
    phot = Photometry([[1.9, 2.6, 3.1]], [[1/120, 1/150, 1/60]],
        ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z"))
    scores, decision = baseline.score(phot, **args)
    assert scores[0].status == "ok" and decision["eligible"][0]
    assert scores[0].bands_used == phot.bands and scores[0].n_bands_used == 3
    assert scores[0].p_sameq == pytest.approx(np.exp(scores[0].log_r_per_unit_z) * scores[0].dz_match_eff)
    bands = band_labels()
    flux = np.full((41, 41), np.nan); variance = np.full_like(flux, np.inf)
    flux[np.arange(41), np.arange(41)] = 3.
    variance[np.arange(41), np.arange(41)] = .01
    scores, _ = baseline.score(Photometry(flux, variance, bands), **args)
    for label, score in zip(bands, scores):
        assert score.bands_used == (label,) and score.n_bands_used == 1
        assert score.status != "insufficient_photometry"
        assert score.log_bayes_factor_qz_bkg == pytest.approx(0., abs=1e-12)
        assert "no_colour_information" in score.quality_flags


def test_bundle_integrity_refuses_modified_population_priors(tmp_path):
    root = ROOT / "models/multisurvey_psf"
    pointer = json.loads((root / "current").read_text())
    shutil.copytree(root / pointer["bundle"], tmp_path / "bundle")
    (tmp_path / "bundle/priors.json").write_text("{}")
    with pytest.raises(ValueError, match="content does not match"):
        PSFMultiSurveyBaseline.load(tmp_path / "bundle")
