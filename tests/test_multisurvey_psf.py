"""The PSF contract is independent of band selection, including singleton input."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qso_pcolor.multisurvey import MultiSurveyOutlier
from qso_pcolor.multisurvey_baseline import PSFMultiSurveyBaseline
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.priors import GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch
from qso_pcolor.score import BlendPolicy
from qso_pcolor.spatial import SpatialSurfaceDensity
from test_joint_spatial import model_with_sky


def baseline():
    model = model_with_sky()
    model.meta.update(population="psf", run_id="synthetic")
    priors = {}
    for label in ("sdss:u", "allwise:w2", "vhs:ks"):
        meta = dict(population="psf", model_run_id="synthetic", reference_band=label,
                    transform_id=model.transform_id)
        qp = GridQSOPrior(np.array([.5, 2.5]), np.array([10., 30.]), np.ones((2, 2)), meta=meta)
        bd = SpatialSurfaceDensity(4, 2, np.array([0., 40.]), {}, {}, np.array([2.]), 100., meta)
        priors[label] = qp, bd
    outlier = MultiSurveyOutlier(np.full(41, 21.), np.eye(41), 1., 0., model.transform.bands,
        model.transform_id, {"*": ([0., 40.], [.01])},
        meta=dict(model_run_id="synthetic"), family="student_t", nu=2.)
    return PSFMultiSurveyBaseline(model, priors, outlier,
                                  dict(bundle_id="synthetic", min_abs_b_deg=25.))


def kwargs():
    return dict(z_primary=1.5, l_deg=0., b_deg=45., match=RedshiftMatch(half_width_kms=2000.),
                blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
                ood_flag_sigma=4., separation_arcsec=6., fracflux=.05)


def test_single_band_uses_population_prior_and_has_no_colour_evidence():
    model = baseline()
    # No optical measurement, and no reference r.
    phot = Photometry([[3.], [3.], [3.]], [[.01], [.01], [.01]], ("allwise:w2",))
    scores, decision = model.score(phot, morphology=np.array(["PSF", "EXP", "unknown"]), **kwargs())
    assert decision["eligible"].tolist() == [True, False, False]
    assert scores[1:] == [None, None]
    score = scores[0]
    assert score.log_bayes_factor_qz_bkg == pytest.approx(0., abs=1e-12)
    assert "no_colour_information" in score.quality_flags
    # Flat Q intensity=1 per z, support width=2, field intensity=2.
    assert score.p_sameq == pytest.approx(score.dz_match_eff / 4.)
    assert score.config_hash and score.background_nside == 4


def test_psf_selection_and_blend_support_policies_cannot_be_bypassed():
    model = baseline()
    phot = Photometry([[3.]], [[.01]], ("allwise:w2",))
    args = kwargs()
    args.pop("separation_arcsec")
    scores, decision = model.score(phot, morphology=["PSF"], **args)
    assert scores[0].status == "blended_not_scored" and not decision["eligible"][0]
    args = kwargs(); args["blend_policy"] = BlendPolicy(3., .2, action="flag")
    with pytest.raises(ValueError, match="excluding blend"):
        model.score(phot, morphology=["PSF"], **args)
    args = kwargs(); args["b_deg"] = 10.
    scores, decision = model.score(phot, morphology=["PSF"], **args)
    assert scores == [None] and decision["reason"][0] == "outside_latitude"


def test_missing_prior_retains_likelihood_but_cannot_enter_ranking():
    model = baseline()
    phot = Photometry([[3.]], [[.01]], ("sdss:g",))
    scores, decision = model.score(phot, morphology=["PSF"], **kwargs())
    assert scores[0].status == "no_prior_posterior_unavailable"
    assert scores[0].log_bayes_factor_qz_bkg == pytest.approx(0., abs=1e-12)
    assert not decision["eligible"][0] and np.isnan(scores[0].p_sameq)


def test_local_refit_preserves_arbitrary_bands_and_enforces_cone_scope():
    model = baseline()
    phot = Photometry(np.full((8, 2), 3.), np.full((8, 2), .01), ("allwise:w2", "vhs:ks"))
    # Six usable field stars: the first is extended and the second a known QSO.
    args = dict(morphology=np.array(["EXP"] + ["PSF"] * 7),
        known_quasar=np.array([False, True] + [False] * 6),
        l_deg=np.linspace(.1, .2, 8), b_deg=np.full(8, 45.),
        centre_l_deg=0., centre_b_deg=45., radius_deg=.5,
        excluded_positions=np.array([[0., 45.], [.01, 45.]]), exclusion_arcsec=1.,
        usable_area_deg2={"allwise:w2": .5}, colour_n0=10., density_n0=100.,
        max_iter=100, tol=1e-7)
    local = model.fit_local(phot, **args)
    assert local.model.spatial_background.meta["n_used"] == 6
    assert local.model.spatial_background.meta["density_bands_refitted"] == ["allwise:w2"]
    assert local.priors["vhs:ks"] is model.priors["vhs:ks"]
    assert np.array_equal(local.model.background.covs, model.model.background.covs)
    single = phot.subset([2]).keep_bands(("allwise:w2",))
    scores, decision = local.score(single, morphology=["PSF"], **kwargs())
    assert scores[0].background_model_mode == "local" and decision["eligible"][0]
    assert scores[0].background_density_level == 3
    outside = kwargs(); outside["l_deg"] = 180.
    with pytest.raises(ValueError, match="outside"):
        local.score(single, morphology=["PSF"], **outside)
    args["usable_area_deg2"] = {}
    with pytest.raises(ValueError, match="usable areas"):
        model.fit_local(phot, **args)


def load_script(name, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_psf_training_selection_does_not_require_a_photometric_band(monkeypatch):
    module = load_script("recover_multisurvey_psf", monkeypatch)
    rows = dict(type=np.array(["PSF", "PSF", "EXP", "DUP"]), maskbits=np.array([0, 1, 0, 0]),
                ra=np.ones(4), dec=np.zeros(4))
    assert module.psf_rows(rows).tolist() == [True, False, False, False]


def test_psf_background_fit_records_bounds_and_preserves_field_holdouts(tmp_path, monkeypatch):
    module = load_script("recover_multisurvey_psf", monkeypatch)
    model = model_with_sky()
    model.spatial_background = None
    model_path = tmp_path / "source.json"; model.save(model_path)
    rng = np.random.default_rng(551)
    n = 48
    fields = np.repeat(np.arange(8), 6)
    held = fields >= 6
    np.savez(tmp_path / "background.npz", bands=model.transform.bands,
             flux=rng.uniform(1., 3., (n, 41)), variance=np.full((n, 41), .01),
             held=held, field=fields)
    cfg = dict(source_model=str(model_path), seed=20260928, max_background_fit=24,
        background_selection_fraction=.2,
        min_band_training=1, background_k_candidates=[1], selection_max_iter=2,
        max_iter=2, tol=1e-5, regularization=.001)
    _, record = module.fit_part(cfg, tmp_path, "background")
    assert np.asarray(record["bounds"]).shape == (41, 2)
    assert not held[record["fitted_rows"]].any()
    assert record["heldout_fields"] == [6, 7]


def test_zero_count_bins_are_merged_without_changing_the_total(monkeypatch):
    module = load_script("complete_multisurvey_psf", monkeypatch)
    values = np.array([1.1, 1.2, 4.1, 4.2, 4.3])
    edges = module.merge_empty_bins(values, np.arange(7.))
    counts = np.histogram(values, edges)[0]
    assert (counts > 0).all() and counts.sum() == len(values)
    assert edges[0] == 0. and edges[-1] == 6.


def test_faint_limit_excludes_faint_and_missing_limit_band_before_scoring():
    model = baseline()
    model.model.meta["faint_limit"] = dict(bands=["decals_dr9_south:r", "decals_dr9_north:r"], min_snr=10.)
    bands = ("decals_dr9_south:r", "allwise:w2")
    phot = Photometry([[100., 3.], [5., 3.], [np.nan, 3.]], [[1., .01], [1., .01], [np.inf, .01]], bands)
    scores, decision = model.score(phot, morphology=["PSF"]*3, **kwargs())
    assert decision["reason"][1] == "too_faint" and decision["reason"][2] == "no_faint_limit_band"
    assert scores[1] is None and scores[2] is None and not decision["eligible"][1:].any()
    assert decision["accepted"].tolist() == [True, False, False]
