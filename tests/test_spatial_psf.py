"""Required spatial behaviour, independently measurable on controlled skies."""
import json
from copy import copy

import healpy as hp
import numpy as np
import pytest

from qso_pcolor.background import BackgroundColourModel, galactic_healpix
from qso_pcolor.baseline import XDQSOBaseline
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.spatial import (BackgroundAdaptation, SpatialSurfaceDensity,
    component_log_prob, effective_weights, fit_component_weights,
    fit_local_background, fit_spatial_background, cone_within_cell)
from qso_pcolor.score import BlendPolicy
from qso_pcolor.qso_model import RedshiftMatch


def synthetic():
    rng = np.random.default_rng(91)
    mix = GaussianMixture(np.array([.5, .5]), np.array([[-3.], [3.]]), np.full((2, 1, 1), .16), ("g/r",))
    n = 4000
    component = np.r_[rng.random(n) > .9, rng.random(n) > .1].astype(int)
    x = mix.means[component] + rng.normal(0, .5, (2 * n, 1))
    cov = np.full((2 * n, 1, 1), .09)
    obs = np.ones_like(x, bool)
    l, b = np.repeat([0., 180.], n), np.full(2 * n, 45.)
    base = BackgroundColourModel(1, 1, np.array([19., 21.]), {}, {}, [mix], {}, {}, 1.,
                                 "background", "test", mix.labels)
    data = dict(m=np.full(2 * n, 20.), l=l, b=b, cone=np.repeat([0, 1], n),
        nonq=np.ones(2 * n), sampling_weight=np.ones(2 * n),
        log_components=component_log_prob(mix, x, cov, obs))
    cones = [dict(l=0., b=45., area=1., nonq_counts=[4000.], sampling_weight=1.),
             dict(l=180., b=45., area=2., nonq_counts=[4000.], sampling_weight=1.)]
    return base, data, cones, x, cov, obs


def test_sky_changes_colour_and_density_with_the_correct_direction():
    base, data, cones, x, cov, obs = synthetic()
    bg, den = fit_spatial_background(base, data, cones, nside=4, nside_parent=2,
        colour_n0=100., density_n0=100., density_edges=np.array([19., 21.]),
        global_density=np.array([1500.]), max_iter=200, tol=1e-8, meta={})
    a, b = [effective_weights(bg, 20., l, 45.) for l in (0., 180.)]
    assert a[0] > .85 and b[0] < .15
    rates = den(np.array([20., 20.]), np.array([0., 180.]), np.array([45., 45.]))
    np.testing.assert_allclose(rates, [2000., 1000.], rtol=.01)
    assert den.level([20., 20.], [0., 180.], [45., 45.]).tolist() == [0, 0]
    # Same photometry and magnitude, different sky: a real density change.
    lp = bg.log_prob(np.array([[-3.], [-3.]]), None, np.array([20., 20.]),
                     np.array([0., 180.]), np.array([45., 45.]))
    assert lp[0] - lp[1] > 1.5


def test_missing_cell_falls_back_and_observed_empty_cell_is_not_missing():
    p = int(galactic_healpix(np.array([0.]), np.array([45.]), 4)[0])
    density = SpatialSurfaceDensity(4, 2, [19., 21.], {(p, 0): 0.}, {p: 1.}, [100.], 100., {})
    rate = density([20.], [0.], [45.])[0]
    assert 0 < rate < 100
    l, b = hp.pix2ang(4, np.arange(192), nest=True, lonlat=True)
    levels = density.level(np.full(192, 20.), l, b)
    assert set(levels) == {0, 1, 2}
    np.testing.assert_allclose(density(np.full(192, 20.), l, b)[levels == 2], 100.)
    restored = SpatialSurfaceDensity.from_dict(density.to_dict())
    np.testing.assert_allclose(restored(np.full(192, 20.), l, b), density(np.full(192, 20.), l, b))


def test_cone_boundaries_are_checked_beyond_the_centre():
    l, b = hp.pix2ang(4, 30, nest=True, lonlat=True)
    assert cone_within_cell(l, b, .3, nside=4, boundary_factor=256)
    # A centre exactly on an edge still belongs to one pixel, but its cone
    # overlaps neighbours. Discard the entire cone so its area remains valid.
    edge = hp.boundaries(4, 30, step=8, nest=True)[:, 4]
    l, b = hp.vec2ang(edge, lonlat=True)
    assert not cone_within_cell(float(l[0]), float(b[0]), .3, nside=4, boundary_factor=256)


def test_withholding_cone_recomputes_area_weights_without_mutating_input():
    from scripts.fit_spatial_psf_background import subset
    data = dict(cone=np.array([1, 1, 2, 3]), sampling_weight=np.ones(4))
    cones = [dict(cone=1, cell=10, area=1., stratum_area=100.),
             dict(cone=2, cell=10, area=3., stratum_area=100.),
             dict(cone=3, cell=11, area=2., stratum_area=200.)]
    rows, selected = subset(data, cones, [1, 3])
    np.testing.assert_allclose(rows['sampling_weight'], [100., 100., 100.])
    assert [c['sampling_weight'] for c in selected] == [100., 100.]
    assert 'sampling_weight' not in cones[0]
    np.testing.assert_array_equal(data['sampling_weight'], np.ones(4))


def test_active_bundle_uses_galactic_position_and_preserves_global_shapes():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from qso_pcolor.legacy import dereddened_relative_fluxes
    baseline = XDQSOBaseline.load('models/legacy_psf_xdqso/current')
    assert baseline.manifest['background_adaptation']['mode'] == 'spatial'
    original = XDQSOBaseline.load('models/legacy_psf_xdqso/8e2a27c013ed')
    for h, part in baseline.parts.items():
        bg = part['background']
        assert bg.nside == baseline.manifest['background_adaptation']['selected'][h]['nside']
        assert bg.local and bg.parent and len(part['background_density'].area) > 1
        for a, b in zip(bg.global_, original.parts[h]['background'].global_):
            np.testing.assert_array_equal(a.means, b.means)
            np.testing.assert_array_equal(a.covs, b.covs)
            np.testing.assert_array_equal(a.weights, b.weights)
        for (_, j), mix in bg.local.items():
            np.testing.assert_array_equal(mix.means, bg.global_[j].means)
            np.testing.assert_array_equal(mix.covs, bg.global_[j].covs)
    bg = baseline.parts['south']['background']
    cells = sorted({cell for cell, mb in bg.local})
    l, b = hp.pix2ang(bg.nside, cells, nest=True, lonlat=True)
    coords = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic').icrs
    # Identical photometry placed in different southern sky cells.
    source = local_rows()
    rows = {k: np.repeat(v[:1], len(cells)) for k, v in source.items()}
    rows['ra'], rows['dec'] = coords.ra.deg, coords.dec.deg
    keep = baseline.selection.decide(rows)['accepted'] & (abs(b) >= 25)
    rows = {k: v[keep] for k, v in rows.items()}
    l, b = l[keep], b[keep]
    scores, decision = baseline.score_rows(rows, z_primary=1.8,
        match=RedshiftMatch(half_width_kms=2000.), blend_policy=BlendPolicy(3., .2),
        ood_flag_sigma=4., separation_arcsec=6., fracflux=.05)
    fs, _ = dereddened_relative_fluxes(rows, 'south')
    expected = bg.log_prob(fs.x, fs.cov, fs.ref_mag, l, b, observed=fs.observed)
    outlier = baseline.parts['south']['outlier']
    eta = outlier.fraction_at(fs.ref_mag)
    with np.errstate(divide='ignore'):
        expected = np.logaddexp(np.log1p(-eta) + expected,
            np.log(eta) + outlier.log_prob(fs.x, fs.cov, observed=fs.observed))
    actual = np.array([s.loglike_bkg for s in scores])
    np.testing.assert_allclose(actual, expected)
    assert np.ptp(actual) > .01
    assert np.ptp(baseline.parts['south']['background_density'](fs.ref_mag, l, b)) > 0
    assert np.ptp([s.log_r_per_unit_z for s in scores if s.status == 'ok']) > 0
    np.testing.assert_allclose([s.loglike_qso_zprimary for s in scores], scores[0].loglike_qso_zprimary)
    for s in scores:
        assert s.background_model_mode == 'spatial' and s.background_nside == bg.nside
        assert s.model_manifest_id == baseline.bundle_id and s.config_hash
        assert json.loads(s.scoring_config_json)['match']['half_width_kms'] == 2000.


def test_pooling_changes_predictions_in_populated_cells():
    base, data, cones, *_ = synthetic()
    args = dict(nside=4, nside_parent=2, density_edges=np.array([19., 21.]),
                global_density=np.array([1500.]), max_iter=200, tol=1e-8, meta={})
    a, da = fit_spatial_background(base, data, cones, colour_n0=10., density_n0=10., **args)
    b, db = fit_spatial_background(base, data, cones, colour_n0=1e8, density_n0=1e8, **args)
    assert effective_weights(a, 20., 0., 45.)[0] > .85
    assert effective_weights(b, 20., 0., 45.)[0] == pytest.approx(.5, abs=.001)
    assert da([20.], [0.], [45.])[0] > db([20.], [0.], [45.])[0] + 400


def test_component_likelihoods_keep_covariance_and_missing_dimensions():
    mix = GaussianMixture(np.array([.3, .7]), np.array([[0., 1.], [3., -1.]]),
                          np.array([[[1., .3], [.3, 2.]], [[2., -.2], [-.2, 1.]]]))
    x = np.array([[.2, .8], [2., np.nan]])
    cov = np.array([[[.4, .1], [.1, .3]], [[.2, 0.], [0., 0.]]])
    obs = np.array([[True, True], [True, False]])
    lp = component_log_prob(mix, x, cov, obs)
    from scipy.special import logsumexp
    np.testing.assert_allclose(logsumexp(lp + np.log(mix.weights), axis=1), mix.log_prob(x, cov, observed=obs))
    w, _ = fit_component_weights(np.empty((0, 2)), mix.weights, np.empty(0), max_iter=10, tol=1e-6)
    np.testing.assert_array_equal(w, mix.weights)


def test_zero_weight_components_are_valid_for_spatial_populations():
    mix = GaussianMixture(np.array([1., 0.]), np.array([[0.], [10.]]), np.ones((2, 1, 1)))
    np.testing.assert_allclose(mix.log_prob(np.array([[0.], [2.]])),
                               -.5 * np.log(2 * np.pi) - .5 * np.array([0., 4.]))


def local_rows():
    rng = np.random.default_rng(55)
    n = 500
    rows = dict(ra=180. + rng.uniform(-.05, .05, n), dec=rng.uniform(-.05, .05, n),
        release=np.full(n, 9010), type=np.full(n, "PSF"), maskbits=np.zeros(n, int),
        known_quasar=np.zeros(n, bool))
    for band, flux in zip(("g", "r", "z", "w1", "w2"), (8., 10., 12., 40., 60.)):
        rows[f"flux_{band}"] = flux * rng.uniform(.9, 1.1, n)
        rows[f"flux_ivar_{band}"] = np.full(n, 100.)
        rows[f"nobs_{band}"] = np.full(n, 3)
        rows[f"mw_transmission_{band}"] = np.ones(n)
    rows["known_quasar"][-10:] = True
    rows["type"][2] = "EXP"
    rows["ra"][:2] = [180., 180.005]; rows["dec"][:2] = 0.
    return rows


def test_local_psf_refit_excludes_targets_preserves_models_and_records_scope(tmp_path):
    baseline = XDQSOBaseline.load("models/legacy_psf_xdqso/current")
    rows = local_rows()
    fitted = fit_local_background(baseline, rows, hemisphere="south", centre_ra=180., centre_dec=0.,
        radius_deg=.1, usable_area_deg2=.025, exclude_ra=np.array([180., 180.005]),
        exclude_dec=np.array([0., 0.]), exclusion_arcsec=3., colour_n0=100., density_n0=100.,
        max_iter=200, tol=1e-5, min_density_fraction=.05)
    assert fitted.meta["n_used"] <= 487
    assert fitted.meta["n_known"] == 10
    path = tmp_path / "local.json"; fitted.save(path)
    model = XDQSOBaseline.load("models/legacy_psf_xdqso/current", background=path)
    assert model.parts["south"]["qso"] is not None
    assert model.bundle_id != baseline.bundle_id
    assert model.parts["south"]["background_density"].level([20.], [0.], [45.])[0] == 3
    candidate = {k: v[:1].copy() for k, v in rows.items()}
    candidate["ra"][:] = 200.
    result, decision = model.score_rows(candidate, z_primary=1.8,
        match=RedshiftMatch(half_width_kms=3000.), blend_policy=BlendPolicy(3., .2),
        ood_flag_sigma=4., separation_arcsec=6., fracflux=.05)
    assert result == [None] and decision["reason"][0] == "outside_local_background"
    # Corrupted content cannot silently alter either counts or sky scope.
    d = json.loads(path.read_text()); d["meta"]["radius_deg"] = 100.
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        BackgroundAdaptation.load(path)
    changed = copy(baseline)
    changed.manifest = dict(baseline.manifest, bundle_id="different", base_bundle_id="different")
    with pytest.raises(ValueError, match="another baseline"):
        fitted.apply(changed)


@pytest.mark.parametrize("area", [0., -1., np.nan, .5])
def test_local_area_must_be_measured_and_fit_inside_cone(area):
    baseline = XDQSOBaseline.load("models/legacy_psf_xdqso/current")
    with pytest.raises(ValueError, match="area"):
        fit_local_background(baseline, local_rows(), hemisphere="south", centre_ra=180., centre_dec=0.,
            radius_deg=.1, usable_area_deg2=area, exclude_ra=np.array([180., 180.005]),
            exclude_dec=np.array([0., 0.]), exclusion_arcsec=3., colour_n0=100., density_n0=100.,
            max_iter=100, tol=1e-5, min_density_fraction=.05)
