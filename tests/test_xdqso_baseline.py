"""Science eligibility and provenance of the actual published XDQSO bundle."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from qso_pcolor.baseline import XDQSOBaseline, file_sha256, resolve_bundle
from qso_pcolor.legacy import dereddened_relative_fluxes
from qso_pcolor.qso_model import RedshiftMatch
from qso_pcolor.score import BlendPolicy


BUNDLE = Path(__file__).resolve().parents[1] / "models/legacy_psf_xdqso/current"


@pytest.fixture(scope="module")
def baseline():
    return XDQSOBaseline.load(BUNDLE)


def catalogue(hemisphere="south", magnitude=20., g_multiplier=1.):
    rflux = 10 ** ((22.5 - magnitude) / 2.5)
    rows = dict(ra=np.array([180.]), dec=np.array([0. if hemisphere == "south" else 50.]),
                release=np.array([9010 if hemisphere == "south" else 9011]),
                type=np.array(["PSF"]), maskbits=np.zeros(1, int))
    for band, ratio in zip(("g", "r", "z", "w1", "w2"), (.8, 1., 1.2, 4., 6.)):
        rows[f"flux_{band}"] = np.array([rflux * ratio])
        rows[f"flux_ivar_{band}"] = np.array([100.])
        rows[f"nobs_{band}"] = np.array([3])
        rows[f"mw_transmission_{band}"] = np.array([1.])
    rows["flux_g"] *= g_multiplier
    return rows


def score(baseline, rows, **kwargs):
    opts = dict(z_primary=1.8, match=RedshiftMatch(half_width_kms=2000.),
                blend_policy=BlendPolicy(3., .2), ood_flag_sigma=4.,
                separation_arcsec=6., fracflux=.05)
    opts.update(kwargs)
    return baseline.score_rows(rows, **opts)


@pytest.mark.parametrize("hemisphere", ["south", "north"])
def test_published_bundle_clean_row_and_identity(baseline, hemisphere):
    scores, decision = score(baseline, catalogue(hemisphere))
    s = scores[0]
    assert decision["eligible"].tolist() == [True]
    assert s.status == "ok" and s.model_manifest_id == baseline.bundle_id
    assert s.p_sameq == pytest.approx(np.exp(s.log_r_per_unit_z) * s.dz_match_eff)


@pytest.mark.parametrize("hemisphere", ["south", "north"])
def test_far_from_both_populations_is_never_a_ranked_candidate(baseline, hemisphere):
    # This high-S/N bright source previously had P(Q)=1 and ln R about +2.
    scores, decision = score(baseline, catalogue(hemisphere, 18.5, 8.), z_primary=.65)
    s = scores[0]
    assert min(s.qso_ood_sigma_any_z, s.bkg_ood_sigma) > 4.
    assert s.status == "outside_both_models"
    assert "outside_both_models" in s.quality_flags
    assert not decision["eligible"][0]
    assert decision["reason"][0] == "status:outside_both_models"
    for name in ("log_r_per_unit_z", "p_sameq", "p_sameq_vs_bkg", "p_zmatch_given_qso", "p_outlier"):
        assert np.isnan(getattr(s, name))
    assert np.isfinite(s.loglike_qso_zprimary) and np.isfinite(s.loglike_bkg)
    assert s.dz_match_eff > 0


@pytest.mark.parametrize("missing", ["separation_arcsec", "fracflux"])
def test_missing_blend_measurement_refuses_science_score(baseline, missing):
    scores, decision = score(baseline, catalogue(), **{missing: None})
    assert scores[0].status == "blended_not_scored"
    assert not decision["eligible"][0]


@pytest.mark.parametrize("policy", [None, BlendPolicy(3.), BlendPolicy(3., .2, "flag")])
def test_blend_policy_cannot_be_disabled(baseline, policy):
    with pytest.raises(ValueError, match="BlendPolicy"):
        score(baseline, catalogue(), blend_policy=policy)


@pytest.mark.parametrize("threshold", [0., -1., np.nan, np.inf])
def test_ood_policy_cannot_be_disabled(baseline, threshold):
    with pytest.raises(ValueError, match="ood_flag_sigma"):
        score(baseline, catalogue(), ood_flag_sigma=threshold)


def test_dereddened_adapter_against_explicit_jacobian():
    rows = catalogue()
    bands = ("g", "r", "z", "w1", "w2")
    transmission = np.array([.7, .8, .9, .95, .98])
    rows["flux_g"] *= -1
    for b, t in zip(bands, transmission):
        rows[f"mw_transmission_{b}"][:] = t
    fs, ok = dereddened_relative_fluxes(rows, "south")
    f = np.array([rows[f"flux_{b}"][0] for b in bands]) / transmission
    variance = .01 / transmission ** 2
    other = [0, 2, 3, 4]
    jac = np.zeros((4, 5))
    for i, j in enumerate(other):
        jac[i, j] = 1 / f[1]
        jac[i, 1] = -f[j] / f[1] ** 2
    np.testing.assert_allclose(fs.x[0], f[other] / f[1])
    np.testing.assert_allclose(fs.cov[0], (jac * variance) @ jac.T)
    assert ok[0] and fs.x[0, 0] < 0
    rows["nobs_w2"][:] = 0
    missing, ok = dereddened_relative_fluxes(rows, "south")
    assert ok[0] and not missing.observed[0, -1]
    np.testing.assert_allclose(missing.cov[0, :3, :3], fs.cov[0, :3, :3])


def test_component_selection_identity_is_checked_even_with_valid_hash(tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(resolve_bundle(BUNDLE), root)
    path = root / "south_qso_prior.json"
    d = json.loads(path.read_text())
    d["meta"]["selection_id"] = "another-selection"
    path.write_text(json.dumps(d))
    man = json.loads((root / "manifest.json").read_text())
    man["files"][path.name] = file_sha256(path)
    (root / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="different selection"):
        XDQSOBaseline.load(root)
