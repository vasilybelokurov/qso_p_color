"""Reference-band surface densities for the multi-survey model.

With these the multi-survey scorer returns p_sameq and log R, which it could
not before. The checks are against independent routes: the stored counts, the
validated original prior, and densities of the same population measured
through other surveys' r bands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import MultiSurveyModel, _conditional_min_mahalanobis, load_priors
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.priors import GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch

ROOT = Path(__file__).resolve().parents[1]
PRIORS = ROOT / "models/multisurvey_priors.json"


@pytest.fixture(scope="module")
def model():
    return MultiSurveyModel.load(ROOT / "models/multisurvey.json")


@pytest.fixture(scope="module")
def priors(model):
    return load_priors(PRIORS, model)


def q_total(p, lo, hi):
    ms = np.arange(lo + 0.005, hi, 0.01)
    zs = np.linspace(0.15, 4.35, 421)
    return sum(np.trapezoid(p(zs, m), zs) * 0.01 for m in ms)


def test_every_pair_is_labelled_for_its_band_and_transform(model, priors):
    assert len(priors) >= 30
    for label, (qp, bd) in priors.items():
        for p in (qp, bd):
            assert p.meta["reference_band"] == label
            assert p.meta["transform_id"] == model.transform_id
        assert label in model.transform.bands


def test_a_file_for_another_transform_is_refused(model, tmp_path):
    d = json.loads(PRIORS.read_text())
    d["transform_id"] = "0" * 64
    (tmp_path / "p.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="transform"):
        load_priors(tmp_path / "p.json", model)


def test_background_density_is_the_stored_count_over_the_stored_area(priors):
    for label, (_, bd) in priors.items():
        n = sum(bd.counts.values())
        area = sum(bd.area.values())
        e = bd.mag_edges
        ms = np.concatenate([np.linspace(a, b, 21)[:-1] + (b - a) / 40 for a, b in zip(e[:-1], e[1:])])
        dm = np.repeat(np.diff(e) / 20, 20)
        integral = float(np.sum(bd(ms, np.zeros(ms.size), np.full(ms.size, 60.0)) * dm))
        assert integral == pytest.approx(n / area, rel=1e-6), label


def test_legacy_south_r_prior_reproduces_the_validated_original():
    ref = GridQSOPrior.load(ROOT / "models/sigma_q_south.json")
    d = json.loads(PRIORS.read_text())
    new = GridQSOPrior.from_dict(d["anchors"]["decals_dr9_south:r"]["qso_prior"])
    assert q_total(new, 17.0, 22.5) == pytest.approx(q_total(ref, 17.0, 22.5), rel=0.01)


def test_other_r_bands_see_the_same_quasar_population():
    """Different survey, same sky: r-band quasar densities within 20 %.

    Not enforced by construction -- only the Legacy-south normalisation is --
    so this checks the footprint areas and the sampling weights.
    """
    d = json.loads(PRIORS.read_text())
    ref = q_total(GridQSOPrior.from_dict(d["anchors"]["decals_dr9_south:r"]["qso_prior"]), 17, 21)
    for a in ("sdss:r", "ps1:r", "decals_dr9_north:r", "nsc:r"):
        val = q_total(GridQSOPrior.from_dict(d["anchors"][a]["qso_prior"]), 17, 21)
        assert abs(val / ref - 1) < 0.2, (a, val, ref)


def test_the_multisurvey_scorer_now_returns_a_posterior(model, priors):
    ph = Photometry(np.array([[1.9, 2.6, 3.1]]), 1 / np.array([[120.0, 150.0, 60.0]]),
                    ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z"))
    r = model.score(ph, z_primary=np.array([1.8]), match=RedshiftMatch(half_width_kms=2000.0),
                    min_bands=2, l_deg=np.array([276.337]), b_deg=np.array([60.189]),
                    priors=priors)[0]
    assert r.status == "ok" and r.reference_band == "decals_dr9_south:r"
    assert np.isfinite(r.log_r_per_unit_z) and 0 < r.p_sameq < 1
    assert r.p_sameq == pytest.approx(np.exp(r.log_r_per_unit_z) * r.dz_match_eff, rel=1e-10)
    assert np.isfinite(r.qso_ood_sigma_any_z) and np.isfinite(r.bkg_ood_sigma)


def test_vectorised_conditional_distance_matches_a_direct_solve():
    rng = np.random.default_rng(0)
    d = 5

    def mix(k):
        a = rng.normal(size=(k, d, d))
        return GaussianMixture(np.full(k, 1 / k), rng.normal(size=(k, d)),
                               a @ a.transpose(0, 2, 1) + 0.1 * np.eye(d))
    ms = [mix(3), mix(2)]
    x = rng.normal(size=(9, d)) * 2
    s = np.array([np.diag(rng.uniform(0.01, 0.3, d)) for _ in range(9)])
    obs = np.ones((9, d), bool)
    obs[2, 3] = obs[5, 1] = obs[5, 4] = False
    obs[7] = False
    obs[7, [0, 2]] = True
    obs[8] = False
    obs[8, 3] = True                                   # anchor unmeasured -> NaN
    got = _conditional_min_mahalanobis(ms, x, s, obs, anchor=0, chunk=2)
    for i in range(8):
        dims = [k for k in range(d) if obs[i, k] and k != 0]
        best = np.inf
        for m in ms:
            for mu, v in zip(m.means, m.covs):
                t = v + s[i]
                c = t[dims, 0]
                mean = mu[dims] + c / t[0, 0] * (x[i, 0] - mu[0])
                cc = t[np.ix_(dims, dims)] - np.outer(c, c) / t[0, 0]
                r = x[i, dims] - mean
                best = min(best, np.sqrt(r @ np.linalg.solve(cc, r)))
        assert got[i] == pytest.approx(best, rel=1e-10)
    assert np.isnan(got[8])
