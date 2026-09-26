"""The repository must be usable from a clone with no database access.

Three files ship and are the model: ``models/multisurvey.json`` (quasar and
field densities, luptitude transform, band schema), ``models/multisurvey_priors.json``
(surface densities per reference band) and ``models/multisurvey_outlier.json``
(the unmodelled hypothesis). Together they give log BF, log R and p_sameq for a
candidate measured in any of the seven surveys. The first test is the README's
offline example, executed; if its numbers drift the README is stale.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = tuple(f"decals_dr9_south:{b}" for b in ("g", "r", "z", "w1", "w2"))


def load():
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors

    model = MultiSurveyModel.load(ROOT / "models/multisurvey.json")
    return (model, load_priors(ROOT / "models/multisurvey_priors.json", model),
            MultiSurveyOutlier.load(ROOT / "models/multisurvey_outlier.json"))


def score_readme_candidate(bands=LEGACY, flux=(1.9, 2.6, 3.1, 11.0, 14.0),
                           ivar=(120.0, 150.0, 60.0, 8.0, 3.0), outlier=True):
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy

    model, priors, out = load()
    phot = Photometry(np.array([flux]), 1.0 / np.array([ivar]), bands)
    return model.score(
        phot, z_primary=np.array([1.8]), match=RedshiftMatch(half_width_kms=2000.0),
        min_bands=2, l_deg=np.array([276.337]), b_deg=np.array([60.189]),
        priors=priors, outlier=out if outlier else None,
        blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
        separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]), ood_flag_sigma=4.0,
    )[0]


def test_the_readme_example():
    s = score_readme_candidate()
    assert s.status == "ok" and s.reference_band == "decals_dr9_south:r"
    assert s.surveys_used == ("decals",)
    # the README quotes these
    assert s.log_bayes_factor_qz_bkg == pytest.approx(3.91, abs=0.05)
    assert s.log_r_per_unit_z == pytest.approx(-1.75, abs=0.05)
    assert s.p_sameq == pytest.approx(6.5e-3, rel=0.05)
    assert s.p_outlier < 1e-2                         # 1.4e-3: near the loci U is minor
    assert s.p_sameq == pytest.approx(np.exp(s.log_r_per_unit_z) * s.dz_match_eff, rel=1e-10)
    assert "outside_both_models" not in s.quality_flags


def test_the_model_files_describe_one_consistent_model():
    model, priors, out = load()
    assert len(model.transform.bands) == 41
    assert {"decals_dr9_south:w1", "decals_dr9_north:w2", "allwise:w1"} <= set(model.transform.bands)
    assert [m.labels for m in model.background_marginals] == [LEGACY[:3], LEGACY]
    assert out.transform_id == model.transform_id
    assert out.family == "student_t" and out.nu > 0
    assert "*" in out.fractions
    assert len(priors) == 37
    for label, (qp, bd) in priors.items():
        assert qp.meta["transform_id"] == model.transform_id == bd.meta["transform_id"]


def test_the_unmodelled_term_outweighs_both_models_far_from_their_loci():
    """The Student-t tail is a power law, so far enough out it exceeds every
    Gaussian component whatever its width. Checked on the README candidate
    pushed away from both loci in several directions."""
    from qso_pcolor.multisurvey import _ConditionalQSO
    from qso_pcolor.multisurvey_data import Photometry

    model, _, out = load()
    bands = model.transform.bands
    a = bands.index("decals_dr9_south:r")
    cq = _ConditionalQSO(model.qso, a)
    u_mod = out.conditional(a, "decals_dr9_south:r", model.qso.system)
    base = Photometry(np.array([[1.9, 2.6, 3.1, 11.0, 14.0]]),
                      1.0 / np.array([[120.0, 150.0, 60.0, 8.0, 3.0]]), LEGACY).align(bands)
    f = model.transform(base)
    rng = np.random.default_rng(0)
    for _ in range(6):
        d = np.zeros(len(bands)); d[[bands.index(b) for b in LEGACY if b != LEGACY[1]]] = rng.normal(size=4)
        d /= np.linalg.norm(d)
        x = f.x + 25.0 * d                     # 25 luptitudes from the candidate
        lq = cq._log_p_slices(x, f.cov, f.observed).max()
        lb = model.background_log_prob(x, f.cov, f.observed, a)[0]
        lu = u_mod.log_prob(x, f.cov, observed=f.observed)[0]
        assert lu > lq and lu > lb, (lu, lq, lb)


def test_infrared_only_input_gets_a_posterior():
    s = score_readme_candidate(bands=("allwise:w1", "allwise:w2"), flux=(40.0, 30.0),
                               ivar=(2.0, 0.5))
    assert s.reference_band == "allwise:w1" and s.status == "ok"
    assert np.isfinite(s.log_r_per_unit_z)


def test_an_archived_model_is_not_the_model():
    """The retired files live under models/archive/, never at a top-level path."""
    for name in ("qso_south_full", "background_south_global", "sigma_q_south",
                 "multisurvey_joint_20260921"):
        assert not (ROOT / f"models/{name}.json").exists()
