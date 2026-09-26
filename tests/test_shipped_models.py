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
    assert s.log_bayes_factor_qz_bkg == pytest.approx(3.90, abs=0.05)
    assert s.log_r_per_unit_z == pytest.approx(-1.73, abs=0.05)
    assert s.p_sameq == pytest.approx(6.6e-3, rel=0.05)
    assert s.p_outlier < 1e-4
    assert s.p_sameq == pytest.approx(np.exp(s.log_r_per_unit_z) * s.dz_match_eff, rel=1e-10)
    assert "outside_both_models" not in s.quality_flags


def test_the_model_files_describe_one_consistent_model():
    model, priors, out = load()
    assert len(model.transform.bands) == 41
    assert {"decals_dr9_south:w1", "decals_dr9_north:w2", "allwise:w1"} <= set(model.transform.bands)
    assert [m.labels for m in model.background_marginals] == [LEGACY[:3], LEGACY]
    assert out.transform_id == model.transform_id and out.kappa > out.kappa_min
    assert "*" in out.fractions
    assert len(priors) == 37
    for label, (qp, bd) in priors.items():
        assert qp.meta["transform_id"] == model.transform_id == bd.meta["transform_id"]


def test_the_unmodelled_term_dominates_every_shipped_component():
    from qso_pcolor.outlier import min_dominant_kappa

    model, _, out = load()
    bands = model.transform.bands
    extra = []
    for m in model.background_marginals:
        idx = np.array([bands.index(b) for b in m.labels])
        for v in m.covs:
            e = np.zeros((len(bands), len(bands)))
            e[np.ix_(idx, idx)] = v
            extra.append(e)
    k = min_dominant_kappa(out.cov, list(model.qso.mixtures) + [model.background],
                           np.array(extra))
    assert k < 1.0


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
