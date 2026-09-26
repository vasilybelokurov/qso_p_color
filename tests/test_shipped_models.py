"""The repository must be usable from a clone with no database access.

Three files ship: the quasar colour model, a footprint-average background
colour model with its surface density, and the quasar surface density. Together
they are everything the scorer needs, so a colleague can compute log BF, log R
and p_sameq for a candidate immediately. This test is the README's offline
example, executed.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANDS = ("g", "r", "z", "w1", "w2")


def score_readme_candidate(g_factor: float = 1.0, outlier: bool = False):
    """The README candidate; ``g_factor`` scales its g flux to make it an outlier."""
    from qso_pcolor.background import BackgroundColourModel
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.outlier import OutlierModel
    from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
    from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
    from qso_pcolor.score import BlendPolicy, score_candidates

    qso = SlicedColourRedshiftModel.load(ROOT / "models/qso_south_full.json")
    bkg = BackgroundColourModel.load(ROOT / "models/background_south_global.json")
    dens = BackgroundSurfaceDensity.load(ROOT / "models/background_density_south_global.json")
    prior = GridQSOPrior.load(ROOT / "models/sigma_q_south.json")

    tr = RelativeFluxTransform(reference_band="r")
    flux = np.array([[1.9 * g_factor, 2.6, 3.1, 11.0, 14.0]])
    ivar = np.array([[120.0, 150.0, 60.0, 8.0, 3.0]])
    trans = np.array([[0.97, 0.98, 0.99, 1.0, 1.0]])
    f, v = deredden(flux, ivar, trans)
    feat = tr(f, v, BANDS)

    rows = score_candidates(
        feat, z_primary=np.array([1.8]),
        l_deg=np.array([276.337]), b_deg=np.array([60.189]),
        qso_model=qso, background_model=bkg, background_density=dens,
        qso_prior=prior, match=RedshiftMatch(half_width_kms=2000.0),
        blend_policy=BlendPolicy(min_separation_arcsec=3.0, max_fracflux=0.2),
        separation_arcsec=np.array([6.0]), fracflux=np.array([0.05]),
        outlier_model=(OutlierModel.load(ROOT / "models/outlier_south.json")
                       if outlier else None),
    )
    return rows[0]


def test_shipped_files_exist_and_load():
    for name in ("qso_south_full", "background_south_global",
                 "background_density_south_global", "sigma_q_south", "outlier_south"):
        p = ROOT / f"models/{name}.json"
        assert p.exists(), f"{p} must ship with the repository"
    s = score_readme_candidate()
    assert s.status == "ok", s.status
    for k in ("log_bayes_factor_qz_bkg", "log_r_per_unit_z", "p_sameq",
              "p_zmatch_given_qso", "frac_norm_outside_support"):
        assert np.isfinite(getattr(s, k)), k
    assert 0.0 <= s.p_sameq <= 1.0
    # the README quotes these; a drift here means the README is stale
    assert abs(s.log_bayes_factor_qz_bkg - 3.42) < 0.05, s.log_bayes_factor_qz_bkg
    assert abs(s.log_r_per_unit_z - (-2.52)) < 0.05, s.log_r_per_unit_z
    u = score_readme_candidate(outlier=True)       # the README's recommended call
    assert abs(u.log_bayes_factor_qz_bkg - 3.42) < 0.05, u.log_bayes_factor_qz_bkg
    assert abs(u.log_r_per_unit_z - (-2.52)) < 0.05, u.log_r_per_unit_z


def test_shipped_background_records_its_provenance():
    from qso_pcolor.background import BackgroundColourModel

    m = BackgroundColourModel.load(ROOT / "models/background_south_global.json").meta
    assert m["n_cones"] == 8 and m["skipped"] == []
    assert all(c["n_raw"] >= m["full_cone_min_sources"] for c in m["cones"])
    assert all(c["n_known_quasars"] > 0 for c in m["cones"]), "every cone must lie in the spectroscopic footprint"
    assert all(abs(c["b"]) > 25 for c in m["cones"])
    assert "no morphology cut" in m["selection"] and "maskbits = 0" in m["selection"]


if __name__ == "__main__":
    s = score_readme_candidate()
    print(f"log BF {s.log_bayes_factor_qz_bkg:+.2f}  log R {s.log_r_per_unit_z:+.2f}  "
          f"p_sameq {s.p_sameq:.2e}  p(z in W|Q) {s.p_zmatch_given_qso:.3f}  status {s.status}")
