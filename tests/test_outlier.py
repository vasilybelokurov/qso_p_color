"""The unmodelled hypothesis and the tail diagnostics.

Far from every component of both colour models, the quasar/background ratio is
set by which Gaussian tail is wider and runs away.  These tests pin the remedy:
a broad, normalised fourth density that dominates every component's tail, with
its share of the field fitted by likelihood, plus two distances that say when
an object sits in that regime.  Each numerical piece is checked against an
independent route (scipy, a direct solve, a 1-D optimiser), not against itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.outlier import (OutlierModel, envelope_covariance, fit_outlier_fraction,
                                min_dominant_kappa, mixture_moments)

import test_score as ts

ROOT = Path(__file__).resolve().parents[1]


def random_mixture(rng, k, d, wide_axis=None, wide=50.0):
    means = rng.normal(0, 1, (k, d))
    a = rng.normal(0, 0.4, (k, d, d))
    covs = a @ np.swapaxes(a, 1, 2) + 0.05 * np.eye(d)
    if wide_axis is not None:
        covs[0, wide_axis, wide_axis] += wide**2      # one rare, very wide component
    return GaussianMixture(np.full(k, 1.0 / k), means, covs)


def p_quasar(row):
    lam = [row.log_lambda_sameq, row.log_lambda_fieldq, row.log_lambda_bkg]
    lo = row.log_lambda_out
    every = lam + ([lo] if np.isfinite(lo) else [])
    return float(np.exp(np.logaddexp(lam[0], lam[1]) - logsumexp(every)))


# -- building blocks -------------------------------------------------------------

def test_envelope_dominates_the_reference_and_every_component():
    rng = np.random.default_rng(1)
    ref = random_mixture(rng, 5, 4)
    others = [random_mixture(rng, 6, 4, wide_axis=2), random_mixture(rng, 3, 4)]
    _, base = mixture_moments([ref])
    env = envelope_covariance(base, others + [ref])
    for c in [base] + [v for m in others + [ref] for v in m.covs]:
        assert np.linalg.eigvalsh(env - c).min() > -1e-9
    assert min_dominant_kappa(env, others + [ref]) <= 1.0 + 1e-12
    # anisotropic: far narrower than the isotropic scaling that would also dominate
    iso = min_dominant_kappa(base, others + [ref]) ** 2 * base
    assert np.linalg.slogdet(env)[1] < np.linalg.slogdet(iso)[1] - 1.0


def test_kappa_below_kappa_min_is_refused():
    with pytest.raises(ValueError, match="dominate"):
        OutlierModel(np.zeros(2), np.eye(2), [0.01], [0.0, 1.0], kappa=1.0, kappa_min=1.2,
                     system="s", labels=("a", "b"))


def test_outlier_density_matches_scipy_with_noise_and_a_missing_band():
    rng = np.random.default_rng(2)
    mean = rng.normal(size=3)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3)
    um = OutlierModel(mean, cov, [0.01], [0.0, 30.0], kappa=2.0, kappa_min=1.0,
                      system="s", labels=("a", "b", "c"))
    x = rng.normal(size=(5, 3))
    s = np.array([np.diag(rng.uniform(0.1, 1.0, 3)) for _ in range(5)])
    obs = np.ones((5, 3), bool)
    obs[2, 1] = False
    got = um.log_prob(x, s, observed=obs)
    for i in range(5):
        idx = np.flatnonzero(obs[i])
        ref = multivariate_normal(mean[idx], (cov + s[i])[np.ix_(idx, idx)]).logpdf(x[i, idx])
        assert got[i] == pytest.approx(ref, abs=1e-10)


def test_fit_outlier_fraction_is_the_likelihood_maximum_and_recovers_the_truth():
    rng = np.random.default_rng(3)
    n, eta_true = 200_000, 0.01
    is_out = rng.random(n) < eta_true
    x = np.where(is_out, rng.normal(0, 20, n), rng.normal(0, 1, n))
    lb = multivariate_normal(0, 1).logpdf(x)
    lu = multivariate_normal(0, 400).logpdf(x)
    eta = fit_outlier_fraction(lb, lu)
    nll = lambda e: -np.mean(np.logaddexp(np.log1p(-e) + lb, np.log(e) + lu))  # noqa: E731
    opt = minimize_scalar(nll, bounds=(1e-8, 0.5), method="bounded",
                          options={"xatol": 1e-10})
    assert eta == pytest.approx(opt.x, rel=1e-4)
    assert abs(eta - eta_true) < 4 * np.sqrt(eta_true * (1 - eta_true) / n)


def test_min_mahalanobis_matches_a_direct_solve():
    rng = np.random.default_rng(4)
    mix = random_mixture(rng, 4, 3)
    x = rng.normal(size=(7, 3)) * 3
    s = np.array([np.diag(rng.uniform(0.01, 0.5, 3)) for _ in range(7)])
    obs = np.ones((7, 3), bool)
    obs[3, 0] = obs[5, 2] = False
    got = mix.min_mahalanobis(x, s, observed=obs, chunk=2)
    for i in range(7):
        idx = np.flatnonzero(obs[i])
        ref = min(np.sqrt((x[i, idx] - mu[idx]) @ np.linalg.solve(
            (v + s[i])[np.ix_(idx, idx)], x[i, idx] - mu[idx]))
            for mu, v in zip(mix.means, mix.covs))
        assert got[i] == pytest.approx(ref, rel=1e-10)


# -- the scorer on the synthetic universe -----------------------------------------

@pytest.fixture(scope="module")
def models():
    u = ts.make_universe()
    qso = ts.fit_sliced_model(
        u["xq"], u["cov_q"], u["z"], z_edges=np.linspace(0.3, 3.0, 19),
        n_components=3, min_per_slice=100, overlap=0.5,
        system=ts.SYSTEM, labels=ts.LABELS, seed=0, max_iter=200, tol=1e-6)
    bkg = ts.fit_background_model(
        u["xb"], u["cov_b"], u["mag_b"], u["l_b"], u["b_b"],
        mag_edges=np.array([19.0, 20.0, 21.0]), nside=4, nside_parent=1,
        n_components=3, min_per_cell=400, n0=300.0, system=ts.SYSTEM,
        labels=ts.LABELS, seed=0, max_iter=200)
    qp = ts.EmpiricalQSOPrior.build(
        u["z"], u["mag_q"], area_deg2=1000.0, z_edges=np.linspace(0.3, 3.0, 19),
        mag_edges=np.array([19.0, 20.0, 21.0]))
    bd = ts.BackgroundSurfaceDensity.from_catalogue(
        u["mag_b"], u["l_b"], u["b_b"], mag_edges=np.array([19.0, 20.0, 21.0]),
        nside=4, nside_parent=1, total_area_deg2=1000.0)
    return u, qso, bkg, qp, bd


def outlier_for(models, eta):
    _, qso, bkg, _, _ = models
    check = list(qso.mixtures) + list(bkg.global_) + list(bkg.parent.values()) \
        + list(bkg.local.values())
    return OutlierModel.build(list(bkg.global_), check, kappa=1.05,
                              fraction=np.full(2, eta), mag_edges=bkg.mag_edges,
                              system=ts.SYSTEM, labels=ts.LABELS)


def test_zero_fraction_reproduces_the_three_hypothesis_scorer(models):
    u = models[0]
    x, cov = u["xq"][:40], u["cov_q"][:40]
    a = ts.run(models, x, cov, 20.0, u["z"][:40])
    b = ts.run(models, x, cov, 20.0, u["z"][:40], outlier_model=outlier_for(models, 0.0))
    for ra, rb in zip(a, b):
        for k in ("loglike_bkg", "log_bayes_factor_qz_bkg", "log_r_per_unit_z", "p_sameq",
                  "p_sameq_vs_bkg", "log_lambda_bkg", "p_zmatch_given_qso"):
            assert getattr(rb, k) == pytest.approx(getattr(ra, k), rel=1e-12, abs=1e-300,
                                                   nan_ok=True), k
        assert rb.status == ra.status
        assert rb.p_outlier == 0.0 or (np.isnan(rb.p_outlier) and np.isnan(ra.p_sameq))
    assert sum(r.status == "ok" for r in a) >= 30


def test_four_hypothesis_posteriors_sum_to_one(models):
    u = models[0]
    rows = ts.run(models, np.vstack([u["xq"][:20], u["xb"][:20]]),
                  np.concatenate([u["cov_q"][:20], u["cov_b"][:20]]), 20.0, 1.2,
                  outlier_model=outlier_for(models, 1e-3))
    for r in rows:
        lam = [r.log_lambda_sameq, r.log_lambda_fieldq, r.log_lambda_bkg, r.log_lambda_out]
        tot = logsumexp(lam)
        assert r.p_sameq == pytest.approx(np.exp(lam[0] - tot), rel=1e-10)
        assert r.p_outlier == pytest.approx(np.exp(lam[3] - tot), rel=1e-10)
        assert r.p_sameq == pytest.approx(np.exp(r.log_r_per_unit_z) * r.dz_match_eff, rel=1e-10)


def test_near_the_loci_the_outlier_term_changes_nothing_measurable(models):
    u = models[0]
    x, cov = u["xq"][:200], u["cov_q"][:200]
    a = ts.run(models, x, cov, 20.0, u["z"][:200])
    b = ts.run(models, x, cov, 20.0, u["z"][:200], outlier_model=outlier_for(models, 1e-3))
    d = np.array([rb.log_r_per_unit_z - ra.log_r_per_unit_z for ra, rb in zip(a, b)])
    assert np.array_equal(np.isnan(d), [np.isnan(r.log_r_per_unit_z) for r in a])
    d = d[np.isfinite(d)]
    assert d.size >= 150
    assert np.all(d <= 1e-12)                  # an extra explanation can only lower R
    assert np.median(np.abs(d)) < 1e-3


def test_a_colour_outlier_goes_to_unmodelled_not_to_quasar(models):
    um = outlier_for(models, 1e-3)
    # walk outward from the background locus; errors fixed
    ray = np.array([1.2, 0.9]) + np.outer(np.array([4, 8, 16, 32, 64]), [0.8, -0.6])
    cov = np.broadcast_to(0.05**2 * np.eye(2), (ray.shape[0], 2, 2))
    rows = ts.run(models, ray, cov, 20.0, 1.2, outlier_model=um)
    p_out = np.array([r.p_outlier for r in rows])
    assert np.all(np.diff(p_out) >= -1e-12)
    assert p_out[-1] > 0.999 and p_quasar(rows[-1]) < 1e-3
    assert rows[-1].log_bayes_factor_qz_bkg < 0     # the field, with U, explains it better
    assert rows[-1].qso_ood_sigma_any_z > 20 and rows[-1].bkg_ood_sigma > 20


def test_outside_both_models_needs_an_explicit_threshold(models):
    far = np.array([[40.0, -30.0]])
    cov = 0.05**2 * np.eye(2)[None]
    assert "outside_both_models" not in ts.run(models, far, cov, 20.0, 1.2)[0].quality_flags
    r = ts.run(models, far, cov, 20.0, 1.2, ood_flag_sigma=5.0)[0]
    assert "outside_both_models" in r.quality_flags
    near = ts.run(models, models[0]["xq"][:1], models[0]["cov_q"][:1], 20.0,
                  models[0]["z"][:1], ood_flag_sigma=5.0)[0]
    assert "outside_both_models" not in near.quality_flags


def test_any_z_distance_never_exceeds_the_z0_distance(models):
    u = models[0]
    rows = ts.run(models, u["xq"][:100], u["cov_q"][:100], 20.0, 0.5)
    assert all(r.qso_ood_sigma_any_z <= r.qso_ood_sigma + 1e-12 for r in rows)


def test_background_distance_covers_the_local_and_parent_mixtures(models):
    _, _, bkg, _, _ = models
    assert bkg.local or bkg.parent, "the toy background must have a hierarchy to test"
    u = models[0]
    x, cov, m = u["xb"][:300], u["cov_b"][:300], u["mag_b"][:300]
    l, b = u["l_b"][:300], u["b_b"][:300]
    got = bkg.ood_score(x, cov, m, l, b)
    glob = np.array([bkg.global_[int(j)].min_mahalanobis(x[i:i+1], cov[i:i+1])[0]
                     for i, j in enumerate(bkg.mag_bin(m))])
    assert np.all(got <= glob + 1e-12) and np.any(got < glob - 1e-6)


def test_the_outlier_model_round_trips(tmp_path, models):
    um = outlier_for(models, 2e-4)
    um.save(tmp_path / "u.json")
    back = OutlierModel.load(tmp_path / "u.json")
    assert np.array_equal(back.cov, um.cov) and np.array_equal(back.fraction, um.fraction)
    assert back.kappa == um.kappa and back.labels == um.labels


def test_the_outlier_model_must_match_the_system_and_layout(models):
    u = models[0]
    um = outlier_for(models, 1e-3)
    bad = OutlierModel(um.mean, um.cov, um.fraction, um.mag_edges, um.kappa, um.kappa_min,
                       "other_system", um.labels)
    with pytest.raises(ValueError, match="system"):
        ts.run(models, u["xq"][:1], u["cov_q"][:1], 20.0, 1.0, outlier_model=bad)
    bad = OutlierModel(um.mean, um.cov, um.fraction, um.mag_edges, um.kappa, um.kappa_min,
                       ts.SYSTEM, ("c2", "c1"))
    with pytest.raises(ValueError, match="layout"):
        ts.run(models, u["xq"][:1], u["cov_q"][:1], 20.0, 1.0, outlier_model=bad)


def test_the_multisurvey_scorer_refuses_an_unconditional_outlier_model(models):
    from qso_pcolor.multisurvey import MultiSurveyModel

    ms = MultiSurveyModel.load(ROOT / "models/multisurvey.json")
    with pytest.raises(ValueError, match="conditional"):
        ms.score(None, z_primary=np.array([1.0]), match=ts.RedshiftMatch(half_width_kms=2000.0),
                 min_bands=2, l_deg=np.array([0.0]), b_deg=np.array([60.0]),
                 outlier_model=outlier_for(models, 1e-3))


# -- the shipped models ---------------------------------------------------------

def test_shipped_outlier_model_dominates_every_shipped_component():
    from qso_pcolor.background import BackgroundColourModel
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    um = OutlierModel.load(ROOT / "models/archive/original_legacy_south/outlier_south.json")
    qso = SlicedColourRedshiftModel.load(ROOT / "models/archive/original_legacy_south/qso_south_full.json")
    bkg = BackgroundColourModel.load(ROOT / "models/archive/original_legacy_south/background_south_global.json")
    check = list(qso.mixtures) + list(bkg.global_)
    assert min_dominant_kappa(um.cov, check) < 1.0
    assert um.system == qso.system and um.labels == qso.labels
    assert np.array_equal(um.mag_edges, bkg.mag_edges)
    assert np.all((um.fraction > 0) & (um.fraction < 1e-2))
    m = um.meta
    assert m["chosen_by"].startswith("max held-out") and set(m["set_A"]).isdisjoint(m["set_B"])
    assert min(m["set_A"] + m["set_B"]) >= 8, "cones 00-07 are the background's own fields"


def _readme(g_factor=1.0, outlier=False):
    from test_archived_original import score_readme_candidate
    return score_readme_candidate(g_factor=g_factor, outlier=outlier)


def test_shipped_models_send_a_g_band_outlier_to_unmodelled():
    before, after = _readme(8.0), _readme(8.0, outlier=True)
    assert p_quasar(before) > 0.999                     # the defect, as measured
    assert after.p_outlier > 0.5 and p_quasar(after) < 0.5
    assert after.log_bayes_factor_qz_bkg < before.log_bayes_factor_qz_bkg - 10
    # and the README candidate itself does not move
    a, b = _readme(), _readme(outlier=True)
    assert abs(b.log_r_per_unit_z - a.log_r_per_unit_z) < 0.01
    assert abs(b.log_bayes_factor_qz_bkg - a.log_bayes_factor_qz_bkg) < 0.01
    assert b.p_outlier < 1e-6
