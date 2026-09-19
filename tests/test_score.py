"""End-to-end checks of the scoring layer on a synthetic universe.

The synthetic universe has a quasar colour locus that moves with redshift, a
separate background locus, and known surface densities.  Because the truth is
known, the tests can assert calibration-like statements that no real data set
would let us check cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from qso_pcolor.background import BackgroundColourModel, fit_background_model
from qso_pcolor.features import FeatureSet
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.priors import BackgroundSurfaceDensity, EmpiricalQSOPrior
from qso_pcolor.qso_model import (
    RedshiftMatch,
    SlicedColourRedshiftModel,
    fit_sliced_model,
)
from qso_pcolor.score import score_candidates

SYSTEM = "toy_2colour"
LABELS = ("c1", "c2")


def qso_locus(z):
    """A curved, non-monotonic colour track, of the kind a linear model struggles with."""
    return np.stack([0.9 * np.sin(2.0 * z), 0.7 * np.cos(1.5 * z) - 0.3], axis=-1)


def make_universe(seed=0, n_qso=12000, n_bkg=20000):
    rng = np.random.default_rng(seed)

    z = rng.uniform(0.3, 3.0, size=n_qso)
    intrinsic = 0.08
    xq = qso_locus(z) + rng.normal(0, intrinsic, size=(n_qso, 2))
    err_q = rng.uniform(0.03, 0.12, size=n_qso)
    xq_obs = xq + rng.normal(0, err_q[:, None])
    cov_q = err_q[:, None, None] ** 2 * np.eye(2)

    xb = rng.normal([1.2, 0.9], [0.35, 0.30], size=(n_bkg, 2))
    err_b = rng.uniform(0.03, 0.12, size=n_bkg)
    xb_obs = xb + rng.normal(0, err_b[:, None])
    cov_b = err_b[:, None, None] ** 2 * np.eye(2)

    mag_q = rng.uniform(19.0, 21.0, size=n_qso)
    mag_b = rng.uniform(19.0, 21.0, size=n_bkg)
    l_b = rng.uniform(0, 360, size=n_bkg)
    b_b = rng.uniform(30, 80, size=n_bkg)
    return dict(
        z=z, xq=xq_obs, cov_q=cov_q, mag_q=mag_q,
        xb=xb_obs, cov_b=cov_b, mag_b=mag_b, l_b=l_b, b_b=b_b,
    )


@pytest.fixture(scope="module")
def models():
    u = make_universe()
    qso = fit_sliced_model(
        u["xq"], u["cov_q"], u["z"],
        z_edges=np.linspace(0.3, 3.0, 19),
        n_components=3, min_per_slice=100, overlap=0.5,
        system=SYSTEM, labels=LABELS, seed=0, max_iter=200, tol=1e-6,
    )
    bkg = fit_background_model(
        u["xb"], u["cov_b"], u["mag_b"], u["l_b"], u["b_b"],
        mag_edges=np.array([19.0, 20.0, 21.0]),
        nside=4, nside_parent=1, n_components=3, min_per_cell=400,
        n0=300.0, system=SYSTEM, labels=LABELS, seed=0, max_iter=200,
    )
    qso_prior = EmpiricalQSOPrior.build(
        u["z"], u["mag_q"], area_deg2=1000.0,
        z_edges=np.linspace(0.3, 3.0, 19),
        mag_edges=np.array([19.0, 20.0, 21.0]),
    )
    bkg_density = BackgroundSurfaceDensity.from_catalogue(
        u["mag_b"], u["l_b"], u["b_b"],
        mag_edges=np.array([19.0, 20.0, 21.0]), nside=4, nside_parent=1,
    )
    return u, qso, bkg, qso_prior, bkg_density


def make_features(x, cov, ref_mag):
    x = np.atleast_2d(x)
    n = x.shape[0]
    return FeatureSet(
        x=x,
        cov=np.atleast_3d(cov).reshape(n, 2, 2),
        observed=np.ones((n, 2), dtype=bool),
        ref_flux=np.full(n, 10.0),
        ref_mag=np.broadcast_to(np.atleast_1d(ref_mag), (n,)).astype(float),
        ref_snr=np.full(n, 20.0),
        labels=LABELS,
        flags={},
    )


def run(models, x, cov, ref_mag, z_primary, match=None, **kw):
    _, qso, bkg, qp, bd = models
    return score_candidates(
        make_features(x, cov, ref_mag),
        z_primary=np.atleast_1d(z_primary),
        l_deg=np.full(np.atleast_2d(x).shape[0], 120.0),
        b_deg=np.full(np.atleast_2d(x).shape[0], 60.0),
        qso_model=qso,
        background_model=bkg,
        match=match or RedshiftMatch(half_width_kms=2000.0),
        qso_prior=kw.pop("qso_prior", qp),
        background_density=kw.pop("background_density", bd),
        z_grid=np.linspace(0.31, 2.99, 300),
        **kw,
    )


# -- the model must know the quasar locus ---------------------------------

def test_on_locus_beats_off_locus_at_the_primary_redshift(models):
    z0 = 1.4
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    on = qso_locus(np.array([z0]))
    off = np.array([[1.2, 0.9]])  # the background locus
    s_on = run(models, on, cov, 20.0, z0)[0]
    s_off = run(models, off, cov, 20.0, z0)[0]

    assert s_on.log_bayes_factor_qz_bkg > 3.0
    assert s_off.log_bayes_factor_qz_bkg < -3.0
    # The two-class number the brief asks for: quasar-at-z0 versus background.
    assert s_on.p_sameq_vs_bkg > 0.9
    assert s_off.p_sameq_vs_bkg < 0.05
    assert s_on.p_sameq > 20 * s_off.p_sameq


def test_a_velocity_window_is_far_narrower_than_any_colour_redshift(models):
    """The central limitation, asserted rather than assumed.

    A +/-2000 km/s window at z=1.4 is dz = 0.016, while colours constrain the
    redshift to sigma_z of order 0.1.  So even for an object sitting exactly on
    the locus, almost all of the quasar intensity lands outside the window and
    ``p_sameq`` stays low: colours cannot establish a velocity-scale redshift
    match.  Widening the window to the photometric resolution recovers a high
    posterior, which is the regime where this machinery is informative.
    """
    z0 = 1.4
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    on = qso_locus(np.array([z0]))

    tight = run(models, on, cov, 20.0, z0,
                match=RedshiftMatch(half_width_kms=2000.0))[0]
    wide = run(models, on, cov, 20.0, z0, match=RedshiftMatch(dz_half_width=0.3))[0]

    assert tight.p_sameq < 0.3
    assert wide.p_sameq > 0.8
    # The evidence is identical; only the question changed.
    assert wide.log_bayes_factor_qz_bkg == pytest.approx(
        tight.log_bayes_factor_qz_bkg, abs=1e-12
    )


def test_a_quasar_at_the_wrong_redshift_is_not_evidence_for_a_pair(models):
    """The failure mode that motivates the field-quasar hypothesis."""
    z0, z_true = 1.4, 2.6
    cov = np.array([[0.004, 0.0], [0.0, 0.004]])
    x = qso_locus(np.array([z_true]))
    s = run(models, x, cov, 20.0, z0)[0]

    # It looks nothing like a background source ...
    assert s.log_bayes_factor_qz_bkg > -1e9
    # ... but the same-redshift posterior must still be small, because the
    # field-quasar hypothesis explains it far better.
    assert s.p_sameq < 0.1
    assert s.log_lambda_fieldq > s.log_lambda_sameq
    # And the two-class number is the misleading one: it has no way to know.
    assert s.p_sameq_vs_bkg > s.p_sameq


def test_photometric_redshift_recovers_the_truth(models):
    cov = np.array([[0.002, 0.0], [0.0, 0.002]])
    for z_true in (0.8, 1.6, 2.4):
        s = run(models, qso_locus(np.array([z_true])), cov, 20.0, 1.0)[0]
        assert abs(s.z_phot_mode - z_true) < 0.25


def test_p_zmatch_is_high_only_when_the_redshift_agrees(models):
    cov = np.array([[0.002, 0.0], [0.0, 0.002]])
    x = qso_locus(np.array([1.5]))
    match = RedshiftMatch(dz_half_width=0.15)
    good = run(models, x, cov, 20.0, 1.5, match=match)[0]
    bad = run(models, x, cov, 20.0, 2.5, match=match)[0]
    assert good.p_zmatch_given_qso > 0.4
    assert bad.p_zmatch_given_qso < 0.05


# -- probability algebra ---------------------------------------------------

def test_posterior_components_sum_to_one(models):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    s = run(models, qso_locus(np.array([1.2])), cov, 20.0, 1.2)[0]
    lam = np.exp(
        np.array([s.log_lambda_sameq, s.log_lambda_fieldq, s.log_lambda_bkg])
        - s.log_lambda_sameq
    )
    p = lam / lam.sum()
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(s.p_sameq, rel=1e-6)


def test_bayes_factor_is_invariant_under_a_prior_shift(models):
    """Changing only the surface densities must move the posterior, not the evidence."""
    u, qso, bkg, qp, bd = models
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    x = qso_locus(np.array([1.2]))

    base = run(models, x, cov, 20.0, 1.2)[0]

    # Scale the quasar prior up by 100; leave every likelihood untouched.
    from qso_pcolor.priors import GridQSOPrior

    boosted = GridQSOPrior(qp.z_centres, qp.mag_centres, qp.sigma * 100.0, qp.meta)
    shifted = run(models, x, cov, 20.0, 1.2, qso_prior=boosted)[0]

    assert shifted.log_bayes_factor_qz_bkg == pytest.approx(
        base.log_bayes_factor_qz_bkg, abs=1e-12
    )
    assert shifted.loglike_qso_zprimary == pytest.approx(
        base.loglike_qso_zprimary, abs=1e-12
    )
    assert shifted.p_sameq > base.p_sameq

    # The odds against the background must scale exactly with the prior ratio.
    def odds(s):
        return np.exp(s.log_lambda_sameq - s.log_lambda_bkg)

    assert odds(shifted) / odds(base) == pytest.approx(100.0, rel=1e-6)


def test_no_prior_means_no_posterior_not_a_made_up_one(models):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    s = run(models, qso_locus(np.array([1.2])), cov, 20.0, 1.2, qso_prior=None)[0]
    assert s.status == "no_prior_posterior_unavailable"
    assert np.isnan(s.p_sameq) and np.isnan(s.p_sameq_vs_bkg)
    # The prior-independent evidence is still delivered.
    assert np.isfinite(s.log_bayes_factor_qz_bkg)


# -- error handling and flags ---------------------------------------------

def test_missing_colour_is_marginalised_and_widens_the_evidence(models):
    """Losing a band must reduce discrimination, not crash or silently impute."""
    _, qso, bkg, qp, bd = models
    z0 = 1.4
    x = np.array([[1.2, 0.9]])  # background-like
    cov = np.array([[[0.01, 0.0], [0.0, 0.01]]])

    full = make_features(x, cov, 20.0)
    partial = make_features(x.copy(), cov.copy(), 20.0)
    partial.observed[0, 1] = False
    partial.x[0, 1] = np.nan

    kw = dict(
        z_primary=np.array([z0]), l_deg=np.array([120.0]), b_deg=np.array([60.0]),
        qso_model=qso, background_model=bkg,
        match=RedshiftMatch(half_width_kms=2000.0),
        qso_prior=qp, background_density=bd,
        z_grid=np.linspace(0.31, 2.99, 300),
        min_bands=1,
    )
    s_full = score_candidates(full, **kw)[0]
    s_part = score_candidates(partial, **kw)[0]
    assert s_part.n_bands_used == 1
    assert np.isfinite(s_part.log_bayes_factor_qz_bkg)
    # One colour discriminates less than two.
    assert abs(s_part.log_bayes_factor_qz_bkg) < abs(s_full.log_bayes_factor_qz_bkg)


def test_too_few_bands_gives_a_status_not_a_number(models):
    _, qso, bkg, qp, bd = models
    fs = make_features(np.array([[1.0, 1.0]]), np.eye(2)[None] * 0.01, 20.0)
    fs.observed[:] = [[True, False]]
    fs.x[0, 1] = np.nan
    s = score_candidates(
        fs, z_primary=np.array([1.4]), l_deg=np.array([120.0]), b_deg=np.array([60.0]),
        qso_model=qso, background_model=bkg, match=RedshiftMatch(dz_half_width=0.1),
        qso_prior=qp, background_density=bd, min_bands=2,
    )[0]
    assert s.status == "insufficient_photometry"
    assert np.isnan(s.p_sameq)


def test_photometric_system_mismatch_raises(models):
    _, qso, bkg, qp, bd = models
    fs = make_features(np.array([[1.0, 1.0]]), np.eye(2)[None] * 0.01, 20.0)
    with pytest.raises(ValueError, match="photometric system mismatch"):
        score_candidates(
            fs, z_primary=np.array([1.4]), l_deg=np.array([120.0]),
            b_deg=np.array([60.0]), qso_model=qso, background_model=bkg,
            match=RedshiftMatch(dz_half_width=0.1), system="sdss_ugriz",
        )


def test_feature_layout_mismatch_raises(models):
    _, qso, bkg, qp, bd = models
    fs = make_features(np.array([[1.0, 1.0]]), np.eye(2)[None] * 0.01, 20.0)
    fs.labels = ("wrong", "names")
    with pytest.raises(ValueError, match="feature layout mismatch"):
        score_candidates(
            fs, z_primary=np.array([1.4]), l_deg=np.array([120.0]),
            b_deg=np.array([60.0]), qso_model=qso, background_model=bkg,
            match=RedshiftMatch(dz_half_width=0.1),
        )


def test_out_of_distribution_colours_are_flagged(models):
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    on = run(models, qso_locus(np.array([1.4])), cov, 20.0, 1.4)[0]
    far = run(models, np.array([[8.0, -6.0]]), cov, 20.0, 1.4)[0]
    assert far.qso_ood_sigma > 10.0
    assert on.qso_ood_sigma < 3.0


def test_redshift_match_definitions_are_explicit():
    with pytest.raises(ValueError, match="exactly one"):
        RedshiftMatch()
    with pytest.raises(ValueError, match="exactly one"):
        RedshiftMatch(half_width_kms=1000.0, dz_half_width=0.1)
    m = RedshiftMatch(half_width_kms=3000.0)
    # dz = (1+z) dv / c
    assert m.half_width(2.0) == pytest.approx(3.0 * 3000.0 / 299792.458)
    assert m.describe()["kernel"] == "tophat"


def test_gaussian_match_kernel_is_broadened_by_the_primary_error():
    narrow = RedshiftMatch(dz_half_width=0.01, kernel="gaussian")
    wide = RedshiftMatch(dz_half_width=0.01, z_primary_err=0.05, kernel="gaussian")
    z = np.linspace(0.9, 1.1, 501)
    assert np.trapezoid(wide.weight(z, 1.0), z) > 3 * np.trapezoid(
        narrow.weight(z, 1.0), z
    )


# -- serialisation ---------------------------------------------------------

def test_models_round_trip_through_disk(models, tmp_path):
    _, qso, bkg, qp, bd = models
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    x = qso_locus(np.array([1.3]))

    before = run(models, x, cov, 20.0, 1.3)[0]

    qso.save(tmp_path / "q.json")
    bkg.save(tmp_path / "b.json")
    qp.save(tmp_path / "qp.json")
    bd.save(tmp_path / "bd.json")

    from qso_pcolor.priors import GridQSOPrior

    after = score_candidates(
        make_features(x, cov, 20.0),
        z_primary=np.array([1.3]), l_deg=np.array([120.0]), b_deg=np.array([60.0]),
        qso_model=SlicedColourRedshiftModel.load(tmp_path / "q.json"),
        background_model=BackgroundColourModel.load(tmp_path / "b.json"),
        match=RedshiftMatch(half_width_kms=2000.0),
        qso_prior=GridQSOPrior.load(tmp_path / "qp.json"),
        background_density=BackgroundSurfaceDensity.load(tmp_path / "bd.json"),
        z_grid=np.linspace(0.31, 2.99, 300),
    )[0]

    assert after.loglike_qso_zprimary == pytest.approx(
        before.loglike_qso_zprimary, abs=1e-10
    )
    assert after.p_sameq == pytest.approx(before.p_sameq, abs=1e-10)


def test_sliced_model_conditional_is_normalised(models):
    """p(c | Q, z) must integrate to one over colour space at any z."""
    _, qso, *_ = models
    grid = np.linspace(-4, 4, 161)
    g1, g2 = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([g1.ravel(), g2.ravel()], axis=1)
    for z in (0.6, 1.5, 2.7):
        lp = qso.log_p_colour_given_z(pts, None, np.array([z]))[:, 0]
        p = np.exp(lp).reshape(grid.size, grid.size)
        total = np.trapezoid(np.trapezoid(p, grid, axis=1), grid)
        assert total == pytest.approx(1.0, abs=2e-3)
