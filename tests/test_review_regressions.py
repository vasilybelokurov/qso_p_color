"""Regressions for defects found in the 2026-09-19 external review.

Each test reproduces a specific failure that the code once had, with the
numbers from the original report, so that a future change cannot quietly
reintroduce it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qso_pcolor.background import BackgroundColourModel
from qso_pcolor.features import AsinhColourTransform, deredden
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
from qso_pcolor.features import FeatureSet
from qso_pcolor.score import score_candidates
from qso_pcolor.xd import fit_xd

# the synthetic-universe fixtures live in test_score.py
from test_score import make_features, models, qso_locus  # noqa: F401

BANDS = ("g", "r", "i", "z")


# ---------------------------------------------------------------- helpers

def _toy(mean=0.0):
    g = GaussianMixture(np.array([1.0]), np.array([[mean]]), np.array([[[1.0]]]),
                        labels=("c",))
    qso = SlicedColourRedshiftModel(np.array([0.0, 0.5, 1.0]), [g, g, g],
                                    np.array([9, 9, 9]), "toy", ("c",))
    bkg = BackgroundColourModel(1, 1, np.array([0.0, 30.0]), {}, {}, [g], {}, {},
                                1.0, "background", "toy", ("c",))
    fs = FeatureSet(np.array([[0.0]]), np.zeros((1, 1, 1)), np.ones((1, 1), bool),
                    np.array([1.0]), np.array([20.0]), np.array([99.0]), ("c",), {})
    bd = BackgroundSurfaceDensity(1, 1, np.array([0.0, 30.0]), {(0, 0): 1.0}, {0: 1.0})
    return qso, bkg, fs, bd


def _score(prior, z_grid, z0, match, mean=0.0):
    qso, bkg, fs, bd = _toy(mean)
    return score_candidates(
        fs, z_primary=np.array([z0]), l_deg=np.array([0.0]), b_deg=np.array([80.0]),
        qso_model=qso, background_model=bkg, match=match, qso_prior=prior,
        background_density=bd, z_grid=z_grid, min_bands=1,
    )[0]


# ---------------------------------------------------------------- scoring

def test_window_crossing_the_support_edge_is_clipped():
    """A window half outside the trained range must count only the half inside.

    The closed-form shortcut multiplied by the whole window area regardless,
    inventing quasars where the model knows of none: p_zmatch came out 0.2 when
    the truth is 0.1.
    """
    pr = GridQSOPrior(np.array([0.0, 0.5, 1.0]), np.array([10.0, 30.0]), np.ones((3, 2)))
    s = _score(pr, np.array([0.0, 0.5, 1.0]), 0.0, RedshiftMatch(dz_half_width=0.1))
    assert s.p_zmatch_given_qso == pytest.approx(0.1, rel=1e-6)
    assert s.dz_match_eff == pytest.approx(0.1, rel=1e-6)   # half of 0.2


def test_match_probability_never_exceeds_one():
    """A spiky prior on a coarse grid once returned p_zmatch = 1.6."""
    pr = GridQSOPrior(
        np.array([0.0, 0.49, 0.5, 0.51, 1.0]), np.array([10.0, 30.0]),
        np.array([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], float),
    )
    s = _score(pr, np.linspace(0, 1, 5), 0.5, RedshiftMatch(dz_half_width=0.2))
    assert 0.0 <= s.p_zmatch_given_qso <= 1.0


def test_extreme_surface_density_does_not_underflow():
    """Scaling by the colour likelihood alone drove log_lambda_sameq to -inf."""
    pr = GridQSOPrior(np.array([0.0, 0.5, 1.0]), np.array([10.0, 30.0]),
                      np.full((3, 2), 1e300))
    s = _score(pr, np.linspace(0, 1, 201), 0.5,
               RedshiftMatch(dz_half_width=0.01), mean=40.0)
    assert np.isfinite(s.log_lambda_sameq)
    assert s.log_lambda_sameq == pytest.approx(-114.055, abs=0.01)


def test_velocity_window_is_not_quantised_by_the_grid():
    """A top hat a few grid steps wide was integrated as a sampled mask.

    The area used must be the true window area, not a multiple of the grid step.
    """
    g = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([[[1.0]]]),
                        labels=("c",))
    qso = SlicedColourRedshiftModel(np.array([0.4, 1.7, 3.0]), [g, g, g],
                                    np.array([9, 9, 9]), "toy", ("c",))
    bkg = BackgroundColourModel(1, 1, np.array([0.0, 30.0]), {}, {}, [g], {}, {},
                                1.0, "background", "toy", ("c",))
    fs = FeatureSet(np.array([[0.0]]), np.zeros((1, 1, 1)), np.ones((1, 1), bool),
                    np.array([1.0]), np.array([20.0]), np.array([99.0]), ("c",), {})
    bd = BackgroundSurfaceDensity(1, 1, np.array([0.0, 30.0]), {(0, 0): 1.0}, {0: 1.0})
    pr = GridQSOPrior(np.array([0.4, 1.7, 3.0]), np.array([10.0, 30.0]), np.ones((3, 2)))

    # z0 chosen well inside the support so no clipping is involved: the only
    # thing under test is that the window area is exact, not grid-quantised.
    for z0 in (1.0, 1.8, 2.2, 2.6):
        m = RedshiftMatch(half_width_kms=2000.0)
        s = score_candidates(
            fs, z_primary=np.array([z0]), l_deg=np.array([0.0]),
            b_deg=np.array([80.0]), qso_model=qso, background_model=bkg, match=m,
            qso_prior=pr, background_density=bd,
            z_grid=np.linspace(0.4, 3.0, 261), min_bands=1,
        )[0]
        assert s.dz_match_eff == pytest.approx(m.effective_width(z0), rel=1e-6)


def test_background_model_feature_order_is_checked():
    qso, bkg, fs, bd = _toy()
    bad = BackgroundColourModel(1, 1, np.array([0.0, 30.0]), {}, {}, bkg.global_,
                                {}, {}, 1.0, "background", "toy", ("wrong",))
    with pytest.raises(ValueError, match="background model"):
        score_candidates(fs, z_primary=np.array([0.5]), l_deg=np.array([0.0]),
                         b_deg=np.array([80.0]), qso_model=qso, background_model=bad,
                         match=RedshiftMatch(dz_half_width=0.1))


# ---------------------------------------------------------------- priors

def test_surveyed_but_empty_cells_stay_in_the_area_denominator():
    bd = BackgroundSurfaceDensity(1, 1, np.array([0.0, 1.0]), {(0, 0): 10.0},
                                  {0: 1.0, 1: 1.0})
    assert bd._global(0) == pytest.approx(5.0)


def test_area_must_be_supplied_rather_than_assumed():
    """Assuming a full HEALPix pixel understated Sigma_B by 17x for a 1 deg cone."""
    with pytest.raises(ValueError, match="area_per_pixel or total_area_deg2"):
        BackgroundSurfaceDensity.from_catalogue(
            np.array([20.0]), np.array([10.0]), np.array([80.0]),
            mag_edges=np.array([19.0, 21.0]),
        )


def test_quasar_prior_is_zero_outside_its_magnitude_range():
    pr = GridQSOPrior(np.array([0.5, 1.0]), np.array([19.0, 21.0]),
                      np.array([[1.0, 2.0], [1.0, 2.0]]))
    assert pr(np.array([0.75]), 30.0)[0] == 0.0
    assert not pr.in_support(np.array([0.75]), 30.0)[0]
    assert pr(np.array([0.75]), 20.0)[0] == pytest.approx(1.5)


# ---------------------------------------------------------------- XD

def test_covariance_survives_a_large_mean_offset():
    """E[bb^T] - mu mu^T lost all precision at 1e8, returning variance 0."""
    x = (1e8 + np.array([-1.0, 1.0]))[:, None]
    init = GaussianMixture(np.array([1.0]), np.array([[1e8]]), np.array([[[1.0]]]))
    r = fit_xd(x, None, n_components=1, init=init, max_iter=1, tol=0)
    assert float(r.mixture.covs[0, 0, 0]) == pytest.approx(1.0, rel=1e-6)


def test_reported_likelihood_describes_the_returned_mixture():
    """It used to belong to the model before the final M step (-51.4 vs -1.4)."""
    x = np.array([[-1.0], [1.0]])
    init = GaussianMixture(np.array([1.0]), np.array([[10.0]]), np.array([[[1.0]]]))
    r = fit_xd(x, None, n_components=1, init=init, max_iter=1, tol=0)
    assert r.mean_loglike == pytest.approx(float(np.mean(r.mixture.log_prob(x))), abs=1e-9)


# ---------------------------------------------------------------- features

def test_asinh_does_not_spread_nan_across_good_colours():
    """0 * NaN in the colour matrix poisoned colours built from good bands."""
    f, v = deredden(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[0.0, 1.0, 1.0, 1.0]]),
                    np.ones((1, 4)))
    fs = AsinhColourTransform(softening={b: 0.5 for b in BANDS})(f, v, BANDS)
    assert not np.isnan(fs.x[0][fs.observed[0]]).any()
    assert list(fs.observed[0]) == [False, True, True]


# ---------------------------------------------------------------- gaussmix

def test_asymmetric_covariance_is_rejected():
    """Cholesky reads the lower triangle, so this was silently reinterpreted."""
    with pytest.raises(ValueError, match="symmetric"):
        GaussianMixture(np.array([1.0]), np.zeros((1, 2)),
                        np.array([[[1.0, 100.0], [0.0, 1.0]]]))


def test_deserialisation_rejects_a_non_positive_definite_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        GaussianMixture.from_dict(
            {"weights": [1.0], "means": [[0.0]], "covs": [[[-1.0]]]}
        )


def test_model_marginal_quantiles_need_a_grid_that_spans_the_mixture():
    """A fixed quantile grid clipped the WISE features in a diagnostic figure.

    Guards the general rule: a numerical quantile of a mixture must be taken on
    a grid derived from the mixture, not on a guessed range. With a grid ending
    at 4 the 84th percentile of a component at 10 silently returned 4.
    """
    from scipy.stats import norm

    mix = GaussianMixture(np.array([1.0]), np.array([[10.0]]), np.array([[[4.0]]]))

    def quantile(q, grid):
        cdf = norm.cdf((grid - mix.means[0, 0]) / np.sqrt(mix.covs[0, 0, 0]))
        return float(np.interp(q, cdf, grid))

    truth = 10.0 + 2.0 * norm.ppf(0.84)
    bad = quantile(0.84, np.linspace(-1.0, 4.0, 1200))
    mu, sd = mix.means[0, 0], np.sqrt(mix.covs[0, 0, 0])
    good = quantile(0.84, np.linspace(mu - 6 * sd, mu + 6 * sd, 4000))

    assert bad == pytest.approx(4.0, abs=1e-6)        # silently clamped
    assert good == pytest.approx(truth, rel=1e-3)


# ---------------------------------------------------------- background scope

def test_known_quasars_are_removed_from_the_background():
    """Measured: quasars are 1% of the background overall but 72% of it at
    quasar colours, so leaving them in makes the background compete with the
    quasar locus using quasars."""
    from qso_pcolor.data import drop_known_quasars

    # 0.0001 deg = 0.36", inside the 1" radius; 0.001 deg = 3.6", outside it.
    ra = np.array([180.0, 180.0001, 180.001, 181.0])
    dec = np.zeros(4)
    keep = drop_known_quasars(ra, dec, np.array([180.0]), np.array([0.0]))
    assert list(keep) == [False, False, True, True]
    # no quasars supplied -> nothing removed
    assert drop_known_quasars(ra, dec, np.array([]), np.array([])).all()


def test_compare_backgrounds_reports_how_much_the_choice_moved_the_answer(models):
    """Two background models, same candidates: the deltas must be real numbers."""
    from qso_pcolor.score import compare_backgrounds

    _, qso, bkg, qp, bd = models
    x = qso_locus(np.array([1.4]))
    fs = make_features(x, np.array([[0.01, 0.0], [0.0, 0.01]]), 20.0)

    # a second density that is deliberately 10x sparser
    sparse = BackgroundSurfaceDensity(
        bd.nside, bd.nside_parent, bd.mag_edges,
        {k: v / 10.0 for k, v in bd.counts.items()}, dict(bd.area),
    )
    out = compare_backgrounds(
        fs, models={"a": bkg, "b": bkg}, densities={"a": bd, "b": sparse},
        l_deg=np.array([120.0]), b_deg=np.array([60.0]),
        z_primary=np.array([1.4]), qso_model=qso,
        match=RedshiftMatch(half_width_kms=2000.0), qso_prior=qp,
        z_grid=np.linspace(0.31, 2.99, 300),
    )
    assert set(out["rows"]) == {"a", "b"}
    # the colour likelihood is identical (same colour model) ...
    assert out["deltas"]["loglike_bkg"][0] == pytest.approx(0.0, abs=1e-12)
    # ... but a sparser background raises p_sameq, so the delta is non-zero
    assert out["deltas"]["p_sameq"][0] > 0
    assert np.isfinite(out["deltas"]["log_r_per_unit_z"][0])


def test_mismatched_model_and_density_keys_are_rejected(models):
    from qso_pcolor.score import compare_backgrounds

    _, qso, bkg, qp, bd = models
    with pytest.raises(ValueError, match="same keys"):
        compare_backgrounds(
            make_features(qso_locus(np.array([1.4])), np.eye(2)[None] * 0.01, 20.0),
            models={"a": bkg}, densities={"b": bd},
            l_deg=np.array([120.0]), b_deg=np.array([60.0]),
            z_primary=np.array([1.4]), qso_model=qso,
            match=RedshiftMatch(dz_half_width=0.1),
        )


# ------------------------------------------- classifier review, 2026-09-19

def test_local_background_area_excludes_masked_sky():
    """Masked counts divided by unmasked area understates Sigma_B.

    The cone query keeps only maskbits == 0 sources. If the area is the full
    pi R^2, the background looks sparser than it is and every quasar posterior
    is inflated -- the same class of error as the 17x Sigma_B bug.
    """
    import inspect

    from qso_pcolor import background, data

    src = inspect.getsource(background.fit_local_background)
    assert "mask_fraction" in src
    assert "1.0 - mask_fraction" in src
    # and the query must not pre-filter, or the fraction is unmeasurable
    assert "AND maskbits = 0" not in data._LS_BACKGROUND_QUERY


def test_training_holdout_is_reserved_before_fitting():
    """The per-channel comparison must not evaluate on the training set."""
    import pathlib

    src = pathlib.Path("scripts/train_qso_model.py").read_text()
    assert "is_held" in src and "held_blocks" in src
    # fits on the complement of the holdout ...
    assert "fit_idx = np.flatnonzero(ok)[~is_held]" in src
    # ... and evaluates only on the holdout
    assert "(channel[ok] == tag) & is_held" in src


def test_photometric_system_names_are_release_aware():
    """north and south are different systems; the names must say which."""
    import pathlib

    fig = pathlib.Path("scripts/make_method_figures.py").read_text()
    train = pathlib.Path("scripts/train_qso_model.py").read_text()
    assert 'SYSTEM = "ls_dr9_south_grzw"' in fig
    assert 'f"ls_dr9_{args.system}_grzw"' in train
    # the ambiguous name must not survive anywhere
    assert '"ls_dr9_grzw"' not in fig
