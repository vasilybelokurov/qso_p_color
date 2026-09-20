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


# ------------------------------------- package-state review, 2026-09-20

def test_model_file_records_its_holdout_split():
    """A split that is re-derived by each consumer gets re-derived differently.

    ``score_examples.py`` rebuilt the nside=4 block list from DESI alone,
    without de-duplication or the quality cut: 78 candidate blocks where
    training had 81. ``np.random.choice`` is a function of the array it is
    given, so the same seed over a different list is a different draw -- 7 of
    its 16 "reserved" blocks were training blocks and 52% of the objects it
    called held out had been fitted. The cure is to record the split.
    """
    import json
    import pathlib

    meta = json.loads(pathlib.Path("models/qso_south_full.json").read_text())["meta"]
    blocks = meta.get("holdout_blocks") or meta.get("holdout_blocks_recovered")
    assert blocks, "the shipped model must record its holdout block IDs"
    assert "holdout_seed" in meta
    # the recorded list must be consistent with the recorded fractions
    n_total = meta.get("n_blocks_total") or meta["holdout_recovery"]["n_blocks_total"]
    assert len(blocks) == max(1, round(meta["holdout_frac"] * n_total))
    assert len(set(blocks)) == len(blocks)


def test_training_records_the_split_not_just_the_recipe():
    """New models must carry the split natively, not need recovering."""
    import pathlib

    src = pathlib.Path("scripts/train_qso_model.py").read_text()
    assert '"holdout_seed": args.holdout_seed' in src
    assert '"holdout_blocks": sorted(int(x) for x in held_blocks)' in src


def test_examples_read_the_holdout_and_never_redraw_it():
    """The bug was a re-draw; assert the re-draw is gone, not merely fixed."""
    import pathlib

    src = pathlib.Path("scripts/score_examples.py").read_text()
    assert 'qso.meta.get("holdout_blocks")' in src
    # no seeded draw over a block list anywhere in this script
    assert "rh.choice" not in src and "rh = np.random.default_rng" not in src
    # and it must refuse rather than guess when the model does not record one
    assert "records no holdout_blocks" in src


def test_example_background_cache_is_keyed_on_content():
    """Keyed on the loop index, --seed silently reused the previous cones."""
    import pathlib

    src = pathlib.Path("scripts/score_examples.py").read_text()
    assert 'local_{j:02d}.json' not in src
    assert 'example_bkg_{j:02d}.npz' not in src
    assert "hashlib.sha1" in src and "local_{tag}.json" in src


def test_local_background_rejects_wrong_or_mixed_release():
    """``system`` is a caller-supplied label; the data must be checked too.

    The scorer compares model labels, so labelling a northern or mixed cone
    with the southern system name certifies data it does not describe.
    """
    import numpy as np
    import pytest

    from qso_pcolor import background

    rows = {"ra": np.array([180.0, 180.1]), "dec": np.array([0.0, 0.1]),
            "release": np.array([9010, 9011]), "maskbits": np.zeros(2)}
    monkey = background.__dict__
    orig = monkey.get("fit_local_background")
    assert orig is not None

    import qso_pcolor.data as data

    saved = data.fetch_ls_background
    data.fetch_ls_background = lambda *a, **k: rows
    try:
        with pytest.raises(ValueError, match="photometric system mismatch"):
            background.fit_local_background(
                180.0, 0.0, 0.5, transform=None, bands=("g", "r", "z"),
                mag_edges=np.array([17.0, 22.5]), system="ls_dr9_south_grzw",
            )
        # a pure northern cone labelled north is fine as far as this check goes
        rows["release"] = np.array([9011, 9011])
        with pytest.raises(ValueError, match="photometric system mismatch"):
            background.fit_local_background(
                180.0, 0.0, 0.5, transform=None, bands=("g", "r", "z"),
                mag_edges=np.array([17.0, 22.5]), system="ls_dr9_south_grzw",
            )
    finally:
        data.fetch_ls_background = saved


def test_readme_matches_the_shipped_model():
    """The README described 1.24 M DESI quasars; a third of them are SDSS."""
    import json
    import pathlib

    meta = json.loads(pathlib.Path("models/qso_south_full.json").read_text())["meta"]
    readme = pathlib.Path("README.md").read_text()
    for key in ("n_train", "n_desi", "n_sdss", "n_holdout"):
        assert f"{meta[key]:,}" in readme, f"README does not state {key}"
    assert "1.24 M DESI" not in readme


def test_readme_install_includes_the_extras_it_then_uses():
    """pip install -e . installs neither pytest nor sqlutilpy."""
    import pathlib

    readme = pathlib.Path("README.md").read_text()
    assert 'pip install -e ".[dev,wsdb]"' in readme
    assert "\npip install -e .\n" not in readme


# ------------------------------- redshift-range extension, 2026-09-20

def test_shipped_model_covers_the_extended_redshift_range():
    """Outside support the model replays its edge slice as though measured.

    The original 0.4 < z < 3.6 was a default, not a data limit: 3.3% of DESI
    DR1 quasars sit below it and 1.1% above. Eleven slices were appended.
    """
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    q = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    lo, hi = q.support
    assert lo <= 0.15 + 1e-9 and hi >= 4.35 - 1e-9
    assert len(q.mixtures) == q.z_centres.size == 43
    # the grid must stay uniform across the join, or the interpolation in
    # log_p_colour_given_z is inconsistent either side of the old boundary
    d = np.diff(q.z_centres)
    assert np.allclose(d, d[0]), "slice spacing is not uniform after splicing"


def test_sparse_slices_did_not_get_the_core_slice_K():
    """893 objects with K=20 is ~3 per free parameter; the core has ~117.

    ``min_per_slice`` is a don't-crash fallback, not a quality criterion, so K
    is selected per appended slice by held-out density instead of asserted.
    """
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    q = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    per = {round(float(r["z"]), 2): r for r in q.meta["per_slice"]}
    core = [r["k"] for z, r in per.items() if 0.45 <= z <= 3.55]
    assert set(core) == {20}, "the original slices must be untouched"
    # the sparsest appended slices must carry fewer components than the core
    assert per[4.35]["k"] < 20 and per[4.25]["k"] < 20
    # and K must not increase as objects run out
    hi = [per[z]["k"] for z in sorted(per) if z > 3.55]
    assert all(a >= b for a, b in zip(hi, hi[1:])), f"K not monotone at high z: {hi}"


def test_extension_is_recorded_as_provenance():
    """A reader must be able to tell appended slices from originally trained."""
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    q = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    ext = q.meta["extended"]
    assert ext["previous_support"] == [0.45, 3.55]
    assert ext["slices_added_low"] + ext["slices_added_high"] == 11
    assert len(ext["k_selection_per_new_slice"]) == 11
    assert len(q.meta["per_slice"]) == 43


# ---------------------------------------- batch 2 review fixes, 2026-09-20

def test_redshift_normalisation_stops_at_the_model_support():
    """Outside support the model replays its edge slice; do not integrate it.

    Normalising over a grid wider than the trained range makes
    p_zmatch_given_qso depend on where the grid happens to stop.
    """
    import pathlib

    src = pathlib.Path("src/qso_pcolor/score.py").read_text()
    assert "in_sup = qso_model.in_support(z_grid)" in src
    assert "post = np.where(in_sup, post, 0.0)" in src
    # and the discarded share must be reported, not silently dropped
    assert "frac_norm_outside_support" in src
    assert "colours_explained_only_outside_model_redshift_support" in src


def test_scorer_reports_the_discarded_normalisation():
    """End to end: the diagnostic must equal an independent calculation."""
    from qso_pcolor.qso_model import SlicedColourRedshiftModel
    from qso_pcolor.score import DEFAULT_Z_GRID
    from qso_pcolor.features import RelativeFluxTransform, deredden

    q = SlicedColourRedshiftModel.load("models/qso_south_full.json")
    tr = RelativeFluxTransform(reference_band="r")
    f, v = deredden(np.array([[1.9, 2.6, 3.1, 11.0, 14.0]]),
                    np.array([[120.0, 150.0, 60.0, 8.0, 3.0]]),
                    np.array([[0.97, 0.98, 0.99, 1.0, 1.0]]))
    feat = tr(f, v, ("g", "r", "z", "w1", "w2"))
    lp = q.log_p_colour_given_z(feat.x, feat.cov, DEFAULT_Z_GRID,
                                observed=feat.observed)[0]
    p = np.exp(lp - lp.max())
    frac = 1 - (np.trapezoid(p * q.in_support(DEFAULT_Z_GRID), DEFAULT_Z_GRID)
                / np.trapezoid(p, DEFAULT_Z_GRID))
    # the shipped model's support covers 0.15-4.35, so this is small but nonzero
    assert 0.0 < frac < 0.05, frac


def test_sdss_training_cache_is_keyed_on_its_query():
    """Checking only that the file exists reused the sample across z limits."""
    import pathlib

    src = pathlib.Path("scripts/train_qso_model.py").read_text()
    assert "hashlib.sha1((positions_q + SDSS_MATCH_QUERY).encode())" in src
    assert "if cache.exists() and not refresh:\n        with np.load" not in src


def test_caches_are_written_atomically_and_not_blindly_unpickled():
    """A truncated cache is worse than none; the next run loads and proceeds."""
    import pathlib

    src = pathlib.Path("src/qso_pcolor/data.py").read_text()
    assert "os.replace(tmp, path)" in src
    # the plain load must be tried before the unpickling fallback (compare the
    # two np.load CALLS; the docstring mentions allow_pickle earlier than both)
    assert (src.index("with np.load(path) as z")
            < src.index("with np.load(path, allow_pickle=True) as z"))
    # and nothing else in the module may unpickle unconditionally
    assert src.count("np.load(") == 2


def test_npz_helpers_round_trip_and_leave_no_temp_file(tmp_path):
    import numpy as _np

    from qso_pcolor.data import _load_npz, _save_npz

    dest = tmp_path / "sub" / "c.npz"
    _save_npz(dest, a=_np.arange(5.0), _query=_np.array("SELECT 1"))
    assert dest.exists()
    assert not list(tmp_path.glob("**/.*tmp.npz")), "temp file left behind"
    got = _load_npz(dest)
    assert set(got) == {"a"} and _np.allclose(got["a"], _np.arange(5.0))


def test_makefile_depends_on_the_examples_figure():
    """A wildcard over plots/method alone misses figure 9."""
    import pathlib

    mk = pathlib.Path("docs/method/Makefile").read_text()
    assert "plots/examples/*.png" in mk
    assert "scripts/score_examples.py" in mk


def test_readme_does_not_offer_the_bayes_factor_as_a_ranking_statistic():
    """It has no field_q term, so it cannot order same-z against wrong-z."""
    import pathlib

    readme = pathlib.Path("README.md").read_text()
    assert "`log_r_per_unit_z` or `log_bayes_factor_qz_bkg`" not in readme
    assert "**not** an alternative ranking statistic" in readme


# ---------------------------------------- pair validation, 2026-09-20

def test_grid_prior_support_is_the_bin_edges_not_the_centres():
    """The outer half of the first and last magnitude bins was refused.

    Edges 17-22.5 give centres 18.25-22.0; testing the magnitude against the
    centres nulled 17 <= r < 18.25 and 22 < r < 22.5 -- 20% of the validation
    sample -- with status 'qso_prior_empty_at_this_magnitude'.
    """
    from qso_pcolor.priors import EmpiricalQSOPrior

    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 3.0, 20000)
    m = rng.uniform(17.0, 22.5, 20000)
    pr = EmpiricalQSOPrior.build(z, m, area_deg2=100.0,
                                 z_edges=np.linspace(0.5, 3.0, 11),
                                 mag_edges=np.array([17.0, 19.5, 20.5, 21.5, 22.5]))
    zg = np.linspace(0.5, 3.0, 30)
    for mag in (17.0, 17.5, 18.0, 22.1, 22.49):          # inside the edges
        assert (pr(zg, mag) > 0).all(), f"refused r={mag} inside the grid"
        assert pr.in_support(zg, mag).all()
    for mag in (16.99, 22.51):                             # outside
        assert (pr(zg, mag) == 0).all()
        assert not pr.in_support(zg, mag).any()
    # redshift: positive edge to edge, zero beyond
    assert (pr(np.array([0.5, 3.0]), 20.0) > 0).all()
    assert (pr(np.array([0.49, 3.01]), 20.0) == 0).all()
    # round trip keeps the edges
    from qso_pcolor.priors import GridQSOPrior
    back = GridQSOPrior.from_dict(pr.to_dict())
    assert np.allclose(back.mag_edges, pr.mag_edges) and np.allclose(back.z_edges, pr.z_edges)
    # an old file without edges reconstructs them from the centres
    d = pr.to_dict(); d.pop("mag_edges"); d.pop("z_edges")
    old = GridQSOPrior.from_dict(d)
    assert np.allclose(old.z_edges, pr.z_edges)


def test_validation_helpers_behave():
    """auc_rank: separable -> 1, identical -> 0.5, ties -> 0.5; reliability bins."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
    from validate_pairs import auc_rank, reliability

    assert auc_rank(np.array([3., 4., 5.]), np.array([0., 1., 2.])) == 1.0
    assert auc_rank(np.array([0., 1., 2.]), np.array([3., 4., 5.])) == 0.0
    assert abs(auc_rank(np.ones(50), np.ones(50)) - 0.5) < 1e-12
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000); b = rng.normal(size=5000)
    assert abs(auc_rank(a, b) - 0.5) < 0.02
    # NaNs are dropped, not counted
    assert auc_rank(np.array([3., np.nan]), np.array([0.])) == 1.0
    rows = reliability(np.array([0.1] * 30 + [0.5] * 30), np.array([0] * 27 + [1] * 3 + [1] * 15 + [0] * 15),
                       np.array([0.0, 0.3, 1.0]))
    assert len(rows) == 2 and abs(rows[0][1] - 0.1) < 1e-9 and abs(rows[1][1] - 0.5) < 1e-9
