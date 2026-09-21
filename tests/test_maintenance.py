"""Regression checks for the maintenance pass; no trained models are refitted."""
import sys
from pathlib import Path

import numpy as np
import pytest

from qso_pcolor.priors import GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch
from test_review_regressions import _score, _toy


def test_spike_between_grid_points_uses_same_total_and_window():
    # A triangular spike of base .01 and height .999, atop a constant .001.
    prior = GridQSOPrior(
        np.array([0, .5, .505, .51, 1]), np.array([10., 30.]),
        np.repeat(np.array([.001, .001, 1., .001, .001])[:, None], 2, axis=1),
    )
    s = _score(prior, np.linspace(0, 1, 101), .505, RedshiftMatch(dz_half_width=.005))
    total = .001 + .5 * .01 * .999
    same = .001 * .01 + .5 * .01 * .999
    assert s.status == 'ok'
    assert s.p_zmatch_given_qso == pytest.approx(same / total, rel=1e-6)
    assert np.exp(s.log_lambda_sameq) == pytest.approx(same / np.sqrt(2*np.pi))
    assert s.p_sameq == pytest.approx(same / (total + 1/30))


def test_narrower_prior_support_has_no_trapezoid_tail():
    prior = GridQSOPrior(np.array([.3, .5, .7]), np.array([10., 30.]),
                        np.ones((3, 2)), z_edges=np.array([.25, .4, .6, .75]))
    s = _score(prior, np.array([0., .5, 1.]), .5, RedshiftMatch(dz_half_width=.1))
    assert s.p_zmatch_given_qso == pytest.approx(.2 / .5)
    assert s.p_sameq == pytest.approx(.2 / (.5 + 1/30))


def test_empty_prior_keeps_evidence_and_window_width():
    prior = GridQSOPrior(np.array([0., .5, 1.]), np.array([10., 30.]), np.zeros((3, 2)))
    s = _score(prior, np.linspace(0, 1, 101), .5, RedshiftMatch(dz_half_width=.01))
    assert s.status == 'qso_prior_empty_at_this_magnitude'
    assert s.log_bayes_factor_qz_bkg == pytest.approx(0.)
    assert s.loglike_qso_zprimary == pytest.approx(-.5 * np.log(2*np.pi))
    assert s.dz_match_eff == pytest.approx(.02)
    assert np.isnan(s.p_sameq) and np.isnan(s.log_r_per_unit_z)


def test_public_redshift_helpers_respect_support_and_resolve_window():
    q, _, f, _ = _toy()
    grid = np.linspace(-.2, 2., 221)
    post = q.redshift_posterior(f.x, f.cov, grid)[0]
    assert np.all(post[~q.in_support(grid)] == 0)
    assert np.allclose(post[q.in_support(grid)], 1.)
    prob = q.p_zmatch_given_qso(f.x, f.cov, .505,
                               RedshiftMatch(dz_half_width=.001), grid)
    assert prob[0] == pytest.approx(.002)
    empty = q.redshift_posterior(f.x, f.cov, grid, log_z_prior=np.full(grid.size, -np.inf))
    assert np.isnan(empty).all()


def test_public_match_and_scorer_agree_on_nonuniform_prior():
    q, _, f, _ = _toy()
    grid = np.linspace(0., 1., 101)
    density = 1 + grid
    prior = GridQSOPrior(grid, np.array([10., 30.]), np.repeat(density[:, None], 2, axis=1))
    match = RedshiftMatch(dz_half_width=.001)
    s = _score(prior, grid, .505, match)
    prob = q.p_zmatch_given_qso(f.x, f.cov, .505, match, grid, log_z_prior=np.log(density))
    assert prob[0] == pytest.approx(s.p_zmatch_given_qso)
    assert prob[0] == pytest.approx(.002 * 1.505 / 1.5)


def test_feature_subset_preserves_flags_with_boolean_and_integer_indices():
    _, _, f, _ = _toy()
    f = f.subset(np.array([0, 0, 0]))
    f.flags = {'suspect': np.array([True, False, True])}
    for indices in (np.array([False, True, True]), np.array([1, 2])):
        sub = f.subset(indices)
        assert sub.n_obs == 2
        assert sub.flags['suspect'].tolist() == [False, True]


sys.path.insert(0, str(Path('scripts').resolve()))


def test_validation_relabels_for_requested_window_and_rejects_unknown_redshift():
    from validate_pairs import validation_labels
    types = np.array(['QSO', 'QSO', 'QSO', 'STAR'])
    dv = np.array([2500., -4500., np.nan, 0.])
    assert validation_labels(types, dv, 3000).tolist() == [
        'same_z', 'field_q', 'invalid_redshift', 'non_qso']
    assert validation_labels(types, dv, 5000)[1] == 'same_z'


def test_validation_quality_cuts_reject_missing_fracflux():
    from validate_pairs import clean_photometry
    assert clean_photometry(np.array([0, 1, 0, 0]), np.array([.1, .1, np.nan, .3]), .2).tolist() == [
        True, False, False, False]


@pytest.mark.parametrize('mask_cut, expected', [(True, [0., 2.]), (False, [0., 1., 2.])])
def test_extension_inherits_mask_cut(monkeypatch, tmp_path, mask_cut, expected):
    import train_qso_model as train
    from extend_qso_model_redshift import load_edge_sample
    data = {'release': np.full(3, 9010), 'ra': np.arange(3.), 'dec': np.zeros(3),
            'zspec': np.ones(3), 'maskbits': np.array([0, 1, 0])}
    for prefix in ('flux_', 'flux_ivar_', 'mw_transmission_'):
        for band in ('g', 'r', 'z', 'w1', 'w2'):
            data[prefix+band] = np.ones(3)
    monkeypatch.setattr(train, 'load_desi', lambda *a: data)
    monkeypatch.setattr(train, 'deduplicate', lambda ra, dec, z: np.ones(ra.size, bool))
    got = load_edge_sample(tmp_path, 0, 1, 9010, False, maskbits_cut=mask_cut)
    assert got['ra'].tolist() == expected
