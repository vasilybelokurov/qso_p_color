"""Independent checks of the common-grid old/new validation calculation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from qso_pcolor.features import FeatureSet
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.qso_model import SlicedColourRedshiftModel


def test_comparison_redshift_summaries_against_scalar_quadrature(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('comparison_test', scripts/'compare_old_new_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    centres = np.array([.2, 1.2, 2.2])
    labels = ('colour',)
    mixtures = [GaussianMixture(np.ones(1), np.array([[z]]), np.array([[[.3]]]), labels) for z in centres]
    model = SlicedColourRedshiftModel(centres, mixtures, np.ones(3), 'test', labels)
    x = np.array([[.7], [1.6]])
    cov = np.array([[[.1]], [[.2]]])
    features = FeatureSet(x, cov, np.ones_like(x, bool), np.ones(2), np.ones(2), np.ones(2), labels)
    grid = np.unique(np.r_[np.linspace(.2, 2.2, 201), centres])
    redshift, alternative = np.array([.6, 1.8]), np.array([1.5, .4])
    result = module.redshift_diagnostics(model, features, grid, redshift, alternative)
    for i in range(2):
        endpoint = norm.pdf(x[i,0], loc=centres, scale=np.sqrt(.3+cov[i,0,0]))
        density = lambda z: np.interp(z, centres, endpoint)
        total = quad(density, centres[0], centres[-1], points=centres)[0]
        assert np.exp(result['log_pz'][i]) == pytest.approx(density(redshift[i])/total)
        assert np.exp(result['alternative_log_pz'][i]) == pytest.approx(density(alternative[i])/total)
        integral = quad(density, centres[0], redshift[i], points=centres[centres<redshift[i]])[0]
        assert result['true_cdf'][i] == pytest.approx(integral/total)
        # Equal adjacent endpoint densities make the whole intervening segment
        # a valid mode; do not require one particular grid point in that tie.
        assert density(result['z_mode'][i]) == pytest.approx(endpoint.max())
