"""Point-source test outside Legacy: Riello et al. (2021) C* and its scatter, PS1-PSC, label logic."""
import numpy as np

from qso_pcolor.point_source import cstar, point_source_morphology, sigma_cstar


def test_cstar_polynomial_matches_riello_table2():
    # f at x = 0.2 (first range), 1.0 (second), 5.0 (third), from Riello et al. 2021, Table 2
    f = lambda x, a: sum(ai*x**i for i, ai in enumerate(a))
    for x, a in ((0.2, (1.154360, 0.033772, 0.032277)), (1.0, (1.162004, 0.011464, 0.049255, -0.005879)),
                 (5.0, (1.057572, 0.140537))):
        assert np.isclose(cstar(f(x, a), x), 0.)


def test_sigma_cstar_matches_riello_eq18():
    assert np.isclose(sigma_cstar(20.), 0.0059898 + 8.817481e-12*20.**7.618399)
    assert sigma_cstar(20.) > sigma_cstar(15.) > 0


def test_labels_psc_first_then_gaia_then_unknown():
    n = np.nan
    lab, used = point_source_morphology(gaia_g=np.array([18., 18., n, n, n]),
                                        gaia_bp_rp=np.array([1., 1., n, n, n]),
                                        gaia_excess_factor=np.array([1.162004 + .011464 + .049255 - .005879, 1.6, n, n, n]),
                                        ps_score=np.array([n, n, .95, .2, n]))
    assert list(lab) == ['PSF', 'EXT', 'PSF', 'EXT', 'unknown']
    assert list(used) == ['gaia', 'gaia', 'ps1_psc', 'ps1_psc', '']
    lab2, used2 = point_source_morphology(gaia_g=np.array([18.]), gaia_bp_rp=np.array([1.]),
                                          gaia_excess_factor=np.array([1.6]), ps_score=np.array([.99]))
    assert lab2[0] == 'PSF' and used2[0] == 'ps1_psc'                     # PS1-PSC takes precedence
