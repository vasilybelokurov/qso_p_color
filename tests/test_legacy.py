"""The Legacy DR9 selection and the usable-area measurement."""
import numpy as np
import pytest

from qso_pcolor.legacy import (LegacySelection, assign_bricks, cone_usable_fraction,
                               hemisphere_labels, is_north, legacy_photometry,
                               morphology_status, random_in_cone)


def rows(**over):
    base = dict(ra=np.array([150.0]), dec=np.array([2.0]), release=np.array([9010]),
                type=np.array(["PSF"]), maskbits=np.array([0]),
                flux_r=np.array([3.0]), flux_ivar_r=np.array([10.0]), nobs_r=np.array([3]))
    base.update({k: np.atleast_1d(v) for k, v in over.items()})
    return base


def test_hemisphere_rule():
    # NGC above the boundary is north; the same dec in the SGC is south
    assert is_north([180.0], [40.0]).tolist() == [True]
    assert is_north([180.0], [30.0]).tolist() == [False]
    assert is_north([20.0], [34.0]).tolist() == [False]          # b < 0
    assert hemisphere_labels("north")[1] == "decals_dr9_north:r"


def test_morphology_status():
    s = morphology_status(np.array(["PSF", "REX", "EXP", "DEV", "SER", "DUP", "", "PSF "]))
    assert s.tolist() == ["point", "extended", "extended", "extended", "extended",
                          "unknown", "unknown", "point"]


@pytest.mark.parametrize("over,reason", [
    ({}, ""),
    (dict(release=9011), "wrong_hemisphere_release"),
    (dict(flux_ivar_r=0.0), "reference_unusable"),
    (dict(nobs_r=0), "reference_unusable"),
    (dict(maskbits=2), "masked"),
    (dict(type="REX"), "morphology"),
    (dict(type="DUP"), "morphology"),
    (dict(release=-1), "no_match"),
])
def test_point_selection_reasons(over, reason):
    d = LegacySelection("point").decide(rows(**over))
    assert d["reason"][0] == reason
    assert d["accepted"][0] == (reason == "")


def test_all_selection_accepts_extended_but_not_unknown():
    sel = LegacySelection("all")
    assert sel.decide(rows(type="EXP"))["accepted"][0]
    assert not sel.decide(rows(type="DUP"))["accepted"][0]


def test_north_release_required_in_the_north():
    d = LegacySelection().decide(rows(ra=180.0, dec=40.0, release=9010))
    assert d["reason"][0] == "wrong_hemisphere_release"
    assert LegacySelection().decide(rows(ra=180.0, dec=40.0, release=9011))["accepted"][0]


def test_identity_depends_on_the_declaration():
    assert LegacySelection("point").identity != LegacySelection("all").identity
    assert LegacySelection("point").identity == LegacySelection("point").identity


def test_photometry_keeps_negative_flux_and_masks_unusable_bands():
    r = dict(ra=np.zeros(2), maskbits=np.array([0, 0]))
    for b in ("g", "r", "z", "w1", "w2"):
        r[f"flux_{b}"] = np.array([-0.5, 2.0])
        r[f"flux_ivar_{b}"] = np.array([4.0, 0.0 if b == "z" else 1.0])
        r[f"nobs_{b}"] = np.array([1, 1])
    p = legacy_photometry(r, "south")
    assert p.bands[0] == "decals_dr9_south:g"
    assert p.flux[0, 0] == -0.5 and p.variance[0, 0] == 0.25
    assert not p.observed[1, 2] and p.observed[1, 1]


def test_random_in_cone_is_inside_and_uniform():
    rng = np.random.default_rng(1)
    ra, dec = random_in_cone(10.0, 60.0, 0.3, 20000, rng)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sep = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(10 * u.deg, 60 * u.deg)).deg
    assert sep.max() <= 0.3 + 1e-9
    # uniform in area: P(sep < r/sqrt2) = 1/2
    assert abs(np.mean(sep < 0.3 / np.sqrt(2)) - 0.5) < 0.01


def _synthetic_bricks():
    # four bricks tiling [9.5, 10.5] x [-0.5, 0.5]
    bricks = dict(brickname=np.array(["a", "b", "c", "d"]),
                  ra1=np.array([9.5, 10.0, 9.5, 10.0]), ra2=np.array([10.0, 10.5, 10.0, 10.5]),
                  dec1=np.array([-0.5, -0.5, 0.0, 0.0]), dec2=np.array([0.0, 0.0, 0.5, 0.5]))
    return bricks


def test_assign_bricks():
    b = _synthetic_bricks()
    assert assign_bricks(b, np.array([9.7, 10.2, 9.7, 10.2]),
                         np.array([-0.1, -0.1, 0.1, 0.1])).tolist() == [0, 1, 2, 3]


def test_cone_usable_fraction_with_synthetic_images():
    from astropy.io.fits import Header
    b = _synthetic_bricks()
    centres = {"a": (9.75, -0.25), "b": (10.25, -0.25), "c": (9.75, 0.25), "d": (10.25, 0.25)}

    def fetch(name, hemisphere, kind):
        if name == "d":
            return None                                  # not observed
        h = Header()
        h.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=centres[name][0],
                 CRVAL2=centres[name][1], CRPIX1=50.5, CRPIX2=50.5,
                 CD1_1=-0.006, CD1_2=0.0, CD2_1=0.0, CD2_2=0.006)
        img = np.zeros((100, 100), int)
        if kind == "maskbits" and name == "b":
            img[:] = 2                                    # fully masked
        if kind == "nexp-r":
            img[:] = 1
        return img, h

    res = cone_usable_fraction(10.0, 0.0, 0.3, "south", b, fetch, n_points=40000, seed=3)
    # quadrants a and c usable, b masked, d unobserved
    assert res["fraction"] == pytest.approx(0.5, abs=0.01)
    assert res["bricks"]["d"]["observed"] is False
