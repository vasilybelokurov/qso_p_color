"""fetch_cones cache check: a filename hit is trusted only when centre and radius match."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/euclid'))
from fetch_cones import CacheMismatch, cached_matches  # noqa: E402


def _write(path, ra, dec, r):
    np.savez_compressed(path, centre=np.array([ra, dec]), radius_deg=r)


def test_absent_file_is_not_cached(tmp_path):
    assert cached_matches(tmp_path/'test_001.npz', dict(ra=10., dec=-20., radius_deg=.3)) is False


def test_matching_cone_is_reused(tmp_path):
    p = tmp_path/'test_001.npz'; _write(p, 10., -20., .3)
    assert cached_matches(p, dict(ra=10., dec=-20., radius_deg=.3)) is True


@pytest.mark.parametrize('cone', [dict(ra=10.001, dec=-20., radius_deg=.3), dict(ra=10., dec=-20.5, radius_deg=.3),
                                  dict(ra=10., dec=-20., radius_deg=.25)])
def test_different_cone_under_same_name_raises(tmp_path, cone):
    p = tmp_path/'test_001.npz'; _write(p, 10., -20., .3)
    with pytest.raises(CacheMismatch):
        cached_matches(p, cone)
