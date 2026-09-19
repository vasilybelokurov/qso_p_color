"""The figure-output convention: PNG, in ``plots/``, enforced not documented."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from qso_pcolor.plotting import PLOTS_DIR, plot_path, save_figure  # noqa: E402


def test_extension_is_forced_to_png(tmp_path):
    assert plot_path("locus", root=tmp_path).name == "locus.png"
    assert plot_path("locus.pdf", root=tmp_path).name == "locus.png"
    assert plot_path("locus.svg", root=tmp_path).name == "locus.png"


def test_subdirectories_are_created(tmp_path):
    p = plot_path("m3/counts_by_latitude", root=tmp_path)
    assert p.parent.is_dir()
    assert p.parent.name == "m3"


def test_absolute_and_escaping_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        plot_path("/tmp/sneaky", root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        plot_path("../outside", root=tmp_path)


def test_save_figure_writes_a_png_and_returns_the_path(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_figure(fig, "smoke", root=tmp_path)

    assert out.suffix == ".png"
    assert out.exists() and out.stat().st_size > 0
    # Verify it really is a PNG, not a renamed something else.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The figure is closed, so a long script does not leak figures.
    assert not plt.fignum_exists(fig.number)


def test_default_directory_is_the_repository_plots_folder():
    assert PLOTS_DIR.name == "plots"
    assert (PLOTS_DIR.parent / "pyproject.toml").exists()
