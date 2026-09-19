"""Where figures go, and in what format.

Project convention: **every figure is a PNG and lives in ``plots/``.**  This
module enforces it rather than documenting it, because a convention that depends
on remembering a path is a convention that decays.

    from qso_pcolor.plotting import save_figure

    fig, ax = plt.subplots()
    ...
    path = save_figure(fig, "qso_locus_z1.4")      # -> plots/qso_locus_z1.4.png

Subdirectories are allowed and are created on demand
(``save_figure(fig, "m2/holdout_by_redshift")``), so a milestone's diagnostics
can be grouped without leaving the folder.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PLOTS_DIR", "plot_path", "save_figure"]

#: Repository-level output directory for figures.
PLOTS_DIR = Path(__file__).resolve().parents[2] / "plots"


def plot_path(name: str, *, root: Path | None = None) -> Path:
    """Resolve a figure name to a PNG path inside ``plots/``.

    Parameters
    ----------
    name : str
        Figure name, with or without an extension, optionally with
        subdirectories (``"m3/counts_by_latitude"``).  Any extension given is
        replaced with ``.png``: the project reads figures in reviews and diffs
        them between runs, and a mix of formats makes both harder.
    root : Path, optional
        Override the plots directory.  Intended for tests; production code
        should use the default so that every figure lands in one place.

    Returns
    -------
    Path
        Absolute path ending in ``.png``.  Parent directories are created.

    Raises
    ------
    ValueError
        If ``name`` is absolute or escapes the plots directory.
    """
    base = Path(root) if root is not None else PLOTS_DIR
    p = Path(name)
    if p.is_absolute():
        raise ValueError(f"figure name must be relative, got {name!r}")
    out = (base / p).with_suffix(".png")
    if base.resolve() not in out.resolve().parents:
        raise ValueError(f"figure name escapes the plots directory: {name!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(
    fig,
    name: str,
    *,
    dpi: int = 150,
    root: Path | None = None,
    close: bool = True,
    **savefig_kwargs,
) -> Path:
    """Save a matplotlib figure as a PNG in ``plots/`` and return its path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        As for :func:`plot_path`.
    dpi : int
        150 by default: legible on screen and in a paper draft without producing
        files too large to keep around.
    close : bool
        Close the figure afterwards.  Scripts that write many diagnostics
        otherwise accumulate open figures until matplotlib warns.
    **savefig_kwargs
        Passed through to ``Figure.savefig``.  ``bbox_inches='tight'`` is the
        default and can be overridden.

    Returns
    -------
    Path
        Where the figure was written, so a caller can log or print it.
    """
    out = plot_path(name, root=root)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(out, dpi=dpi, format="png", **savefig_kwargs)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return out
