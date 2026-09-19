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

__all__ = [
    "PLOTS_DIR",
    "SEQUENTIAL_CMAP",
    "SERIES",
    "plot_path",
    "save_figure",
    "use_paper_style",
]

#: Categorical colours, in fixed assignment order.  Validated for colour-vision
#: deficiency on the all-pairs list (worst CVD dE 9.2, worst normal-vision dE
#: 24.0), which caps a scatter-type figure at these three.  Assign them to
#: entities, never to rank, and never cycle past the third: a fourth category
#: folds into "other" or becomes a separate panel.
SERIES = {
    "same_z": "#2a78d6",      # blue   - quasar at the primary redshift
    "field_q": "#eb6834",     # orange - quasar at another redshift
    "background": "#1baf7a",  # aqua   - everything else in the catalogue
    "neutral": "#5c5c57",
    "grid": "#d9d9d4",
}

#: Sequential ramp for redshift and other magnitude encodings.  Monotonic in
#: lightness and colour-vision safe, which is what the light-to-dark rule is
#: protecting; a rainbow map is never used, because its lightness is not
#: monotonic and it invents category boundaries the data does not have.
SEQUENTIAL_CMAP = "viridis"


def use_paper_style() -> None:
    """Set matplotlib defaults for figures destined for the LaTeX write-up.

    Recessive axes and grid, thin marks, and a serif face that matches the
    document body, so a figure does not shout over the text it illustrates.
    Call once at the top of a figure script.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": SERIES["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "xtick.color": "#8a8a85",
        "ytick.color": "#8a8a85",
        "xtick.labelcolor": "#2b2b28",
        "ytick.labelcolor": "#2b2b28",
        "text.color": "#2b2b28",
        "axes.labelcolor": "#2b2b28",
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

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
