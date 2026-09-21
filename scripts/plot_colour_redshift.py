#!/usr/bin/env python
"""Plot intrinsic colour-redshift densities from the saved seven-survey model.

Colours are linear differences of the model's luptitudes. Projecting each
Gaussian exactly integrates out brightness and every other band, retaining
the covariance of the two plotted bands. Nothing is fitted or queried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from qso_pcolor.data import _save_npz
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.plotting import save_figure, use_paper_style
from qso_pcolor.qso_model import SlicedColourRedshiftModel


def project_colour(mixture: GaussianMixture, bands: tuple[str, str]) -> GaussianMixture:
    """Intrinsic density of u_a-u_b, normalised over colour in magnitudes.

    The projected variance in mag squared is V_aa+V_bb-2 V_ab. All other
    luptitudes and the common brightness coordinate are marginalised exactly.
    """
    if len(bands) != 2 or bands[0] == bands[1]:
        raise ValueError("a colour requires two distinct band labels")
    a = np.zeros(mixture.n_dim)
    a[mixture.labels.index(bands[0])] = 1.
    a[mixture.labels.index(bands[1])] = -1.
    mean = mixture.means @ a
    variance = np.einsum("i,kij,j->k", a, mixture.covs, a)
    if not np.all(variance > 0):
        raise ValueError("non-positive colour variance in saved model")
    return GaussianMixture(mixture.weights, mean[:, None], variance[:, None, None],
                           (f"{bands[0]}-{bands[1]}",))


def colour_model(model: SlicedColourRedshiftModel, bands: tuple[str, str]) -> SlicedColourRedshiftModel:
    """Project every slice, preserving the fitted model's density interpolation."""
    mixtures = [project_colour(m, bands) for m in model.mixtures]
    return SlicedColourRedshiftModel(model.z_centres, mixtures, model.n_train,
                                    model.system, mixtures[0].labels)


def mixture_at_z(model: SlicedColourRedshiftModel, z: float) -> GaussianMixture:
    """The normalised interpolated colour mixture at one supported redshift."""
    if not model.in_support(z):
        raise ValueError("requested redshift lies outside model support")
    centres = model.z_centres
    i = int(np.clip(np.searchsorted(centres, z)-1, 0, len(centres)-2))
    t = float(np.clip((z-centres[i])/(centres[i+1]-centres[i]), 0, 1))
    if t == 0:
        return model.mixtures[i]
    if t == 1:
        return model.mixtures[i+1]
    return model.mixtures[i].mixture_with(model.mixtures[i+1], 1-t)


def colour_cdf(mixture: GaussianMixture, colour: float) -> float:
    """CDF of a univariate mixture; colour is in magnitudes."""
    sigma = np.sqrt(mixture.covs[:, 0, 0])
    return float(mixture.weights @ ndtr((colour-mixture.means[:, 0])/sigma))


def colour_quantile(mixture: GaussianMixture, probability: float) -> float:
    """Mixture quantile in magnitudes, including multimodal colour densities."""
    if not 0 < probability < 1:
        raise ValueError("quantile probability must lie strictly between zero and one")
    mean, sigma = mixture.means[:, 0], np.sqrt(mixture.covs[:, 0, 0])
    return brentq(lambda c: colour_cdf(mixture, c)-probability,
                  float(np.min(mean-12*sigma)), float(np.max(mean+12*sigma)))


def evaluate(model: SlicedColourRedshiftModel, specification: dict, cfg: dict) -> dict:
    """Return a density grid and exact CDF-based display/quantile diagnostics."""
    projected = colour_model(model, tuple(specification["bands"]))
    redshift = np.unique(np.r_[np.linspace(*model.support, cfg["redshift_grid_points"]), model.z_centres])
    tail = cfg["axis_tail_probability"]
    # Interpolated CDFs are convex combinations. The extrema of these slice
    # quantiles therefore bound the same tail probability between slices too.
    lo = min(colour_quantile(m, tail) for m in projected.mixtures)
    hi = max(colour_quantile(m, 1-tail) for m in projected.mixtures)
    pad = .04*(hi-lo)
    limits = [float(np.floor(10*(lo-pad))/10), float(np.ceil(10*(hi+pad))/10)]
    colour = np.linspace(*limits, cfg["colour_grid_points"])
    density = np.exp(projected.log_p_colour_given_z(colour[:, None], None, redshift))
    quantiles, mass = [], []
    for z in redshift:
        mix = mixture_at_z(projected, float(z))
        quantiles.append([colour_quantile(mix, q) for q in cfg["quantiles"]])
        mass.append(colour_cdf(mix, limits[1])-colour_cdf(mix, limits[0]))
    mass = np.asarray(mass)
    if np.min(mass) < 1-2*tail-1e-10:
        raise AssertionError("colour axis excludes more model probability than declared")
    numerical_mass = np.trapezoid(density, colour, axis=0)
    return dict(title=specification["title"], bands=specification["bands"],
                redshift=redshift, colour=colour, density=density, quantiles=np.asarray(quantiles),
                limits=limits, minimum_displayed_probability=float(mass.min()),
                maximum_integration_error=float(np.max(np.abs(numerical_mass-mass))))


def draw(specification: dict, results: list[dict], density_limits: tuple[float, float]) -> Path:
    """Draw colour against redshift on a shared, absolute log-density scale."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    rows = (len(results)+1)//2
    height = 3.05*rows+1.9
    fig, axes = plt.subplots(rows, 2, figsize=(11.5, height), sharex=True, squeeze=False)
    # Reserve physical space for headings and footer in both figure sizes.
    fig.subplots_adjust(left=.085, right=.875, top=1-1.35/height, bottom=.85/height,
                        hspace=.35, wspace=.25)
    norm = LogNorm(*density_limits)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("white")
    for result, ax in zip(results, axes.flat):
        im = ax.pcolormesh(result["redshift"], result["colour"], result["density"],
                           shading="auto", cmap=cmap, norm=norm, rasterized=True)
        # A thin white outline keeps the median visible through dark peaks.
        for j, style in enumerate(("--", "-", "--")):
            line, = ax.plot(result["redshift"], result["quantiles"][:, j], color="black",
                            lw=1.0 if j == 1 else .65, ls=style, alpha=1 if j == 1 else .7)
            if j == 1:
                line.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
        ax.set_title(result["title"], fontsize=12, loc="left", pad=7)
        ax.set_ylabel("Luptitude colour (mag)", fontsize=10)
        ax.set_xlim(result["redshift"][[0, -1]])
        ax.set_ylim(result["limits"])
        ax.grid(False)
        ax.tick_params(labelsize=9)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Spectroscopic redshift", fontsize=11)
    bar = fig.colorbar(im, cax=fig.add_axes([.91, .18, .018, .61]), extend="min")
    bar.set_label(r"Intrinsic density $p(c\mid Q,z)$ [mag$^{-1}$]", fontsize=10, labelpad=10)
    fig.suptitle(specification["title"], y=1-.12/height, fontsize=17)
    fig.text(.48, 1-.54/height, "Saved seven-survey model | brightness and unused bands integrated out", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], color="black", lw=1., label="Median"),
                        Line2D([], [], color="black", lw=.65, ls="--", label="16th and 84th percentiles")],
               loc="upper center", bbox_to_anchor=(.48, 1-.72/height), ncol=2, fontsize=9)
    fig.text(.48, .12/height, "Unit-integral colour density at each redshift; shared density scale across both figures. No measurement noise added.",
             ha="center", va="bottom", fontsize=9)
    return save_figure(fig, specification["name"], dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/colour_redshift.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["quantiles"] != [.16, .5, .84]:
        raise ValueError("the current figure legend requires 16/50/84 percentiles")
    model = MultiSurveyModel.load(cfg["model"])
    results = [[evaluate(model.qso, c, cfg) for c in figure["colours"]] for figure in cfg["figures"]]
    maximum = max(float(r["density"].max()) for group in results for r in group)
    vmax = 10.**np.ceil(np.log10(maximum))
    limits = (vmax/10**cfg["density_decades"], vmax)
    import matplotlib
    matplotlib.use("Agg")
    use_paper_style()
    report = dict(model=cfg["model"], model_run=model.meta["run_id"],
                  model_sha256=hashlib.sha256(Path(cfg["model"]).read_bytes()).hexdigest(),
                  config=cfg, redshift_support=list(model.qso.support), density_limits=list(limits),
                  normalisation="integral over all colour = 1 at every z; no finite-axis or peak renormalisation",
                  conditioning="brightness and other bands marginalised; no surface-density/redshift prior",
                  photometry="observed native-system luptitudes; optical AB, ALLWISE/VHS Vega; no added measurement noise",
                  figures=[])
    cache = {"model_sha256": np.array(report["model_sha256"]), "config": np.array(json.dumps(cfg, sort_keys=True))}
    index = 0
    for specification, group in zip(cfg["figures"], results):
        path = draw(specification, group, limits)
        summary = dict(path=str(path.relative_to(Path.cwd().resolve())), colours=[])
        for result in group:
            summary["colours"].append({key: result[key] for key in
                ("title", "bands", "limits", "minimum_displayed_probability", "maximum_integration_error")})
            for key in ("redshift", "colour", "density", "quantiles"):
                cache[f"colour_{index}_{key}"] = result[key]
            index += 1
        report["figures"].append(summary)
        print(path, flush=True)
    path = Path(cfg["report"]); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    _save_npz(Path(cfg["grid_cache"]), **cache)
    print(path, flush=True)


if __name__ == "__main__":
    main()
