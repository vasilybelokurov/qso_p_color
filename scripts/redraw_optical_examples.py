#!/usr/bin/env python
"""Redraw Figure 10 in luptitude colours, retaining its original scores.

The saved sample includes the exact two-dimensional marginal mixtures and
original measurements. No database access, scoring, or fitting is performed.
The original model remains in flux-ratio coordinates; only its display changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from qso_pcolor.features import AsinhColourTransform
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.plotting import SERIES, save_figure, use_paper_style

MAG_FACTOR = 2.5/np.log(10.)


def ratio_to_colour(ratios: np.ndarray, reference_flux: float,
                    softening: np.ndarray) -> np.ndarray:
    """Map (g/r, z/r) to (u_g-u_r, u_z-u_r) in mag at fixed measured r flux.

    Softening follows (g, r, z), in the same nanomaggies as the dereddened flux.
    Negative numerators remain valid; the original reference must be positive.
    """
    b = np.asarray(softening)
    ref_term = np.arcsinh(reference_flux/(2*b[1])) + np.log(b[1])
    return -MAG_FACTOR*(np.arcsinh(np.asarray(ratios)*reference_flux/(2*b[[0, 2]]))
                        + np.log(b[[0, 2]]) - ref_term)


def colour_to_ratio(colours: np.ndarray, reference_flux: float,
                    softening: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse map and log |d(ratios)/d(colours)|, per mag squared.

    The density conversion is log p_colour = log p_ratio + log_jacobian.
    The same Jacobian applies to all hypotheses and cancels in their ratios.
    """
    b = np.asarray(softening)
    ref_term = np.arcsinh(reference_flux/(2*b[1])) + np.log(b[1])
    t = ref_term-np.asarray(colours)/MAG_FACTOR-np.log(b[[0, 2]])
    ratios = 2*b[[0, 2]]*np.sinh(t)/reference_flux
    log_cosh = np.logaddexp(t, -t)-np.log(2.)
    log_jacobian = (np.log(2*b[[0, 2]]/(MAG_FACTOR*reference_flux))+log_cosh).sum(axis=-1)
    return ratios, log_jacobian


def export_sample(qso, backgrounds, gbkg, fq, iq, zq, fst, isx, rows, gal, args) -> dict:
    """Freeze the original example's two-colour mixtures and score records."""
    objects = []
    for tag, fs, indices in (("quasar", fq, iq), ("PSF source", fst, isx)):
        for k, index in enumerate(indices):
            j = len(objects); score = rows[tag][k]; ref = float(fs.ref_flux[index])
            ratios = fs.x[index, :2]
            variance_ref = (ref/fs.ref_snr[index])**2
            variance_other = np.diag(fs.cov[index])[:2]*ref**2-ratios**2*variance_ref
            if (variance_other <= 0).any():
                raise ValueError("cannot reconstruct positive band variances from ratio covariance")
            z = score.z_primary
            i = int(np.clip(np.searchsorted(qso.z_centres, z)-1, 0, len(qso.z_centres)-2))
            t = float(np.clip((z-qso.z_centres[i])/(qso.z_centres[i+1]-qso.z_centres[i]), 0, 1))
            mixes, weights = [], []
            for loc, weight in ((i, 1-t), (i+1, t)):
                if weight > 0:
                    mix = qso.mixtures[loc].marginal(np.array([0, 1]))
                    mixes.append(mix); weights.extend(weight*mix.weights)
            qmix = GaussianMixture(np.array(weights), np.concatenate([m.means for m in mixes]),
                                    np.concatenate([m.covs for m in mixes]), ("g/r", "z/r"))
            models = {"quasar": qmix.to_dict()}
            for name, bg in (("global", gbkg), ("local", backgrounds[j][0])):
                if bg.local or bg.parent:
                    raise ValueError("this Figure 10 export expects the original pooled cone models")
                mb = int(bg.mag_bin([score.ref_mag])[0])
                models[name] = bg.global_[mb].marginal(np.array([0, 1])).to_dict()
            objects.append(dict(kind=tag, index=k+1, l=float(gal[0][j]), b=float(gal[1][j]),
                flux_grz=[float(ratios[0]*ref), ref, float(ratios[1]*ref)],
                variance_grz=[float(variance_other[0]), float(variance_ref), float(variance_other[1])],
                ratio_covariance=fs.cov[index, :2, :2].tolist(), score=asdict(score), mixtures=models))
    return dict(source_script="scripts/score_examples.py", seed=args.seed,
                source_model=str(args.model), model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                photometric_system=qso.system, dereddened=True,
                contours="intrinsic two-dimensional flux-ratio marginals, before measurement-noise convolution",
                objects=objects)


def render(sample: dict, cfg: dict) -> Path:
    """Transform frozen model densities and measurement errors for display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    softening = np.array([cfg["softening"][b] for b in cfg["bands"]])
    objects = sample["objects"]
    flux = np.array([o["flux_grz"] for o in objects])
    variance = np.array([o["variance_grz"] for o in objects])
    transformed = AsinhColourTransform(dict(zip(("g", "r", "z"), softening)))(flux, variance, ("g", "r", "z"))
    # AsinhColourTransform returns (g-r, r-z); display (g-r, z-r).
    signs = np.array([1., -1.])
    colours = transformed.x*signs
    covariance = transformed.cov*signs[:, None]*signs[None, :]
    errors = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    limits = np.array(cfg["colour_limits"], float)
    for dim in range(2):
        limits[dim, 0] = min(limits[dim, 0], np.min(colours[:, dim]-cfg["point_error_extent"]*errors[:, dim])-.1)
        limits[dim, 1] = max(limits[dim, 1], np.max(colours[:, dim]+cfg["point_error_extent"]*errors[:, dim])+.1)
    gx, gy = [np.linspace(*lim, cfg["grid_points"]) for lim in limits]
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    n = len(objects)//2
    use_paper_style()
    fig, axes = plt.subplots(2, n, figsize=(16, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.18, top=.81, hspace=.46, wspace=.12)
    for j, (obj, ax) in enumerate(zip(objects, axes.flat)):
        ratios, log_jacobian = colour_to_ratio(grid, flux[j, 1], softening)
        for name, colour, style in (("quasar", SERIES["same_z"], "-"),
                                    ("global", SERIES["field_q"], "--"),
                                    ("local", SERIES["background"], "-")):
            mixture = GaussianMixture.from_dict(obj["mixtures"][name])
            density = mixture.log_prob(ratios)+log_jacobian
            ax.contour(xx, yy, np.exp(density-density.max()).reshape(xx.shape),
                levels=cfg["contour_peak_fractions"], colors=[colour], linestyles=style, linewidths=.95)
        point, cov = colours[j], covariance[j]
        values, vectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
        ax.add_patch(Ellipse(point, 2*np.sqrt(values[-1]), 2*np.sqrt(values[0]), angle=angle,
                            fill=False, edgecolor="black", linewidth=.9, zorder=7))
        ax.errorbar(*point, xerr=errors[j, 0], yerr=errors[j, 1], fmt="o", color="black",
                    mfc="white", ms=4, lw=.8, zorder=8)
        score = obj["score"]
        ax.set_title(f"$z_0={score['z_primary']:.2f}$   $r={score['ref_mag']:.1f}$\n"
                     f"$(l,b)=({obj['l']:.1f}, {obj['b']:.1f})$", fontsize=9.5, pad=7)
        ax.text(.97, .97, f"$\\ln\\mathrm{{BF}}={score['log_bayes_factor_qz_bkg']:+.1f}$\n"
                f"$p_{{same}}={score['p_sameq']:.1e}$\n$\\Delta z_{{eff}}={score['dz_match_eff']:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2), zorder=10)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    for ax in axes[1]:
        ax.set_xlabel("DECaLS g - DECaLS r")
    axes[0, 0].set_ylabel("Quasars\nDECaLS z - DECaLS r")
    axes[1, 0].set_ylabel("PSF field sources\nDECaLS z - DECaLS r")
    handles = [plt.Line2D([], [], color=colour, ls=style, label=name) for colour, style, name in
               ((SERIES["same_z"], "-", "Quasar at target redshift"),
                (SERIES["field_q"], "--", "Global PSF background"),
                (SERIES["background"], "-", "Local PSF background"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .9), ncol=3, fontsize=9)
    fig.suptitle("DECaLS colours only: original model", y=.976, fontsize=18)
    fig.text(.5, .921, "Original five quasars, five field sources and scores; flux-ratio densities displayed in luptitude colours", ha="center", fontsize=10)
    fig.text(.5, .09, "Axes: dereddened luptitude colours (mag). Contours: 5%, 30%, 80% of peak density per colour area, including the coordinate Jacobian.", ha="center", fontsize=9)
    fig.text(.5, .045, "Circles, bars and ellipses: measurements and 1-sigma colour errors. W1/W2 marginalised. Original scoring window: +/-2000 km/s.", ha="center", fontsize=9)
    return save_figure(fig, cfg["figure"], dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/optical_examples_plot.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    print(render(json.loads(Path(cfg["sample"]).read_text()), cfg))


if __name__ == "__main__":
    main()
