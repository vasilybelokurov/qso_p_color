#!/usr/bin/env python
"""Figure-10-style survey comparisons, using a fixed saved held-out sample.

The committed ten-object sample lets a fresh clone reproduce the figures
without database access or fitting. --reselect draws from the original local
validation caches, using coverage and magnitude only, never model scores.
Contours marginalise unplotted bands; annotations use all selected bands.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from qso_pcolor.gaussmix import GaussianMixture, condition_joint
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry, survey_of
from qso_pcolor.plotting import SERIES, save_figure, use_paper_style
from qso_pcolor.qso_model import RedshiftMatch


def sha256(path: Path | str) -> str:
    """Content identity for the small published sample and its parent caches."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_json(value):
    """Encode unavailable measurements/posteriors as JSON null, never NaN."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")


def selection_settings(cfg: dict) -> dict:
    """The pre-score selection, separate from figure appearance settings."""
    return dict(seed=cfg["seed"], objects_per_class=cfg["objects_per_class"],
                reference=cfg["selection_reference"],
                luptitude_range=cfg["selection_luptitude_range"],
                required_bands=sorted({b for c in cfg["combinations"] for b in c["plane"]}))


def select_sample(cfg: dict, model: MultiSurveyModel) -> dict:
    """Draw distinct held-out QSO blocks and field rows before computing scores."""
    selection = selection_settings(cfg)
    rng = np.random.default_rng(selection["seed"])
    n = selection["objects_per_class"]
    objects, sources = [], {}
    qso_redshifts = None
    for kind, key in (("quasar", "quasars"), ("field", "field_sources")):
        path = Path(cfg[key])
        with np.load(path, allow_pickle=False) as source:
            data = dict(source)
        phot = Photometry(data["flux"], data["variance"], tuple(data["bands"]))
        required = [phot.bands.index(b) for b in selection["required_bands"]]
        eligible = data["held"].copy() & phot.observed[:, required].all(axis=1)
        if kind == "quasar":
            eligible &= model.qso.in_support(data["zspec"])
            eligible &= np.isin(data["group"], model.qso.meta["heldout_blocks"])
        else:
            # This extra field is exclusively validation data, separate from
            # both the original fit fields and component-selection fields.
            eligible &= ~np.isin(data["field"], [f["field"] for f in model.meta["fields"]])
        candidates = np.flatnonzero(eligible)
        features = model.transform(phot.subset(candidates))
        ref = features.x[:, features.labels.index(selection["reference"])]
        lo, hi = selection["luptitude_range"]
        candidates = candidates[(ref > lo) & (ref < hi)]
        if kind == "quasar":
            groups = np.unique(data["group"][candidates])
            if len(groups) < n:
                raise ValueError("too few reserved QSO blocks with the required coverage")
            chosen_groups = rng.choice(groups, n, replace=False)
            rows = np.array([rng.choice(candidates[data["group"][candidates] == g]) for g in chosen_groups])
            # Order by redshift for easier reading; selection itself is random.
            rows = rows[np.argsort(data["zspec"][rows])]
            qso_redshifts = data["zspec"][rows]
            assigned = qso_redshifts
        else:
            if len(candidates) < n:
                raise ValueError("too few reserved field sources with the required coverage")
            rows = rng.choice(candidates, n, replace=False)
            assigned = qso_redshifts  # same target z in each column, across both rows
        sources[kind] = dict(path=str(path), sha256=sha256(path), eligible_count=len(candidates),
                             groups=np.unique(data["group" if kind == "quasar" else "field"][candidates]))
        for j, (row, z0) in enumerate(zip(rows, assigned)):
            objects.append(dict(id=f"{'Q' if kind == 'quasar' else 'F'}{j+1}", kind=kind,
                source_row=int(row), held=True,
                group=int(data["group" if kind == "quasar" else "field"][row]),
                ra=data["ra"][row], dec=data["dec"][row], l=data["l"][row], b=data["b"][row],
                z_primary=z0, zspec=data["zspec"][row] if kind == "quasar" else None,
                flux=phot.flux[row], variance=phot.variance[row]))
    return finite_json(dict(selection=selection, transform_id=model.transform_id,
                            bands=phot.bands, sources=sources, objects=objects))


def colour_projection(mix: GaussianMixture, plane: tuple[str, str, str],
                      anchor_value: float, noise: np.ndarray) -> GaussianMixture:
    """Noisy two-colour marginal given the measured anchor, per mag squared.

    ``plane`` is (anchor, x band, y band); ``noise`` follows that order, in
    mag squared. Convolution precedes conditioning. Subtracting the measured
    anchor then changes coordinates with unit Jacobian. Unplotted bands are
    marginalised, not fixed to their observed values.
    """
    idx = [mix.labels.index(b) for b in plane]
    marginal = GaussianMixture(mix.weights, mix.means[:, idx],
        mix.covs[:, idx, :][:, :, idx] + noise, plane)
    _, conditional = condition_joint(marginal, [anchor_value], [0])
    # Conditioning can underflow an irrelevant component's weight to exact
    # zero. Omit that zero term before the plotting density takes log(weights).
    positive = conditional.weights > 0
    return GaussianMixture(conditional.weights[positive], conditional.means[positive] - anchor_value,
                           conditional.covs[positive], plane[1:])


def qso_projection(model: MultiSurveyModel, z: float, plane: tuple[str, str, str],
                   anchor_value: float, noise: np.ndarray) -> GaussianMixture:
    """Interpolate conditional slice densities, as in the public scorer."""
    if not model.qso.in_support(z):
        raise ValueError("example redshift lies outside the fitted range")
    centres = model.qso.z_centres
    i = int(np.clip(np.searchsorted(centres, z) - 1, 0, len(centres) - 2))
    t = float(np.clip((z-centres[i])/(centres[i+1]-centres[i]), 0, 1))
    projected, weights = [], []
    for index, weight in ((i, 1-t), (i+1, t)):
        if weight > 0:
            p = colour_projection(model.qso.mixtures[index], plane, anchor_value, noise)
            projected.append(p)
            weights.extend(weight*p.weights)
    return GaussianMixture(np.array(weights), np.concatenate([p.means for p in projected]),
                           np.concatenate([p.covs for p in projected]), plane[1:])


def background_for_observed(model: MultiSurveyModel, observed: np.ndarray) -> GaussianMixture:
    """Route using ALL scored bands, before reducing to the plotted plane."""
    used = {b for b, ok in zip(model.transform.bands, observed) if ok}
    for marginal in sorted(model.background_marginals, key=lambda m: m.n_dim):
        if used <= set(marginal.labels):
            return marginal
    return model.background


def quantile(mix: GaussianMixture, dim: int, probability: float) -> float:
    """Exact univariate mixture quantile in colour magnitudes."""
    mean, sigma = mix.means[:, dim], np.sqrt(mix.covs[:, dim, dim])
    return brentq(lambda v: np.dot(mix.weights, ndtr((v-mean)/sigma))-probability,
                  float(np.min(mean-12*sigma)), float(np.max(mean+12*sigma)))


def label(band: str) -> str:
    system, name = band.split(":")
    names = dict(decals_dr9_south="DECaLS", sdss="SDSS", allwise="WISE",
                 ps1="PS1", nsc="NSC", skymapper="SkyMapper", vhs="VHS")
    if system in ("allwise", "vhs"):
        name = name.upper()
    return f"{names[system]} {name}"


def make_figure(cfg: dict, combination: dict, model: MultiSurveyModel,
                sample: dict, phot: Photometry):
    """Score every available selected band and draw its two-colour projection."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    objects = sample["objects"]
    selected = phot.keep_surveys(tuple(combination["surveys"]))
    features = model.transform(selected)
    scores = model.score(selected, z_primary=np.array([o["z_primary"] for o in objects]),
        l_deg=np.array([o["l"] for o in objects]), b_deg=np.array([o["b"] for o in objects]),
        candidate_id=[o["id"] for o in objects], min_bands=cfg["min_bands"],
        match=RedshiftMatch(half_width_kms=cfg["half_width_kms"]))
    plane = tuple(combination["plane"])
    idx = [features.labels.index(b) for b in plane]
    if not features.observed[:, idx].all():
        raise ValueError("saved sample lacks a requested plotting band")
    if any(s.reference_band != plane[0] for s in scores):
        raise ValueError("plot and scorer must condition on the same reference")
    records, projections, points, colour_covs = [], [], [], []
    transform = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    for i, (obj, score) in enumerate(zip(objects, scores)):
        x = features.x[i, idx]
        noise = features.cov[i][np.ix_(idx, idx)]
        bg = background_for_observed(model, features.observed[i])
        projections.append((qso_projection(model, obj["z_primary"], plane, x[0], noise),
                            colour_projection(bg, plane, x[0], noise)))
        points.append(transform @ x)
        colour_covs.append(transform @ noise @ transform.T)
        records.append(dict(object_id=obj["id"], score=asdict(score),
                            background="southern_grz_marginal" if bg is not model.background else "joint",
                            plotted_colours=points[-1], colour_covariance=colour_covs[-1]))

    points, colour_covs = np.array(points), np.array(colour_covs)
    limits = []
    for dim in range(2):
        low = [quantile(p, dim, cfg["axis_tail_probability"]) for pair in projections for p in pair]
        high = [quantile(p, dim, 1-cfg["axis_tail_probability"]) for pair in projections for p in pair]
        err = cfg["point_error_extent"]*np.sqrt(colour_covs[:, dim, dim])
        lo = min(min(low), np.min(points[:, dim]-err))
        hi = max(max(high), np.max(points[:, dim]+err))
        limits.append((lo-.04*(hi-lo), hi+.04*(hi-lo)))
    gx, gy = [np.linspace(*lim, cfg["grid_points"]) for lim in limits]
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # A low-weight broad tail can enlarge a quantile range while contributing
    # none of the contours being shown. Frame the displayed contours and every
    # measurement, keeping one common frame across the ten panels.
    contour_points = []
    for pair in projections:
        for density in pair:
            lp = density.log_prob(grid)
            contour_points.append(grid[lp >= lp.max()+np.log(min(cfg["contour_peak_fractions"]))])
    vertices = np.concatenate(contour_points)
    for dim in range(2):
        err = cfg["point_error_extent"]*np.sqrt(colour_covs[:, dim, dim])
        lo = min(np.min(vertices[:, dim]), np.min(points[:, dim]-err))
        hi = max(np.max(vertices[:, dim]), np.max(points[:, dim]+err))
        limits[dim] = (lo-.08*(hi-lo), hi+.12*(hi-lo))
    gx, gy = [np.linspace(*lim, cfg["grid_points"]) for lim in limits]
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    n = cfg["objects_per_class"]
    fig, axes = plt.subplots(2, n, figsize=(16, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.81, hspace=.46, wspace=.12)
    for i, (ax, obj, score, pair) in enumerate(zip(axes.flat, objects, scores, projections)):
        for density, colour, style in zip(pair, (SERIES["same_z"], SERIES["background"]), ("solid", "dashed")):
            lp = density.log_prob(grid).reshape(xx.shape)
            ax.contour(xx, yy, np.exp(lp-lp.max()), levels=cfg["contour_peak_fractions"],
                       colors=[colour], linewidths=.95, linestyles=style, alpha=.9)
        point, cov = points[i], colour_covs[i]
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ax.add_patch(Ellipse(point, 2*np.sqrt(vals[-1]), 2*np.sqrt(vals[0]), angle=angle,
                            fill=False, edgecolor="black", linewidth=.9, zorder=7))
        ax.errorbar(*point, xerr=np.sqrt(cov[0, 0]), yerr=np.sqrt(cov[1, 1]),
                    fmt="o", mfc="white", mec="black", color="black", ms=4, lw=.8, zorder=8)
        reference = features.x[i, idx[0]]
        ax.set_title(f"{obj['id']}   $z_0={obj['z_primary']:.2f}$   $u_{{ref}}={reference:.1f}$\n"
                     f"$(l,b)=({obj['l']:.1f}, {obj['b']:.1f})$", fontsize=9.5, pad=7)
        percent = 100*score.p_zmatch_given_qso
        pz_text = f"={percent:.2f}" if percent >= .01 else "<0.01"
        annotation = (f"$\\ln \\mathrm{{BF}}={score.log_bayes_factor_qz_bkg:+.1f}$\n"
                      f"$P_z{pz_text}\\%$   $N_{{band}}={len(score.bands_used)}$")
        ax.text(.97, .97, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2), zorder=10)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", labelsize=8)
    axes[0, 0].set_ylabel(f"Quasars\n{label(plane[2])} - {label(plane[0])}")
    axes[1, 0].set_ylabel(f"Field sources\n{label(plane[2])} - {label(plane[0])}")
    for ax in axes[1]:
        ax.set_xlabel(f"{label(plane[1])} - {label(plane[0])}")
    fig.suptitle(combination["name"], y=.976, fontsize=18)
    fig.text(.5, .921, "The same five reserved quasars and five reserved field sources in every version", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=SERIES["same_z"], label="Quasar at target redshift"),
                        Line2D([], [], color=SERIES["background"], ls="--", label="All-source background"),
                        Line2D([], [], color="black", marker="o", mfc="white", ls="none", label="Measured object; 1-sigma errors")],
               loc="upper center", bbox_to_anchor=(.5, .903), ncol=3, fontsize=9)
    fig.text(.5, .098, "Axes: native-system luptitude colours (mag). Contours: two-colour marginals at 5%, 30%, 80% of peak; conditioned on the reference.",
             ha="center", fontsize=9)
    fig.text(.5, .063, "Scores use ALL available bands of the selected surveys. $P_z$: target-redshift probability conditional on being a quasar; flat redshift prior.",
             ha="center", fontsize=9)
    fig.text(.5, .028, f"Window: +/-{cfg['half_width_kms']:g} km/s. Field objects share one overlap field and have no class labels. Population posterior unavailable.",
             ha="center", fontsize=9)
    slug = "_".join(combination["surveys"])
    path = save_figure(fig, f"{cfg['figure_dir']}/{slug}", dpi=180)
    return dict(name=combination["name"], surveys=combination["surveys"], plane=plane,
                figure=str(path.relative_to(Path.cwd().resolve())), axis_limits=limits, objects=records)


def make_pdf(path: Path, results: list[dict]) -> None:
    """Collect the PNG figures into a twelve-page, zoomable comparison atlas."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    width, height = 16*72, 8*72
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.setTitle("Figure 10 survey-combination examples")
    pdf.setAuthor("qso_pcolor")
    for i, result in enumerate(results):
        pdf.bookmarkPage(str(i))
        pdf.addOutlineEntry(result["name"], str(i), 0)
        pdf.drawImage(ImageReader(result["figure"]), 12, 30, width=width-24, height=height-42,
                      preserveAspectRatio=True, anchor="c")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(22, 13, "Reproduce: python scripts/make_multisurvey_examples.py | Objects and exact scores: docs/examples/")
        pdf.drawRightString(width-22, 13, f"{i+1} / {len(results)}")
        pdf.showPage()
    pdf.save()


def make_gallery(path: Path, sample: dict, results: list[dict], cfg: dict) -> None:
    """Publish the figures, fixed sample, and score comparison together."""
    n = sample["selection"]["objects_per_class"]
    lo, hi = sample["selection"]["luptitude_range"]
    q, b = sample["sources"]["quasar"], sample["sources"]["field"]
    lines = [f"# Figure 10 versions for {len(results)} survey combinations", "",
        f"[Download the {len(results)}-page PDF](multisurvey_examples.pdf). Each image below also opens at full resolution.", "",
        f"All versions use the same **{n} held-out spectroscopic quasars** (Q1-Q{n}) and "
        f"**{n} unclassified field sources** (F1-F{n}). These are new examples selected for common coverage, "
        "rather than the objects in Figure 10. Figure 10 now uses the same luptitude-colour axis convention, "
        "while retaining its original objects, models and scores; its photometry remains dereddened. "
        "The [original flux-ratio plot](../../plots/examples/optical_only_examples_flux_ratio.png) is archived.", "",
        f"The sample was drawn with seed {sample['selection']['seed']}, before evaluating scores, from "
        f"{q['eligible_count']} eligible quasars in {len(q['groups'])} "
        f"reserved sky blocks and {b['eligible_count']} eligible field sources. Quasars come from {n} different blocks; "
        "all field sources come from the extra reserved overlap field (field 24). "
        f"Selection requires the plotted bands to be measured and {lo:g} < {label(sample['selection']['reference'])} luptitude < {hi:g}. "
        "This small common-coverage illustration is not a representative performance test; "
        "the [larger validation](../MULTISURVEY_VALIDATION.md) supplies those results.", "",
        "Blue contours show the quasar model at the target redshift; green dashed contours show "
        "the all-source background. Both are normalised two-colour densities conditioned on the "
        "measured reference band, with all other bands marginalised and the plotted bands' measurement "
        "noise convolved before conditioning. Levels are 5%, 30%, and 80% of peak density, not enclosed "
        "probabilities. Circles mark measurements; bars and ellipses show one-sigma colour errors, "
        "including the covariance from the common reference. Axes are shared across all ten panels "
        "within each figure and include every point and its two-sigma error extent.", "",
        "**Annotations use every available band in the selected surveys**, so two displayed colours "
        "need not explain the full score. `ln BF` is the natural-log quasar-at-target-z/background "
        f"likelihood ratio. `P_z` is `p_zmatch_given_qso` for a +/-{cfg['half_width_kms']:g} km/s window and a flat redshift "
        "prior over model support (0.15-4.35). It assumes the object is a quasar and is not the "
        "probability of a physical companion. Population posteriors and `log R` remain unavailable "
        "without matching surface-density priors. The same target redshift is used in the top and "
        "bottom panel of each column.", "",
        "Optical bands retain their native AB calibration; ALLWISE and VHS retain Vega calibration. "
        "Colours are differences of the model's saved luptitudes, not ordinary magnitude colours at "
        "low signal-to-noise. Usable negative fluxes remain measurements. DECaLS-only uses the "
        "southern grz background marginal; all other versions use the joint background. These figures "
        "show catalogue-object photometric diagnostics, without a per-object local background fit or "
        "a claim that the sources meet a close-companion blend policy.", "",
        "## Fixed objects", "", "| ID | RA (deg) | Dec (deg) | Target z | Reserved block/field |",
        "|---|---:|---:|---:|---:|"]
    for obj in sample["objects"]:
        lines.append(f"| {obj['id']} | {obj['ra']:.7f} | {obj['dec']:.7f} | {obj['z_primary']:.5f} | {obj['group']} |")
    lines += ["", "F1-F5 have assigned target redshifts, not measured spectroscopic redshifts.", "",
              "## Scores on the same objects", "",
              "Natural-log Bayes factors; signs do not provide spectroscopic class labels for the field sources.", "",
              "| Survey inputs | " + " | ".join(o["id"] for o in sample["objects"]) + " |",
              "|---|" + "---:|"*len(sample["objects"])]
    for result in results:
        values = " | ".join(f"{o['score']['log_bayes_factor_qz_bkg']:+.1f}" for o in result["objects"])
        lines.append(f"| {result['name']} | {values} |")
    lines += ["", "These examples were retained as drawn, without replacing ambiguous cases. Positive "
              "evidence for an unclassified field source must not be read as a confirmed rejection failure.", ""]
    for i, result in enumerate(results):
        link = "../../"+result["figure"]
        lines += [f"## {i+1}. {result['name']}", "", f"[![{result['name']}]({link})]({link})", ""]
    lines += ["## Reproduce", "", "```bash", 'pip install -e ".[figures]"',
        "python scripts/make_multisurvey_examples.py", "```", "",
        "The committed [sample](multisurvey_sample.json) contains all ten objects' native fluxes and "
        "variances, parent-cache hashes, source row numbers, and selection settings. The script uses "
        "the saved model and needs no WSDB access or training. `--reselect` deliberately regenerates "
        "the sample from the original local caches. The [configuration](../../configs/multisurvey_examples.json) "
        "declares the survey combinations, plotting bands, redshift window, and seed. "
        "[Exact scores](multisurvey_scores.json) record all 120 score rows, band lists, status and quality "
        "flags, plotted covariance matrices, and model/sample hashes. No model files are changed.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multisurvey_examples.json"))
    parser.add_argument("--reselect", action="store_true", help="redraw from local parent caches using the configured seed")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    model = MultiSurveyModel.load(cfg["model"])
    sample_path = Path(cfg["sample"])
    if args.reselect or not sample_path.exists():
        write_json(sample_path, select_sample(cfg, model))
    sample = json.loads(sample_path.read_text())
    if sample["selection"] != selection_settings(cfg) or sample["transform_id"] != model.transform_id:
        raise ValueError("saved example selection/transform differs; use --reselect deliberately")
    phot = Photometry(np.array([o["flux"] for o in sample["objects"]], float),
                      np.array([o["variance"] for o in sample["objects"]], float), tuple(sample["bands"]))
    import matplotlib
    matplotlib.use("Agg")
    use_paper_style()
    results = []
    for combination in cfg["combinations"]:
        results.append(make_figure(cfg, combination, model, sample, phot))
        print(results[-1]["figure"], flush=True)
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"multisurvey_scores.json", dict(model=cfg["model"], model_sha256=sha256(cfg["model"]),
        model_run=model.meta["run_id"], sample_sha256=sha256(sample_path), config=cfg, combinations=results))
    make_pdf(root/"multisurvey_examples.pdf", results)
    make_gallery(root/"README.md", sample, results, cfg)
    print(root/"multisurvey_examples.pdf", flush=True)


if __name__ == "__main__":
    main()
