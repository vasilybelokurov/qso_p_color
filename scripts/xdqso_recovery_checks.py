"""Independent observable-field checks for the relative-flux PSF baseline.

Tails are compared before the science support exclusion. The simulated field
includes unrecognised quasars, non-quasars, the Gaussian unmodelled share, and
each observed object's full measurement covariance and missing-band pattern.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def continue_field_bin(job):
    """One fixed-point continuation on the original fit rows, with refreshed weights."""
    from qso_pcolor.xd import fit_xd
    from qso_pcolor.gaussmix import GaussianMixture
    h, mb, mix, x, cov, obs, weight, xv, cv, ov, wv, floor, iterations, cache = job
    digest = hashlib.sha256(json.dumps(dict(mixture=mix, floor=floor, iterations=iterations), sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    import qso_pcolor.xd as xd_module
    digest.update(Path(xd_module.__file__).read_bytes())
    for array in (x, cov, obs, weight, xv, cv, ov, wv):
        digest.update(np.ascontiguousarray(array).tobytes())
    identity = digest.hexdigest()
    path = Path(cache) / f"{h}_{mb}_{identity}.json" if cache else None
    if path is not None and path.exists():
        return h, mb, json.loads(path.read_text())
    mix = GaussianMixture.from_dict(mix)
    fit = fit_xd(x, cov, observed=obs, weights=weight, init=mix,
                 max_iter=iterations, tol=0., regularization=floor, labels=mix.labels)
    before = np.average(mix.log_prob(xv, cv, observed=ov), weights=wv)
    after = np.average(fit.mixture.log_prob(xv, cv, observed=ov), weights=wv)
    result = dict(n_fit=len(x), n_select=len(xv), delta_select=float(after - before),
                  iterations=fit.n_iter, mixture=fit.mixture.to_dict(), identity=identity)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(result)); tmp.replace(path)
    return h, mb, result


def quasar_weights(prior, magnitude):
    """Per-slice intensities [deg^-2 mag^-1], using the contamination fit's quadrature."""
    return np.array([np.interp(magnitude, prior.mag_centres, row)
                     for row in prior.sigma]).T * np.diff(prior.z_edges)[None]


def draw_population(parts, features, kappa, rng, *, return_populations=False):
    """Draw observed relative fluxes at fixed measured r and error covariance.

    The result is normalised over the observed relative-flux coordinates.
    Non-quasar and quasar class shares use their surface densities at r.
    """
    f = features
    n, d = f.x.shape
    prior, bkg, out = (parts[k] for k in ("qso_prior", "background", "outlier"))
    wz = quasar_weights(prior, f.ref_mag)
    sq = wz.sum(1)
    sb = parts["background_density"](f.ref_mag, np.zeros(n), np.full(n, 90.))
    unseen = (1 - kappa) * sq
    pq = unseen / (unseen + sb)
    is_q = rng.random(n) < pq
    is_out = ~is_q & (rng.random(n) < out.fraction_at(f.ref_mag))
    x = np.empty((n, d))
    for j, mix in enumerate(bkg.global_):
        sel = ~is_q & ~is_out & (bkg.mag_bin(f.ref_mag) == j)
        x[sel] = mix.sample(int(sel.sum()), rng)
    x[is_out] = rng.multivariate_normal(out.mean, out.cov, size=int(is_out.sum()))
    if is_q.any():
        cum = np.cumsum(wz[is_q], axis=1) / sq[is_q, None]
        # Last cumulative mass is exactly one, including floating-point roundoff.
        cum[:, -1] = 1.
        slices = (rng.random(int(is_q.sum()))[:, None] > cum).sum(1)
        rows = np.flatnonzero(is_q)
        for j in np.unique(slices):
            sel = rows[slices == j]
            x[sel] = parts["qso"].mixtures[j].sample(sel.size, rng)
    # Noise is added only in the observed subspace; missing covariances are singular.
    patterns, inverse = np.unique(f.observed, axis=0, return_inverse=True)
    for j, pattern in enumerate(patterns):
        rows = np.flatnonzero(inverse == j)
        dims = np.flatnonzero(pattern)
        if dims.size:
            cov = f.cov[np.ix_(rows, dims, dims)]
            noise = np.linalg.cholesky(cov) @ rng.standard_normal((rows.size, dims.size, 1))
            x[np.ix_(rows, dims)] += noise[..., 0]
    result = (np.where(f.observed, x, np.nan), pq)
    if return_populations:
        return (*result, dict(quasar=is_q, outlier=is_out, background=~is_q & ~is_out))
    return result


def tail_bins(observed, simulations, edges, gates):
    """Poisson plus Monte Carlo uncertainty for the predeclared coarse tail test."""
    sims = np.asarray(simulations, float)
    expected = sims.mean(0)
    rows = []
    for i, e in enumerate(expected):
        error = np.sqrt(e + sims[:, i].var() / len(sims))
        z = (observed[i] - e) / error if error > 0 else None
        ratio = observed[i] / e if e > 0 else None
        fail = (e >= gates["tails_min_expected"]
                and (ratio > gates["tails_factor"] or ratio < 1 / gates["tails_factor"])
                and abs(z) > gates["tails_sigma"])
        rows.append(dict(sigma=[float(edges[i]), None if np.isinf(edges[i + 1]) else float(edges[i + 1])],
                         observed=int(observed[i]), expected=float(e), z=z, ratio=ratio,
                         fail=bool(fail), tested=bool(e >= gates["tails_min_expected"])))
    return rows


def field_checks(cfg, baseline, hemisphere, *, n_draws, seed, gates):
    """Counts and tails on untouched test cones; absent north cones remain untested."""
    from qso_pcolor.legacy import dereddened_relative_fluxes
    from qso_pcolor.features import FeatureSet
    root = Path(cfg["data_dir"])
    design = json.loads((root / "design.json").read_text())
    areas = json.loads((root / "areas.json").read_text())
    parts = baseline.parts[hemisphere]
    prior, bkg = parts["qso_prior"], parts["background"]
    lo, hi = baseline.manifest["domain"]["ref_mag"]
    q_total = np.sum(quasar_weights(prior, prior.mag_centres).sum(1) * np.diff(prior.mag_edges))
    samples, kappa, cells = [], [], {}
    for cone in design["cones"]:
        if cone["role"] != "test" or cone["hemisphere"] != hemisphere:
            continue
        r = dict(np.load(root / "cones" / f"cone_{cone['cone']:03d}.npz", allow_pickle=False))
        dec = baseline.selection.decide(r)
        fs, usable = dereddened_relative_fluxes(r, hemisphere)
        good = dec["accepted"] & (dec["hemisphere"] == hemisphere) & usable
        good &= (fs.ref_mag >= lo) & (fs.ref_mag < hi)
        known = r["known_quasar"].astype(bool)
        area = areas[str(cone["cone"])]["area_deg2"]
        kap = min(1., int((good & known).sum()) / (area * q_total))
        fs = fs.subset(np.flatnonzero(good & ~known))
        samples.append(fs); kappa.append(np.full(fs.n_obs, kap))
        cc = cells.setdefault(cone["cell"], dict(n=np.zeros(prior.mag_centres.size), area=0.,
                  unseen_area=0., weight=design["cells"][str(cone["cell"])]["by_hemisphere"][hemisphere]))
        cc["n"] += np.histogram(fs.ref_mag, prior.mag_edges)[0]
        cc["area"] += area; cc["unseen_area"] += area * (1 - kap)
    if not samples:
        return dict(status="untested: no reserved field cones", pass_=None)
    first = samples[0]
    keys = ("x", "cov", "observed", "ref_flux", "ref_mag", "ref_snr")
    f = FeatureSet(**{k: np.concatenate([getattr(s, k) for s in samples]) for k in keys},
                   labels=first.labels)
    kap = np.concatenate(kappa)
    sig = bkg.ood_score(f.x, f.cov, f.ref_mag, np.zeros(f.n_obs), np.full(f.n_obs, 90.), observed=f.observed)
    edges = np.asarray(cfg["recovery_validation"]["tail_distance_edges"] + [np.inf])
    observed = np.histogram(sig, edges)[0]
    rng = np.random.default_rng(seed)
    sims = []
    components = {k: [] for k in ("quasar", "outlier", "background")}
    for _ in range(n_draws):
        draw, pq, populations = draw_population(parts, f, kap, rng, return_populations=True)
        dist = bkg.ood_score(draw, f.cov, f.ref_mag, np.zeros(f.n_obs), np.full(f.n_obs, 90.), observed=f.observed)
        sims.append(np.histogram(dist, edges)[0])
        for k, mask in populations.items():
            components[k].append(np.histogram(dist[mask], edges)[0])
    bins = tail_bins(observed, sims, edges, gates)
    dm = np.diff(prior.mag_edges)
    sb = parts["background_density"](prior.mag_centres, np.zeros(len(dm)), np.full(len(dm), 90.))
    sq = quasar_weights(prior, prior.mag_centres).sum(1)
    cell = list(cells.values())
    def density(indices, predict=False):
        w = np.array([cell[i]["weight"] for i in indices])
        d = np.array([sb + cell[i]["unseen_area"] / cell[i]["area"] * sq if predict else
                      cell[i]["n"] / cell[i]["area"] / dm for i in indices])
        return np.average(d, weights=w, axis=0)
    idx = np.arange(len(cell)); obs_d, pred_d = density(idx), density(idx, True)
    boots = np.array([np.sum(density(rng.integers(0, len(cell), len(cell))) * dm) for _ in range(1000)])
    counts = dict(n_cells=len(cell), observed=obs_d.tolist(), predicted=pred_d.tolist(),
                  pooled_ratio=float(np.sum(obs_d * dm) / np.sum(pred_d * dm)),
                  ci95=(np.percentile(boots, [2.5, 97.5]) / np.sum(pred_d * dm)).tolist())
    return dict(status="measured", n=f.n_obs, n_cones=len(samples), n_draws=n_draws,
                expected_unrecognised_quasars=float(pq.sum()), bins=bins,
                pass_=not any(b["fail"] for b in bins), counts=counts,
                predicted_by_population={k: np.mean(v, axis=0).tolist() for k, v in components.items()},
                scope="Full observed field before the support gate, conditional on measured r and errors.")
