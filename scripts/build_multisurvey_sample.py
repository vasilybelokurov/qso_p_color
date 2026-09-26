#!/usr/bin/env python
"""Build cached, spatially held-out quasar and field photometry for seven surveys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from qso_pcolor.background import galactic_healpix
from qso_pcolor.data import (_save_npz, drop_known_quasars, fetch_known_quasars,
                             galactic_from_equatorial)
from qso_pcolor.multisurvey_data import (Photometry, band_labels,
    catalogue_photometry, cone_catalogue, match_catalogue)


def xyz(ra, dec):
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)])


def deduplicate(ra, dec, radius):
    """Keep one position from each connected close group, in stable input order."""
    pairs = cKDTree(xyz(ra, dec)).query_pairs(2*np.sin(np.deg2rad(radius/3600)/2), output_type="ndarray")
    keep = np.ones(len(ra), bool)
    for a, b in pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]:
        if keep[a]:
            keep[b] = False
    return keep


def prepare_targets(cfg, root):
    path = root / "targets.npz"
    if path.exists():
        return dict(np.load(path))
    parts = [dict(np.load(cfg[k])) for k in ("desi_cache", "sdss_cache")]
    ra, dec, z = [np.concatenate([p[k] for p in parts]) for k in ("ra", "dec", "zspec")]
    channel = np.concatenate([np.repeat(n, len(p["ra"])) for n, p in zip(("desi", "sdss"), parts)])
    keep = deduplicate(ra, dec, cfg["dedup_radius_arcsec"])
    ra, dec, z, channel = [x[keep] for x in (ra, dec, z, channel)]
    l, b = galactic_from_equatorial(ra, dec)
    eligible = np.abs(b) >= cfg["min_abs_b_deg"]
    edges = np.arange(cfg["z_min"], cfg["z_max"] + cfg["z_step"]/2, cfg["z_step"])
    rng = np.random.default_rng(cfg["seed"])
    chosen = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(eligible & (z >= lo) & (z < hi))
        chosen.extend(rng.choice(idx, min(len(idx), cfg["qso_per_z_bin"]), replace=False))
    chosen = np.sort(chosen)
    meta = json.loads(Path(cfg["holdout_model"]).read_text())["meta"]
    groups = galactic_healpix(l[chosen], b[chosen], meta["holdout_nside"])
    # Reuse every old held-out block; no previous held-out region enters fitting.
    held = np.isin(groups, meta["holdout_blocks"])
    result = dict(ra=ra[chosen], dec=dec[chosen], zspec=z[chosen], channel=channel[chosen],
                  l=l[chosen], b=b[chosen], group=groups, held=held)
    _save_npz(path, **result)
    print(f"Targets: {len(chosen):,}; held out: {held.sum():,}", flush=True)
    return result


def quasar_sample(cfg, root):
    targets = prepare_targets(cfg, root)
    out = root / "quasars.npz"
    if out.exists():
        return
    phot = []
    for name in cfg["surveys"]:
        print(f"Quasars: matching {name}", flush=True)
        raw = match_catalogue(name, targets["ra"], targets["dec"], root / "queries",
            radius_arcsec=cfg["wise_match_radius_arcsec"] if name == "allwise" else cfg["match_radius_arcsec"])
        p = catalogue_photometry(name, raw, clean=cfg["clean"], vhs_bad_bits=cfg["vhs_bad_bits"])
        phot.append(p)
        print(f"  {name}: {(p.observed.sum(1)>=2).sum():,} with >=2 bands", flush=True)
    _save_npz(out, **targets, flux=np.concatenate([p.flux for p in phot], axis=1),
              variance=np.concatenate([p.variance for p in phot], axis=1),
              bands=np.array(sum((p.bands for p in phot), ())))


def combine_field(catalogues, cfg):
    """One-to-one nearest associations; retain unmatched sources in the union.

    Greedy distance ordering prevents a single coarse catalogue source being
    copied onto several optical objects. Competing matches are left unassigned.
    """
    bands = band_labels(tuple(cfg["surveys"]))
    ra, dec = np.empty(0), np.empty(0)
    f, v = np.empty((0, len(bands))), np.empty((0, len(bands)))
    # Coarse AllWISE positions attach after the optical/NIR union is built.
    order = [s for s in cfg["surveys"] if s != "allwise"] + ["allwise"]
    for name in order:
        raw = catalogues[name]
        phot = catalogue_photometry(name, raw, clean=cfg["clean"], vhs_bad_bits=cfg["vhs_bad_bits"])
        use = np.flatnonzero(phot.observed.any(axis=1))
        if not len(use):
            continue
        p = phot.subset(use).align(bands)
        a, d = raw["ra"][use], raw["dec"][use]
        radius = cfg["wise_match_radius_arcsec"] if name == "allwise" else cfg["match_radius_arcsec"]
        if len(ra):
            dist, idx = cKDTree(xyz(ra, dec)).query(xyz(a, d), distance_upper_bound=2*np.sin(np.deg2rad(radius/3600)/2))
        else:
            dist, idx = np.full(len(a), np.inf), np.zeros(len(a), int)
        assigned = np.zeros(len(a), bool)
        occupied = set()
        for j in np.argsort(dist):
            if not np.isfinite(dist[j]):
                break
            k = int(idx[j])
            if k in occupied:
                # Multiple rows from one survey are genuine nearby detections;
                # retain the unmatched row as a separate union source below.
                continue
            occupied.add(k); assigned[j] = True
            good = p.observed[j]
            f[k, good], v[k, good] = p.flux[j, good], p.variance[j, good]
        ra, dec = np.concatenate([ra, a[~assigned]]), np.concatenate([dec, d[~assigned]])
        f, v = np.concatenate([f, p.flux[~assigned]]), np.concatenate([v, p.variance[~assigned]])
    return ra, dec, Photometry(f, v, bands)


def assign_background_holdout(cfg, root):
    """Reserve whole fields while representing every measured band on both sides."""
    path = root / "background.npz"
    arrays = dict(np.load(path)); fields = json.loads((root / "fields.json").read_text())
    observed = np.isfinite(arrays["flux"]) & np.isfinite(arrays["variance"])
    ids = np.unique(arrays["field"])
    coverage = np.array([observed[arrays["field"] == i].sum(0) for i in ids])
    rng = np.random.default_rng(cfg["seed"] + 2)
    count = max(1, round(len(ids) * cfg["background_holdout_fraction"]))
    minimum = cfg["min_band_training"]
    for _ in range(10000):
        chosen = rng.choice(len(ids), count, replace=False)
        val = coverage[chosen].sum(0); train = coverage.sum(0) - val
        if (val >= minimum).all() and (train >= minimum).all():
            held_ids = ids[chosen]
            break
    else:
        raise ValueError("background fields cannot support independent validation in every band")
    arrays["held"] = np.isin(arrays["field"], held_ids)
    _save_npz(path, **arrays)
    for field in fields:
        field["held"] = field["field"] in held_ids
    (root / "fields.json").write_text(json.dumps(fields, indent=2))


def background_sample(cfg, root):
    out = root / "background.npz"
    if out.exists():
        assign_background_holdout(cfg, root)
        return
    t = prepare_targets(cfg, root)
    rng = np.random.default_rng(cfg["seed"] + 1)
    # Spread fields over declination and distinct sky blocks, not one cone.
    selected, used = [], set()
    strata = np.array_split(np.argsort(t["dec"]), cfg["n_background_fields"])
    for stratum in strata:
        for idx in rng.permutation(stratum):
            block = int(t["group"][idx])
            if block not in used:
                selected.append(int(idx)); used.add(block); break
    if len(selected) != cfg["n_background_fields"]:
        raise ValueError("not enough distinct sky blocks for the requested background fields")
    validation_fields = set(rng.choice(len(selected), max(1, round(len(selected)*cfg["background_holdout_fraction"])), replace=False))
    chunks, fields = [], []
    for field, idx in enumerate(selected):
        path = root / f"field_{field:02d}.npz"
        a, d = float(t["ra"][idx]), float(t["dec"][idx])
        if not path.exists():
            raw = {}
            for name in cfg["surveys"]:
                print(f"Field {field+1}/{len(selected)} ({a:.3f},{d:.3f}): {name}", flush=True)
                raw[name] = cone_catalogue(name, a, d, cfg["field_radius_deg"], root / "queries")
            ra, dec, phot = combine_field(raw, cfg)
            q = fetch_known_quasars(a, d, cfg["field_radius_deg"], root / "queries" / f"known_qso_{field}.npz")
            keep = drop_known_quasars(ra, dec, q["ra"], q["dec"], radius_arcsec=cfg["dedup_radius_arcsec"])
            keep &= phot.observed.sum(axis=1) >= cfg["min_bands"]
            l, b = galactic_from_equatorial(ra[keep], dec[keep])
            _save_npz(path, ra=ra[keep], dec=dec[keep], l=l, b=b, flux=phot.flux[keep],
                      variance=phot.variance[keep], bands=np.array(phot.bands),
                      n_raw=np.array(len(ra)), n_known_qso=np.array((~drop_known_quasars(ra, dec, q["ra"], q["dec"], radius_arcsec=cfg["dedup_radius_arcsec"])).sum()))
        c = dict(np.load(path)); n = len(c["ra"])
        c["field"] = np.full(n, field); c["held"] = np.full(n, field in validation_fields)
        chunks.append(c)
        fields.append(dict(field=field, ra=a, dec=d, n=n, held=field in validation_fields,
                           n_raw=int(c["n_raw"]), n_known_qso=int(c["n_known_qso"])))
    _save_npz(out, **{k: np.concatenate([c[k] for c in chunks]) for k in
                     ("ra", "dec", "l", "b", "flux", "variance", "field", "held")},
              bands=chunks[0]["bands"])
    (root / "fields.json").write_text(json.dumps(fields, indent=2))
    assign_background_holdout(cfg, root)


def validation_background_sample(cfg, root, validation):
    """Fetch additional, exclusively held-out overlap fields; never fit them."""
    fields = validation.get("extra_background_fields", [])
    if not fields:
        return
    out = root / "validation_background.npz"
    stamp = out.with_suffix(".json")
    selection = dict(fields=fields, training_config=cfg,
                     selection=validation["extra_field_selection"])
    if out.exists():
        if json.loads(stamp.read_text()) != selection:
            raise ValueError("extra validation field selection changed: use a new cache")
        return
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    original = json.loads((root / "fields.json").read_text())
    centres = SkyCoord([f["ra"] for f in original]*u.deg,
                       [f["dec"] for f in original]*u.deg)
    chunks = []
    for i, field in enumerate(fields):
        a, d, radius = [field[k] for k in ("ra", "dec", "radius_deg")]
        if (SkyCoord(a*u.deg, d*u.deg).separation(centres).deg <=
                radius + cfg["field_radius_deg"]).any():
            raise ValueError("additional validation field overlaps an original field")
        raw = {}
        for name in cfg["surveys"]:
            print(f"Extra validation field {i}: {name}", flush=True)
            raw[name] = cone_catalogue(name, a, d, radius, root / "queries")
        ra, dec, phot = combine_field(raw, cfg)
        known = fetch_known_quasars(a, d, radius,
                                   root / "queries" / f"known_qso_validation_{i}.npz")
        keep = drop_known_quasars(ra, dec, known["ra"], known["dec"],
                                 radius_arcsec=cfg["dedup_radius_arcsec"])
        keep &= phot.observed.sum(1) >= cfg["min_bands"]
        l, b = galactic_from_equatorial(ra[keep], dec[keep])
        n = int(keep.sum())
        chunks.append(dict(ra=ra[keep], dec=dec[keep], l=l, b=b,
                           flux=phot.flux[keep], variance=phot.variance[keep],
                           field=np.full(n, len(original)+i), held=np.ones(n, bool)))
    _save_npz(out, **{k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]},
              bands=np.array(phot.bands))
    stamp.write_text(json.dumps(selection, indent=2))
    print(f"Saved {sum(len(c['ra']) for c in chunks):,} extra held-out field sources", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/multisurvey_lsw.json"))
    ap.add_argument("--part", choices=("quasars", "background", "validation", "all"), default="all")
    ap.add_argument("--validation-config", type=Path, default=Path("configs/multisurvey_validation.json"))
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    root = Path(cfg["data_dir"]); root.mkdir(parents=True, exist_ok=True)
    stamp = root / "config.json"
    if stamp.exists() and json.loads(stamp.read_text()) != cfg:
        raise ValueError("sample configuration changed: choose a new data_dir")
    stamp.write_text(json.dumps(cfg, indent=2))
    if args.part in ("quasars", "all"):
        quasar_sample(cfg, root)
    if args.part in ("background", "all"):
        background_sample(cfg, root)
    if args.part == "validation":
        validation_background_sample(cfg, root, json.loads(args.validation_config.read_text()))


if __name__ == "__main__":
    main()
