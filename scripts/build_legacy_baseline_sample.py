#!/usr/bin/env python
"""Samples for the Legacy-PSF baseline: cone design, field cones, areas, quasars, companions.

Parts (run in order; each caches its output under ``data_dir``):

design     one sky partition and the field cones. Cells are Galactic nested
           HEALPix at ``cell_nside``. A cell's eligible area is the area of the
           observed bricks (nexp_r > 0, on their hemisphere's side of the
           north/south rule) with |b| >= min_abs_b_deg. Cells with >= 50 deg^2
           get 3 cones, 20-50 deg^2 get 1, smaller cells none. Cone centres are
           uniform over the eligible part of the cell, the whole cone inside
           observed bricks of one hemisphere, >= 3 deg apart within a cell.
           Cell roles: ``test`` = the archived held-out blocks; of the rest,
           ``select`` (K and start choice), ``calib`` (Student-t) and ``fit``.
fields     every DR9 row with flux_r > min_flux_r in each cone, known quasars
           flagged (not removed: the selection is applied later, once).
areas      usable area of each cone from the brick MASKBITS and nexp-r images.
quasars    the existing DESI + SDSS target draw matched to DR9 within 1".
pairs      the DESI companions of the pair validation, matched the same way.

    python scripts/build_legacy_baseline_sample.py --part design
    python scripts/build_legacy_baseline_sample.py --part all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def cell_areas(cfg: dict) -> dict:
    """Eligible observed area per (cell, hemisphere), from the brick tables."""
    from astropy.io import fits
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.legacy import is_north
    out = {}
    observed = {}
    for h, path in cfg["hemisphere_bricks"].items():
        t = fits.getdata(path)
        l, b = galactic_from_equatorial(t["ra"], t["dec"])
        on_side = is_north(t["ra"], t["dec"], b) == (h == "north")
        obs = (t["nexp_r"] > 0) & on_side
        observed[h] = set(np.char.strip(np.asarray(t["brickname"][obs]).astype(str)))
        ok = obs & (np.abs(b) >= cfg["min_abs_b_deg"])
        cell = galactic_healpix(l[ok], b[ok], cfg["cell_nside"])
        area = np.asarray(t["area"][ok], float)
        for c in np.unique(cell):
            out.setdefault(int(c), {})[h] = float(area[cell == c].sum())
    return out, observed


def design(cfg: dict, root: Path) -> None:
    import healpy as hp
    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.legacy import NORTH_DEC_MIN, assign_bricks, is_north, load_bricks

    path = root / "design.json"
    if path.exists():
        print(f"{path} exists")
        return
    rng = np.random.default_rng(cfg["seed"])
    areas, observed = cell_areas(cfg)
    bricks = load_bricks(cfg["bricks"])
    names = bricks["brickname"]
    held = set(int(x) for x in json.loads(Path(cfg["holdout_model"]).read_text())["meta"]["holdout_blocks"])
    nside, r = cfg["cell_nside"], cfg["cone_radius_deg"]
    # ring of test points: the whole cone must lie in observed bricks
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    def cone_points(ra, dec):
        pts_ra, pts_dec = [ra], [dec]
        for rr in (r / 2, r):
            dra = rr * np.cos(ang) / np.cos(np.deg2rad(dec))
            pts_ra.extend(ra + dra); pts_dec.extend(dec + rr * np.sin(ang))
        return np.array(pts_ra) % 360, np.array(pts_dec)

    cells = sorted(areas)
    # roles: test = archived held-out blocks; others shuffled into select/calib/fit
    others = [c for c in cells if c not in held]
    perm = rng.permutation(len(others))
    n_sel = round(cfg["partition"]["select_fraction"] * len(others))
    n_cal = round(cfg["partition"]["calib_fraction"] * len(others))
    role = {c: "test" for c in cells if c in held}
    for i, j in enumerate(perm):
        role[others[j]] = "select" if i < n_sel else ("calib" if i < n_sel + n_cal else "fit")

    cones, report = [], {}
    for c in cells:
        a_cell = sum(areas[c].values())
        n_want = next((n for lo, n in cfg["cones_per_cell"] if a_cell >= lo), 0)
        report[c] = dict(area=a_cell, by_hemisphere=areas[c], role=role[c], n_wanted=n_want)
        if not n_want:
            continue
        # uniform points in the cell: sample the cell's bounding cap, keep the cell
        vec = hp.pix2vec(nside, c, nest=True)
        centre_l, centre_b = hp.vec2ang(np.array(vec), lonlat=True)
        chosen, sep = [], cfg["min_cone_separation_deg"]
        attempts = 0
        while len(chosen) < n_want and attempts < 20000:
            attempts += 1
            if attempts % 5000 == 0:
                sep /= 2                                  # small cells: relax, recorded
            zz = rng.uniform(np.cos(np.deg2rad(20.0)), 1.0)
            ph = rng.uniform(0, 2 * np.pi)
            v = np.array([np.sqrt(1 - zz**2) * np.cos(ph), np.sqrt(1 - zz**2) * np.sin(ph), zz])
            # rotate +z to the cell centre
            cl, cb = np.deg2rad(centre_l[0]), np.deg2rad(centre_b[0])
            ry = np.array([[np.sin(cb), 0, np.cos(cb)], [0, 1, 0], [-np.cos(cb), 0, np.sin(cb)]])
            rz = np.array([[np.cos(cl), -np.sin(cl), 0], [np.sin(cl), np.cos(cl), 0], [0, 0, 1]])
            w = rz @ ry @ v
            if hp.vec2pix(nside, *w, nest=True) != c:
                continue
            l, b = hp.vec2ang(w, lonlat=True)
            l, b = float(l[0]), float(b[0])
            if abs(b) < cfg["min_abs_b_deg"]:
                continue
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            eq = SkyCoord(l * u.deg, b * u.deg, frame="galactic").icrs
            ra, dec = float(eq.ra.deg), float(eq.dec.deg)
            h = "north" if is_north([ra], [dec], [b])[0] else "south"
            if b > 0 and abs(dec - NORTH_DEC_MIN) < cfg["boundary_margin_deg"]:
                continue
            pr, pd = cone_points(ra, dec)
            k = assign_bricks(bricks, pr, pd)
            if (k < 0).any() or not all(n in observed[h] for n in names[k]):
                continue
            if any(np.degrees(np.arccos(np.clip(
                    np.dot(hp.ang2vec(l, b, lonlat=True), hp.ang2vec(x["l"], x["b"], lonlat=True)),
                    -1, 1))) < sep for x in chosen):
                continue
            chosen.append(dict(ra=ra, dec=dec, l=l, b=b, hemisphere=h))
        for x in chosen:
            x.update(cone=len(cones), cell=c, role=role[c], cell_area_deg2=a_cell,
                     n_cones_in_cell=len(chosen), min_separation_used_deg=sep)
            cones.append(x)
        report[c].update(n_placed=len(chosen), attempts=attempts)
        print(f"cell {c:3d} {role[c]:6s} {a_cell:6.1f} deg2: {len(chosen)}/{n_want} cones "
              f"({attempts} draws)", flush=True)
    out = dict(config=cfg, cones=cones, cells={str(k): v for k, v in report.items()},
               built=time.strftime("%Y-%m-%d"))
    path.write_text(json.dumps(out, indent=1))
    by = {}
    for x in cones:
        by[x["role"]] = by.get(x["role"], 0) + 1
    print(f"{len(cones)} cones; by role {by}; wrote {path}")


def fields(cfg: dict, root: Path) -> None:
    from qso_pcolor.data import _save_npz, fetch_known_quasars, drop_known_quasars
    from qso_pcolor.legacy import legacy_cone
    d = json.loads((root / "design.json").read_text())
    for x in d["cones"]:
        path = root / "cones" / f"cone_{x['cone']:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        rows = legacy_cone(x["ra"], x["dec"], cfg["cone_radius_deg"],
                           root / "queries" / f"cone_{x['cone']:03d}.npz",
                           min_flux_r=cfg["min_flux_r_nmgy"])
        q = fetch_known_quasars(x["ra"], x["dec"], cfg["cone_radius_deg"],
                                root / "queries" / f"known_qso_{x['cone']:03d}.npz")
        known = ~drop_known_quasars(rows["ra"], rows["dec"], q["ra"], q["dec"], radius_arcsec=1.0)
        rows = {k: v for k, v in rows.items() if not k.startswith("_")}
        _save_npz(path, **rows, known_quasar=known, cone=np.full(len(rows["ra"]), x["cone"]))
        print(f"cone {x['cone']:3d}: {len(rows['ra']):6d} rows, {known.sum():4d} known quasars "
              f"({time.time() - t0:.0f} s)", flush=True)


def areas(cfg: dict, root: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from qso_pcolor.legacy import brick_image_fetcher, cone_usable_fraction, load_bricks
    d = json.loads((root / "design.json").read_text())
    path = root / "areas.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    bricks = load_bricks(cfg["bricks"])
    fetch = brick_image_fetcher(cfg["brick_image_cache"])

    def one(x):
        return x["cone"], cone_usable_fraction(x["ra"], x["dec"], cfg["cone_radius_deg"],
                                               x["hemisphere"], bricks, fetch,
                                               n_points=cfg["area_points_per_cone"],
                                               seed=cfg["seed"] + x["cone"])
    todo = [x for x in d["cones"] if str(x["cone"]) not in done]
    with ThreadPoolExecutor(8) as ex:
        for i, (k, res) in enumerate(ex.map(one, todo)):
            done[str(k)] = {kk: v for kk, v in res.items() if kk != "bricks"} | {
                "unobserved_bricks": [b for b, v in res["bricks"].items() if not v["observed"]]}
            print(f"cone {k:3d}: usable {res['fraction']:.3f} over {res['n_bricks']} bricks", flush=True)
            if i % 10 == 9:
                path.write_text(json.dumps(done, indent=1))
    path.write_text(json.dumps(done, indent=1))


def quasars(cfg: dict, root: Path) -> None:
    from qso_pcolor.data import _save_npz
    from qso_pcolor.legacy import legacy_match
    out = root / "quasars.npz"
    if out.exists():
        return
    t = dict(np.load(cfg["targets"]))
    m = legacy_match(t["ra"], t["dec"], root / "queries", radius_arcsec=cfg["match_radius_arcsec"])
    _save_npz(out, **{f"target_{k}": v for k, v in t.items()},
              **{k: v for k, v in m.items() if k not in ("idx", "input_ra", "input_dec")})
    print(f"quasars: {t['ra'].size:,} targets, {(m['release'] > 0).sum():,} matched")


def pairs(cfg: dict, root: Path) -> None:
    from qso_pcolor.data import _save_npz
    from qso_pcolor.legacy import legacy_match
    out = root / "companions.npz"
    if out.exists():
        return
    p = np.load(cfg["pairs"])
    ra, dec = np.asarray(p["comp_ra"], float), np.asarray(p["comp_dec"], float)
    # one query per distinct position
    key = np.round(ra * 1e6).astype(np.int64) * 10**9 + np.round((dec + 90) * 1e6).astype(np.int64)
    uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    m = legacy_match(ra[first], dec[first], root / "queries", radius_arcsec=cfg["match_radius_arcsec"])
    _save_npz(out, pair_row_to_match=inverse,
              **{k: v for k, v in m.items() if k not in ("idx",)})
    print(f"companions: {ra.size:,} pair rows, {uniq.size:,} distinct positions, "
          f"{(m['release'] > 0).sum():,} matched")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/legacy_baseline.json"))
    ap.add_argument("--part", choices=("design", "fields", "areas", "quasars", "pairs", "all"),
                    default="all")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["data_dir"]); root.mkdir(parents=True, exist_ok=True)
    stamp = root / "config.json"
    if stamp.exists() and json.loads(stamp.read_text()) != cfg:
        raise SystemExit("configuration changed: choose a new data_dir")
    stamp.write_text(json.dumps(cfg, indent=2))
    for part, fn in (("design", design), ("fields", fields), ("areas", areas),
                     ("quasars", quasars), ("pairs", pairs)):
        if args.part in (part, "all"):
            fn(cfg, root)


if __name__ == "__main__":
    main()
