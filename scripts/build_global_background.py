#!/usr/bin/env python
"""Fit the shipped footprint-average background model, with its provenance.

The scorer needs three things: the quasar colour model, a background colour
model, and the two surface densities. Only the first was committed, so a fresh
clone could not compute a single number without WSDB access. This script
builds the other half of what ships.

What it is. A single colour mixture per magnitude bin, fitted to **every**
catalogue source -- PSF and extended alike -- pooled from eight $0.5^\\circ$
fields spread across the southern footprint, with sources in masked regions
removed (``maskbits = 0``), known quasars removed, and the surveyed area
reduced by the masked fraction. That is the population a chance neighbour is
drawn from, and it is built the same way as the quasar model
(``maskbits = 0``) and as ``fit_local_background``.

What it is not. Not local: the field population varies across the sky, and a
candidate at low latitude or in a crowded field deserves its own cone
(``fit_local_background``), which needs the database. This model is the
sensible default when you have no database and a candidate at $|b| \\gtrsim
20^\\circ$; the scorer records which one was used.

Why not the earlier eight-field model. ``score_examples.fit_global_background``
selected ``type = 'PSF'`` only, which was right for a figure about point
sources and wrong as a general background: a model with no galaxies in it
makes every galaxy look less like "background" than it is. It also recorded no
provenance. This one records the cone centres, the counts at each cut, the
mask fractions and the query.

    python scripts/build_global_background.py
    python scripts/build_global_background.py --n-cones 12 --radius 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")
MAG_EDGES = np.array([17.0, 19.5, 20.5, 21.5, 22.5])
SYSTEM = "ls_dr9_south_grzw"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cones", type=int, default=8)
    ap.add_argument("--radius", type=float, default=0.5, help="degrees")
    ap.add_argument("--seed", type=int, default=0,
                    help="cone centres are drawn from this seed; 0 reproduces "
                         "the fields used for the examples figure")
    ap.add_argument("--n-components", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("models/archive/original_legacy_south/background_south_global.json"))
    ap.add_argument("--out-density", type=Path,
                    default=Path("models/archive/original_legacy_south/background_density_south_global.json"))
    args = ap.parse_args()

    from qso_pcolor.background import _RELEASES_FOR_SYSTEM, fit_background_model
    from qso_pcolor.data import (drop_known_quasars, fetch_known_quasars,
                                 fetch_ls_background, galactic_from_equatorial)
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.priors import BackgroundSurfaceDensity

    # Candidate centres inside the DECaLS-south footprint where DESI/SDSS also
    # observed (so known quasars CAN be removed), away from the plane. A first
    # version drew uniformly over RA 140-350, dec +/-15: three of eight cones
    # returned zero sources, two more were half covered, and each still added
    # its full pi R^2 to the area -- understating Sigma_B -- while cones beyond
    # the spectroscopic footprint kept their quasars. Coverage is now checked
    # per cone and a cone that fails is skipped, not averaged in.
    rng = np.random.default_rng(args.seed)
    cand = []
    while len(cand) < 6 * args.n_cones:
        a = float(rng.choice([rng.uniform(130, 250), rng.uniform(335, 400) % 360]))
        d = float(rng.uniform(-8, 25))
        l, b = galactic_from_equatorial(np.array([a]), np.array([d]))
        if abs(float(b[0])) > 25:
            cand.append((a, d))
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    full_cone_min = 20000            # a fully imaged 0.5 deg cone at these depths

    X, V, O, M, L, B = [], [], [], [], [], []
    area, cones, skipped = 0.0, [], []
    t0 = time.time()
    for i, (a, d) in enumerate(cand):
        if len(cones) >= args.n_cones:
            break
        r = fetch_ls_background(Path("data") / f"global_all_{i:02d}.npz",
                                ra=a, dec=d, radius_deg=args.radius)
        n_raw = int(np.size(r["ra"]))
        if n_raw < full_cone_min * (args.radius / 0.5) ** 2:
            skipped.append({"ra": a, "dec": d, "n_raw": n_raw, "why": "not fully covered"})
            print(f"  skip ({a:7.3f},{d:+7.3f})  {n_raw:7,d} raw: not fully covered")
            continue
        rel = sorted(int(x) for x in np.unique(np.asarray(r["release"], int)))
        if not set(rel) <= _RELEASES_FOR_SYSTEM[SYSTEM]:
            raise SystemExit(f"cone {i} at ({a:.2f},{d:.2f}) returned releases {rel}; "
                             f"not all {SYSTEM}")
        unmasked = np.asarray(r["maskbits"], int) == 0
        q = fetch_known_quasars(a, d, args.radius, cache=Path("data") / f"global_qso_{i:02d}.npz")
        if np.size(q["ra"]) == 0:
            skipped.append({"ra": a, "dec": d, "n_raw": n_raw, "why": "no spectroscopic quasars here"})
            print(f"  skip ({a:7.3f},{d:+7.3f})  no known quasars to remove: outside the spectroscopic footprint")
            continue
        notq = drop_known_quasars(r["ra"], r["dec"], q["ra"], q["dec"])
        keep = unmasked & notq
        f, v = deredden(
            np.stack([np.asarray(r[f"flux_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(r[f"flux_ivar_{b}"], float) for b in BANDS], 1)[keep],
            np.stack([np.asarray(r[f"mw_transmission_{b}"], float) for b in BANDS], 1)[keep])
        fs = tr(f, v, BANDS)
        ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag) & (fs.ref_mag < MAG_EDGES[-1])
        l, b = galactic_from_equatorial(np.asarray(r["ra"])[keep][ok],
                                        np.asarray(r["dec"])[keep][ok])
        mask_frac = float(np.mean(~unmasked)) if n_raw else 0.0
        cone_area = np.pi * args.radius**2 * (1.0 - mask_frac)
        area += cone_area
        X.append(fs.x[ok]); V.append(fs.cov[ok]); O.append(fs.observed[ok])
        M.append(fs.ref_mag[ok]); L.append(l); B.append(b)
        cones.append({"ra": a, "dec": d, "l": float(np.median(l)), "b": float(np.median(b)),
                      "n_raw": n_raw, "n_masked": int((~unmasked).sum()),
                      "n_known_quasars": int((unmasked & ~notq).sum()),
                      "n_fitted": int(ok.sum()), "mask_fraction": mask_frac,
                      "area_deg2": float(cone_area), "releases": rel})
        print(f"  cone {len(cones):2d} ({a:7.3f},{d:+7.3f})  l,b=({cones[-1]['l']:6.1f},{cones[-1]['b']:+5.1f})  "
              f"{n_raw:7,d} raw  {int((~unmasked).sum()):6,d} masked  "
              f"{cones[-1]['n_known_quasars']:4d} quasars  {int(ok.sum()):7,d} fitted")
    X = np.concatenate(X); V = np.concatenate(V); O = np.concatenate(O)
    M = np.concatenate(M); L = np.concatenate(L); B = np.concatenate(B)
    print(f"  pooled: {X.shape[0]:,} sources over {area:.2f} deg^2 (mask-corrected) "
          f"({time.time() - t0:.0f} s)")

    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "by": "scripts/build_global_background.py",
        "kind": "footprint average: one mixture per magnitude bin, all source types",
        "selection": "decals_dr9.main, maskbits = 0, known quasars removed, "
                     "release in {9010, 9012}, no morphology cut",
        "n_cones": len(cones), "radius_deg": args.radius, "seed": args.seed,
        "cones": cones, "skipped": skipped, "full_cone_min_sources": full_cone_min,
        "n_fitted": int(X.shape[0]), "area_deg2": float(area),
        "n_components": args.n_components, "mag_edges": MAG_EDGES.tolist(),
        "min_ref_snr": 5.0, "min_dims": 3,
        "caveat": "not local; the field population varies across the sky, and a "
                  "candidate at low |b| or in a crowded field should use "
                  "fit_local_background",
    }
    t0 = time.time()
    model = fit_background_model(
        X, V, M, L, B, mag_edges=MAG_EDGES, nside=1, nside_parent=1, observed=O,
        n_components=args.n_components, min_per_cell=10**9, n0=500.0,
        system=SYSTEM, labels=fs.labels, meta=meta,
        seed=0, max_iter=300, regularization=1e-6,
    )
    dens = BackgroundSurfaceDensity.from_catalogue(
        M, L, B, mag_edges=MAG_EDGES, nside=1, nside_parent=1,
        total_area_deg2=area, meta=meta,
    )
    print(f"  fitted in {time.time() - t0:.0f} s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out); dens.save(args.out_density)
    print(f"wrote {args.out} and {args.out_density}")


if __name__ == "__main__":
    main()
