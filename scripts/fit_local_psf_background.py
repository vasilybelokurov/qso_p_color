#!/usr/bin/env python
"""Fit a PSF background in a candidate's cone, excluding primary and companion.

Input is a cached Legacy catalogue with known_quasar flags. Supply the usable
area measured from imaging masks, including the excluded apertures. No source
count proxy for area is used. The output can be passed to XDQSOBaseline.load
with background=PATH and is valid only inside the declared cone.
"""
import argparse
from pathlib import Path

import numpy as np

from qso_pcolor.baseline import XDQSOBaseline
from qso_pcolor.spatial import fit_local_background


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path("models/legacy_psf_xdqso/current"))
    ap.add_argument("--spatial-background", type=Path)
    ap.add_argument("--cone", type=Path, required=True)
    ap.add_argument("--hemisphere", choices=("south", "north"), required=True)
    ap.add_argument("--centre", type=float, nargs=2, required=True, metavar=("RA", "DEC"))
    ap.add_argument("--radius-deg", type=float, required=True)
    ap.add_argument("--usable-area-deg2", type=float, required=True)
    ap.add_argument("--exclude", type=float, nargs=2, action="append", required=True, metavar=("RA", "DEC"))
    ap.add_argument("--exclusion-arcsec", type=float, required=True)
    ap.add_argument("--colour-n0", type=float, required=True)
    ap.add_argument("--density-n0", type=float, required=True)
    ap.add_argument("--max-iter", type=int, required=True)
    ap.add_argument("--tol", type=float, required=True)
    ap.add_argument("--min-density-fraction", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    baseline = XDQSOBaseline.load(args.bundle, background=args.spatial_background)
    excluded = np.asarray(args.exclude)
    with np.load(args.cone, allow_pickle=False) as f:
        rows = {k: f[k] for k in f.files}
    result = fit_local_background(baseline, rows, hemisphere=args.hemisphere,
        centre_ra=args.centre[0], centre_dec=args.centre[1], radius_deg=args.radius_deg,
        usable_area_deg2=args.usable_area_deg2, exclude_ra=excluded[:, 0], exclude_dec=excluded[:, 1],
        exclusion_arcsec=args.exclusion_arcsec, colour_n0=args.colour_n0, density_n0=args.density_n0,
        max_iter=args.max_iter, tol=args.tol, min_density_fraction=args.min_density_fraction)
    result.save(args.output)
    print(f"Saved local background {result.identity}: {result.meta['n_used']} PSF sources")


if __name__ == "__main__":
    main()
