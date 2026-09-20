#!/usr/bin/env python
"""Did widening the model's redshift range buy anything real?

Appending slices is only worth doing if the new slices carry information the
old model did not. Outside its support the old model clamps its interpolation
index and replays the edge slice, so it still *returns* a density for a z = 4.2
quasar -- just the z = 3.55 one. The question is whether the fitted slice
predicts held-out colours better than that replay.

The test uses quasars in the model's **reserved** spatial blocks, which neither
model was fitted on, and compares

    log p(c | Q, z_spec)

object by object under the two models. A positive mean difference means the new
slices describe real colour evolution; a difference near zero would mean quasar
colours stop changing beyond the old edge and the extension is cosmetic.

Inside the old support the two models must agree exactly -- the original
mixtures are untouched -- so that region doubles as a check that the splice did
not disturb anything.

    python scripts/check_redshift_extension.py \
        --old models/qso_south_full.json --new models/qso_south_extended.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, default=Path("models/qso_south_full.json"))
    ap.add_argument("--new", type=Path, default=Path("models/qso_south_extended.json"))
    ap.add_argument("--cache", type=Path, default=Path("data"))
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extend_qso_model_redshift import load_edge_sample

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.qso_model import SlicedColourRedshiftModel

    old = SlicedColourRedshiftModel.load(args.old)
    new = SlicedColourRedshiftModel.load(args.new)
    print(f"old support {old.support[0]:.2f}-{old.support[1]:.2f}  "
          f"({len(old.mixtures)} slices)")
    print(f"new support {new.support[0]:.2f}-{new.support[1]:.2f}  "
          f"({len(new.mixtures)} slices)")

    # 1. the splice must not have moved anything inside the old support
    lo_old, hi_old = old.support
    keep = (new.z_centres >= lo_old - 1e-9) & (new.z_centres <= hi_old + 1e-9)
    assert keep.sum() == old.z_centres.size, "slice count inside old support changed"
    same = all(
        np.allclose(a.means, b.means) and np.allclose(a.covs, b.covs)
        and np.allclose(a.weights, b.weights)
        for a, b in zip(old.mixtures, [m for m, k in zip(new.mixtures, keep) if k])
    )
    print(f"original mixtures unchanged by the splice: {same}")
    if not same:
        raise SystemExit("the splice altered existing slices; that is a bug")

    meta = new.meta
    held = set(int(x) for x in (meta.get("holdout_blocks")
                                or meta["holdout_blocks_recovered"]))
    nside = int(meta.get("holdout_nside", 4))
    rel = int(meta["release"])
    use_sdss = int(meta.get("n_sdss", 0)) > 0
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)

    print("\nheld-out quasars OUTSIDE the old support "
          "(neither model was fitted on these blocks)")
    print(f"{'range':>14s} {'n':>7s} {'old':>9s} {'new':>9s} {'gain':>9s}")
    rows = []
    for zlo, zhi in ((0.0, lo_old), (hi_old, float(new.support[1]) + 0.05)):
        d = load_edge_sample(args.cache, zlo, zhi, rel, use_sdss)
        f, v = deredden(d["flux"], d["ivar"], d["trans"])
        fs = tr(f, v, BANDS)
        ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
        l, b = galactic_from_equatorial(d["ra"][ok], d["dec"][ok])
        sel = np.flatnonzero(ok)[np.isin(galactic_healpix(l, b, nside), list(held))]
        # restrict to the band the new model actually covers
        sel = sel[(d["z"][sel] >= new.support[0]) & (d["z"][sel] <= new.support[1])]
        if sel.size == 0:
            continue
        zz = d["z"][sel]
        lo_ = np.array([old.log_p_colour_given_z(fs.x[[i]], fs.cov[[i]],
                                                 np.array([d["z"][i]]),
                                                 observed=fs.observed[[i]])[0, 0]
                        for i in sel])
        ln = np.array([new.log_p_colour_given_z(fs.x[[i]], fs.cov[[i]],
                                                np.array([d["z"][i]]),
                                                observed=fs.observed[[i]])[0, 0]
                       for i in sel])
        print(f"{zlo:5.2f}-{zhi:5.2f}  {sel.size:7,d} "
              f"{np.mean(lo_):9.3f} {np.mean(ln):9.3f} {np.mean(ln - lo_):+9.3f}")
        rows.append((zz, lo_, ln))

    if rows:
        zz = np.concatenate([r[0] for r in rows])
        lo_ = np.concatenate([r[1] for r in rows])
        ln = np.concatenate([r[2] for r in rows])
        g = ln - lo_
        print(f"\noverall: {g.size:,} held-out quasars, mean gain "
              f"{g.mean():+.3f} nats, median {np.median(g):+.3f}")
        print(f"  fraction improved: {100 * (g > 0).mean():.1f}%")
        print("\nby redshift:")
        edges = np.arange(np.floor(zz.min() * 5) / 5, zz.max() + 0.2, 0.2)
        for a, bnd in zip(edges[:-1], edges[1:]):
            m = (zz >= a) & (zz < bnd)
            if m.sum() > 50:
                print(f"  {a:4.2f}-{bnd:4.2f}  n = {int(m.sum()):6,d}  "
                      f"gain {g[m].mean():+8.3f}")
        print("\nA gain of order 1 nat or more means the new slices describe")
        print("colour evolution the edge-slice replay was getting wrong. A gain")
        print("near zero would mean the extension is cosmetic.")


if __name__ == "__main__":
    main()
