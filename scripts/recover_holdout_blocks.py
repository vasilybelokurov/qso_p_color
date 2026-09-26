#!/usr/bin/env python
"""Recover the spatial holdout of an already-trained model and record it.

Why this exists. ``train_qso_model.py`` used to reserve whole nside=4 sky
blocks before fitting but record only ``holdout_frac`` and ``holdout_nside`` --
not the seed and not the resulting block IDs. Every downstream consumer
therefore had to *re-derive* the split, and ``score_examples.py`` re-derived it
from a different sample (DESI alone, no de-duplication, no quality cut), giving
78 candidate blocks where training had 81. Same recipe, different ingredients,
different draw: 7 of its 16 "reserved" blocks were training blocks, and 52% of
the objects it called held out had been fitted.

The block partition is a pure function of (caches, release, de-duplication,
quality cut, seed, fraction) and lies entirely **upstream** of the fit, so it
can be replayed in seconds without refitting anything. The replay is only
trustworthy if it reproduces the split the model was actually trained on, so it
is checked against the four object counts the model recorded at training time:

    n_train, n_holdout, n_desi, n_sdss

Those are computed from ``is_held`` and ``channel`` after all cuts, so they
fingerprint the partition. With ~1.1e6 objects over 81 blocks, moving any single
block across the boundary shifts ``n_train`` by ~1e4; matching all four integers
exactly with a different subset is not something a wrong replay achieves. The
check can therefore raise a false alarm (if this script's counting differs from
training's) but cannot certify a wrong split -- the safe direction.

The recovered list is written as ``holdout_blocks_recovered``, never as
``holdout_blocks``, so a reader can always tell a value inferred after the fact
from one recorded by the training run itself.

    python scripts/recover_holdout_blocks.py --model models/archive/original_legacy_south/qso_south_full.json
    python scripts/recover_holdout_blocks.py --model ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")


def replay_split(
    cache_dir: Path,
    *,
    release: int,
    zmin: float,
    zmax: float,
    holdout_frac: float,
    holdout_seed: int,
    holdout_nside: int,
    use_sdss: bool = True,
) -> dict:
    """Re-run train_qso_model's block construction, without fitting.

    Returns a dict with the held block IDs and the four counts that fingerprint
    the partition. The expressions here mirror ``train_qso_model.main`` line for
    line on purpose: a paraphrase would make a count mismatch ambiguous between
    "wrong split" and "counted differently".
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_qso_model import deduplicate, load_desi, load_sdss, stack

    from qso_pcolor.background import galactic_healpix
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.features import RelativeFluxTransform, deredden

    parts = [(load_desi(cache_dir / "desi_qso_full.npz", zmin, zmax), "desi")]
    if use_sdss:
        parts.append((load_sdss(cache_dir / "dr16q_ls.npz", zmin, zmax), "sdss"))

    ra, dec, z, flux, ivar, trans, channel = [], [], [], [], [], [], []
    for r, tag in parts:
        sel = np.asarray(r["release"], int) == release
        if not sel.any():
            continue
        ra.append(np.asarray(r["ra"])[sel])
        dec.append(np.asarray(r["dec"])[sel])
        z.append(np.asarray(r["zspec"])[sel])
        flux.append(stack(r, "flux_")[sel])
        ivar.append(stack(r, "flux_ivar_")[sel])
        trans.append(stack(r, "mw_transmission_")[sel])
        channel.append(np.full(int(sel.sum()), tag))

    ra, dec, z = (np.concatenate(x) for x in (ra, dec, z))
    flux, ivar, trans = (np.concatenate(x) for x in (flux, ivar, trans))
    channel = np.concatenate(channel)

    keep = deduplicate(ra, dec, z)
    ra, dec, z, channel = ra[keep], dec[keep], z[keep], channel[keep]
    flux, ivar, trans = flux[keep], ivar[keep], trans[keep]

    f, v = deredden(flux, ivar, trans)
    fs = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)(f, v, BANDS)
    ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)

    l, b = galactic_from_equatorial(ra[ok], dec[ok])
    groups = galactic_healpix(l, b, holdout_nside)

    rng_h = np.random.default_rng(holdout_seed)
    blocks = np.unique(groups)
    n_hold = max(1, int(round(holdout_frac * blocks.size)))
    held_blocks = set(rng_h.choice(blocks, size=n_hold, replace=False).tolist())
    is_held = np.array([g in held_blocks for g in groups])

    return {
        "n_blocks": int(blocks.size),
        "held_blocks": sorted(int(x) for x in held_blocks),
        "counts": {
            "n_train": int((~is_held).sum()),
            "n_holdout": int(is_held.sum()),
            "n_desi": int((channel[ok][~is_held] == "desi").sum()),
            "n_sdss": int((channel[ok][~is_held] == "sdss").sum()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/archive/original_legacy_south/qso_south_full.json"))
    ap.add_argument("--cache", type=Path, default=Path("data"))
    ap.add_argument("--holdout-seed", type=int, default=0,
                    help="training's default; the model file does not record it")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify and report, write nothing")
    args = ap.parse_args()

    model = json.loads(args.model.read_text())
    meta = model["meta"]
    if "holdout_blocks" in meta:
        print(f"{args.model} already records holdout_blocks natively; nothing to do")
        return

    print(f"{args.model}: trained {meta['trained']}, release {meta['release']}, "
          f"nside {meta['holdout_nside']}, frac {meta['holdout_frac']}")
    print("replaying the block construction (no fitting)")
    t0 = time.time()
    got = replay_split(
        args.cache,
        release=int(meta["release"]),
        zmin=float(meta["z_range"][0]),
        zmax=float(meta["z_range"][1]),
        holdout_frac=float(meta["holdout_frac"]),
        holdout_seed=args.holdout_seed,
        holdout_nside=int(meta["holdout_nside"]),
        use_sdss=int(meta.get("n_sdss", 0)) > 0,
    )
    print(f"  {got['n_blocks']} populated blocks, {len(got['held_blocks'])} held "
          f"({time.time() - t0:.0f} s)")

    print("\nfingerprint check against the counts recorded at training time:")
    bad = []
    for k, v in got["counts"].items():
        want = int(meta[k])
        agree = v == want
        print(f"  {k:<10s} replayed {v:>9,d}   recorded {want:>9,d}   "
              f"{'ok' if agree else 'MISMATCH'}")
        if not agree:
            bad.append(k)

    if bad:
        raise SystemExit(
            "\nrefusing to write: " + ", ".join(bad) + " disagree.\n"
            "Either the on-disk caches are not the ones used for training\n"
            "(data/dr16q_ls.npz is not hash-keyed, so it can have been\n"
            "overwritten), or --holdout-seed is wrong, or this script's\n"
            "counting has drifted from train_qso_model.py. Investigate rather\n"
            "than recording a split that may not be the real one."
        )

    print("\nall four agree -- this is the partition the model was trained on")
    print(f"held blocks: {got['held_blocks']}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    meta["holdout_seed"] = args.holdout_seed
    meta["holdout_blocks_recovered"] = got["held_blocks"]
    meta["holdout_recovery"] = {
        "date": time.strftime("%Y-%m-%d"),
        "by": "scripts/recover_holdout_blocks.py",
        "n_blocks_total": got["n_blocks"],
        "verified_counts": list(got["counts"]),
        "note": "recovered after training by replaying the block construction; "
                "verified against the four counts recorded at training time",
    }
    args.model.write_text(json.dumps(model))
    print(f"\nwrote holdout_seed and holdout_blocks_recovered to {args.model}")


if __name__ == "__main__":
    main()
