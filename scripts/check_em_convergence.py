#!/usr/bin/env python
"""Does the EM iteration cap matter? Measure it rather than guess.

``fit_sliced_model`` logs whether each slice converged or hit ``max_iter``. Some
slices hit it. That is not automatically a problem -- EM improves the likelihood
monotonically, so stopping early leaves a still-improving fit rather than a
diverged one -- but "not automatically a problem" is not evidence.

This fits one redshift slice at several iteration caps and compares the
**held-out** log density. Training likelihood would answer nothing: EM must
improve it by construction. Only the held-out number says whether the extra
iterations buy anything real or merely chase the training set.

    python scripts/check_em_convergence.py --z 1.5 --k 12
"""

from __future__ import annotations

import argparse
import time

import numpy as np

BANDS = ("g", "r", "z", "w1", "w2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--z", type=float, default=1.5, help="slice centre")
    ap.add_argument("--half-width", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--release", type=int, default=9010)
    ap.add_argument("--caps", type=int, nargs="*", default=[100, 300, 1000, 3000])
    ap.add_argument("--max-objects", type=int, default=60000,
                    help="subsample so the comparison finishes in minutes")
    args = ap.parse_args()

    from qso_pcolor.data import fetch_desi_qso_training
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.xd import fit_xd

    r = fetch_desi_qso_training("data/desi_qso_full.npz", zmin=0.4, zmax=3.6)
    sel = (np.asarray(r["release"], int) == args.release) & (
        np.abs(r["zspec"] - args.z) < args.half_width
    )
    st = lambda p: np.stack(  # noqa: E731
        [np.asarray(r[f"{p}{b}"], float)[sel] for b in BANDS], axis=1
    )
    f, v = deredden(st("flux_"), st("flux_ivar_"), st("mw_transmission_"))
    fs = RelativeFluxTransform(reference_band="r")(f, v, BANDS)
    ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
    x, c, o = fs.x[ok], fs.cov[ok], fs.observed[ok]

    rng = np.random.default_rng(0)
    if x.shape[0] > args.max_objects:
        keep = rng.choice(x.shape[0], args.max_objects, replace=False)
        x, c, o = x[keep], c[keep], o[keep]
    n = x.shape[0]
    perm = rng.permutation(n)
    tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n) :]

    print(f"slice z = {args.z} +/- {args.half_width}, release {args.release}, K = {args.k}")
    print(f"{n:,} objects ({len(tr):,} train / {len(va):,} held out)\n")
    print(f"{'max_iter':>9s} {'ran':>6s} {'converged':>10s} "
          f"{'held-out log density':>21s} {'gain':>9s} {'time':>7s}")

    base = None
    for cap in args.caps:
        t0 = time.time()
        res = fit_xd(x[tr], c[tr], n_components=args.k, observed=o[tr], seed=0,
                     max_iter=cap, tol=1e-6, regularization=1e-6)
        lp = float(np.mean(res.mixture.log_prob(x[va], c[va], observed=o[va])))
        if base is None:
            base = lp
        print(f"{cap:9d} {res.n_iter:6d} {str(res.converged):>10s} "
              f"{lp:21.5f} {lp - base:+9.5f} {time.time() - t0:6.0f}s",
              flush=True)

    print("\nThe 'gain' column is what matters: it is the improvement in held-out")
    print("log density over the smallest cap. A gain of a few thousandths is")
    print("noise and the cap is irrelevant; tenths would mean the fits are")
    print("meaningfully under-converged and need redoing.")


if __name__ == "__main__":
    main()
