#!/usr/bin/env python
"""Is the first-order luptitude error propagation good enough?

The model treats each measured luptitude u = g(f) as Gaussian around the true
one, with sigma_u = |g'(f_obs)| sigma_f (the delta method at the MEASURED flux).
The exact density of u given the true flux follows from the Gaussian flux error:
p(u | f_true) = N(g^-1(u); f_true, sigma_f) |d g^-1 / du|. This script draws
flux measurements around real fluxes and errors from the model's own samples
(reserved quasars and field sources), and reports how far the delta-method log
likelihood departs from the exact one, per band and by signal-to-noise.

    python scripts/check_luptitude_errors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAG = 2.5 / np.log(10.0)


def main() -> None:
    from qso_pcolor.multisurvey import MultiSurveyModel
    from qso_pcolor.multisurvey_data import Photometry

    model = MultiSurveyModel.load("models/multisurvey.json")
    bands = model.transform.bands
    soft = model.transform.softening
    cfg = json.loads(Path("configs/multisurvey_lsw.json").read_text())
    root = Path(cfg["data_dir"])
    rng = np.random.default_rng(0)
    rows = {}
    snr_edges = np.array([-np.inf, 1, 3, 5, 10, np.inf])
    for name in ("quasars", "background"):
        r = dict(np.load(root / f"{name}.npz"))
        p = Photometry(r["flux"], r["variance"], tuple(str(x) for x in r["bands"])).align(bands)
        held = r["held"].astype(bool)
        for j, band in enumerate(bands):
            ok = held & p.observed[:, j]
            if ok.sum() < 200:
                continue
            f_true = p.flux[ok, j]
            sig = np.sqrt(p.variance[ok, j])
            s = soft[j]
            f_obs = f_true + rng.normal(0, 1, f_true.size) * sig
            g = lambda f: -MAG * (np.arcsinh(f / (2 * s)) + np.log(s))       # noqa: E731, u - 22.5
            u_obs, u_true = g(f_obs), g(f_true)
            # exact log density of u_obs given f_true
            dfdu = MAG / np.hypot(f_obs, 2 * s)                              # |dg/df| at f_obs
            log_exact = (-0.5 * ((f_obs - f_true) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))
                         - np.log(dfdu))
            # delta method: Gaussian in u with sigma_u from the measured flux
            su = dfdu * sig
            log_delta = -0.5 * ((u_obs - u_true) / su) ** 2 - np.log(su * np.sqrt(2 * np.pi))
            diff = log_delta - log_exact
            snr = f_true / sig
            key = f"{band}"
            rec = rows.setdefault(key, {"n": 0, "bins": {}})
            rec["n"] += int(ok.sum())
            for a, b in zip(snr_edges[:-1], snr_edges[1:]):
                m = (snr >= a) & (snr < b)
                lab = f"{a:g}..{b:g}"
                v = rec["bins"].setdefault(lab, [])
                v.extend(diff[m].tolist())
            rec.setdefault("sigma_over_2s", []).extend((sig / (2 * s)).tolist())
    out = {}
    print(f"{'band':24s} {'n':>7s} {'median sig/2s':>13s} | median |dlnL| (99th pct) by S/N of the true flux")
    for band, rec in rows.items():
        summ = {}
        for lab, v in rec["bins"].items():
            v = np.abs(np.asarray(v))
            if v.size >= 50:
                summ[lab] = {"n": int(v.size), "median": float(np.median(v)),
                             "p99": float(np.percentile(v, 99))}
        allv = np.abs(np.concatenate([np.asarray(v) for v in rec["bins"].values()]))
        out[band] = {"n": rec["n"], "median_sigma_over_2s": float(np.median(rec["sigma_over_2s"])),
                     "all": {"median": float(np.median(allv)), "p99": float(np.percentile(allv, 99))},
                     "by_snr": summ}
        cells = "  ".join(f"{k}:{v['median']:.3f}({v['p99']:.2f})" for k, v in summ.items())
        print(f"{band:24s} {rec['n']:7d} {out[band]['median_sigma_over_2s']:13.2f} | {cells}")
    Path("docs/HEALTH_luptitude.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
