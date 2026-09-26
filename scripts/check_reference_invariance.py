#!/usr/bin/env python
"""Does the answer depend on which band is the reference?

In an exact generative model Sigma_H(u_a) p(u_rest | u_a, H) is the same joint
intensity whichever measured band is a. Here the priors are built separately
per reference band and the field density can route through dedicated fits, so
the choice could matter. Reserved quasars (at their own z) and reserved field
sources (at a random z) with SDSS, Legacy south and PS1 r all measured are
scored with each of the three as reference, all other bands unchanged.

    python scripts/check_reference_invariance.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def main() -> None:
    from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.qso_model import RedshiftMatch

    model = MultiSurveyModel.load("models/multisurvey.json")
    priors = load_priors("models/multisurvey_priors.json", model)
    out = MultiSurveyOutlier.load("models/multisurvey_outlier.json")
    B = model.transform.bands
    refs = ("sdss:r", "decals_dr9_south:r", "ps1:r")
    root = Path(json.loads(Path("configs/multisurvey_lsw.json").read_text())["data_dir"])
    rng = np.random.default_rng(0)
    results = {}
    for name in ("quasars", "background"):
        r = dict(np.load(root / f"{name}.npz"))
        p = Photometry(r["flux"], r["variance"], tuple(str(x) for x in r["bands"])).align(B)
        ok = r["held"].astype(bool) & p.observed[:, [B.index(b) for b in refs]].all(1)
        if name == "quasars":
            ok &= (r["zspec"] > 0.2) & (r["zspec"] < 4.3)
        idx = np.sort(rng.choice(np.flatnonzero(ok), min(400, ok.sum()), replace=False))
        z0 = r["zspec"][idx] if name == "quasars" else rng.uniform(0.3, 4.0, idx.size)
        ph = p.subset(idx)
        scores = {}
        for ref in refs:
            m = replace(model, reference_priority=(ref,) + tuple(b for b in model.reference_priority if b != ref))
            rows = m.score(ph, z_primary=z0, l_deg=r["l"][idx], b_deg=r["b"][idx],
                           match=RedshiftMatch(half_width_kms=2000.0), min_bands=2,
                           priors=priors, outlier=out)
            assert all(s.reference_band == ref for s in rows)
            scores[ref] = {k: np.array([getattr(s, k) for s in rows], float)
                           for k in ("log_r_per_unit_z", "log_bayes_factor_qz_bkg", "p_sameq",
                                     "p_zmatch_given_qso")}
        res = {"n": int(idx.size)}
        for a, b in (("decals_dr9_south:r", "sdss:r"), ("decals_dr9_south:r", "ps1:r"), ("sdss:r", "ps1:r")):
            row = {}
            for k in ("log_r_per_unit_z", "log_bayes_factor_qz_bkg"):
                d = scores[b][k] - scores[a][k]
                d = d[np.isfinite(d)]
                row[k] = {"median": float(np.median(d)), "median_abs": float(np.median(np.abs(d))),
                          "p16_84": np.percentile(d, [16, 84]).tolist()}
            lp = np.log10(scores[b]["p_sameq"]) - np.log10(scores[a]["p_sameq"])
            lp = lp[np.isfinite(lp)]
            row["log10_p_sameq"] = {"median": float(np.median(lp)), "median_abs": float(np.median(np.abs(lp)))}
            from scipy.stats import spearmanr
            fin = np.isfinite(scores[a]["log_r_per_unit_z"]) & np.isfinite(scores[b]["log_r_per_unit_z"])
            row["spearman_log_r"] = float(spearmanr(scores[a]["log_r_per_unit_z"][fin],
                                                    scores[b]["log_r_per_unit_z"][fin]).statistic)
            res[f"{b} - {a}"] = row
        results[name] = res
        print(f"[{name}] n={idx.size}")
        for k, v in res.items():
            if k == "n":
                continue
            print(f"  {k:32s} dlnR median {v['log_r_per_unit_z']['median']:+.2f} |.| "
                  f"{v['log_r_per_unit_z']['median_abs']:.2f} (16-84 {v['log_r_per_unit_z']['p16_84'][0]:+.2f},"
                  f"{v['log_r_per_unit_z']['p16_84'][1]:+.2f})  dlnBF |.| {v['log_bayes_factor_qz_bkg']['median_abs']:.2f}"
                  f"  dlog10 p |.| {v['log10_p_sameq']['median_abs']:.2f}  rho(lnR) {v['spearman_log_r']:.3f}")
    Path("docs/HEALTH_reference.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
