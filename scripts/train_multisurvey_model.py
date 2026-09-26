#!/usr/bin/env python
"""Fit and cache joint survey-band mixtures, retaining the original LS models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import (BandLuptitudeTransform, MultiSurveyModel,
                                   conditional_log_prob)
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.xd import _init_mixture, fit_xd


def redshift_edges(z_min, z_max, step):
    """Include the declared endpoint once, without a roundoff-width final bin."""
    if not np.isfinite([z_min, z_max, step]).all() or step <= 0 or z_max <= z_min:
        raise ValueError("redshift bounds and step must define positive finite bins")
    edges = np.arange(z_min, z_max, step)
    tolerance = 8*np.finfo(float).eps*max(abs(z_min), abs(z_max), step)
    if len(edges) > 1 and abs(edges[-1]-z_max) <= tolerance:
        edges[-1] = z_max
    else:
        edges = np.append(edges, z_max)
    return edges


def reference_order(bands):
    preferred = ["sdss:r", "decals_dr9_south:r", "decals_dr9_north:r", "ps1:r",
                 "nsc:r", "skymapper:r", "vhs:j", "allwise:w1"]
    return tuple([b for b in preferred if b in bands] + [b for b in bands if b not in preferred])


def colour_log_prob(mix, features, priority):
    order = np.array([features.labels.index(b) for b in priority])
    anchor = order[np.argmax(features.observed[:, order], axis=1)]
    lp = np.empty(features.n_obs)
    for a in np.unique(anchor):
        sel = anchor == a
        lp[sel] = conditional_log_prob(mix, features.x[sel], features.cov[sel],
                                      features.observed[sel], int(a))
    return lp


def sample_patterns(observed, eligible, budget, minimum, rng):
    """Retain rare measurement patterns with weights restoring their abundance."""
    rows=np.flatnonzero(eligible)
    _, inverse, counts=np.unique(np.packbits(observed[rows],axis=1),axis=0,
                                return_inverse=True,return_counts=True)
    if np.minimum(counts,minimum).sum()>budget:
        raise ValueError("background budget is too small to retain rare measurement patterns")
    chosen=np.zeros(len(rows),bool)
    for g,count in enumerate(counts):
        candidates=np.flatnonzero(inverse==g)
        chosen[rng.choice(candidates,min(int(count),minimum),replace=False)]=True
    remaining=min(budget,len(rows))-int(chosen.sum())
    if remaining:
        chosen[rng.choice(np.flatnonzero(~chosen),remaining,replace=False)]=True
    selected_counts=np.bincount(inverse[chosen],minlength=len(counts))
    weights=np.zeros(len(eligible))
    weights[rows[chosen]]=counts[inverse[chosen]]/selected_counts[inverse[chosen]]
    return rows[chosen],weights


def fit_and_select(name, features, train, validation, cfg, path, priority, seed):
    if path.exists():
        record = json.loads(path.read_text())
        print(f"{name}: cached K={record['selected_k']}", flush=True)
        return GaussianMixture.from_dict(record["mixture"]), record
    grid = cfg["background_k_candidates"] if name == "background" else cfg["k_candidates"]
    # Initialise absent dimensions from TRAINING rows across the whole range.
    # Neither the final spatial holdout nor internal selection rows enter this.
    global_train = cfg["_global_train"]
    weights=cfg.get("_weights",np.ones(features.n_obs))
    scores, candidates = {}, {}
    start = time.monotonic()
    for k in grid:
        init = _init_mixture(features.x[global_train], features.observed[global_train],
                             k, np.random.default_rng(seed+k))
        result = fit_xd(features.x[train], features.cov[train], observed=features.observed[train],
                        weights=weights[train],
                        init=init, seed=seed, labels=features.labels,
                        max_iter=cfg["selection_max_iter"], tol=cfg["tol"],
                        regularization=cfg["regularization"])
        scores[k] = float(np.average(colour_log_prob(result.mixture, features.subset(validation), priority),
                                     weights=weights[validation]))
        candidates[k] = result.mixture
        print(f"{name}: K={k}, selection colour log density {scores[k]:.4f} ({time.monotonic()-start:.1f}s)", flush=True)
    k = max(scores, key=scores.get)
    final_rows = np.concatenate([np.flatnonzero(train), np.flatnonzero(validation)])
    result = fit_xd(features.x[final_rows], features.cov[final_rows],
                    weights=weights[final_rows],
                    observed=features.observed[final_rows], init=candidates[k], seed=seed,
                    labels=features.labels, max_iter=cfg["max_iter"], tol=cfg["tol"],
                    regularization=cfg["regularization"])
    record = dict(name=name, selected_k=k, selection_scores={str(k):v for k,v in scores.items()},
                  n_fit=len(final_rows), n_selection=int(validation.sum()),
                  weighted_n_fit=float(weights[final_rows].sum()),
                  n_iter=result.n_iter, converged=result.converged,
                  convergence_tail=result.history[-5:],
                  mean_training_log_density=result.mean_loglike,
                  band_counts=features.observed[final_rows].sum(axis=0).tolist(),
                  elapsed_s=time.monotonic()-start, mixture=result.mixture.to_dict())
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(record));tmp.replace(path)
    print(f"{name}: saved K={k}, n={len(final_rows)}, converged={result.converged}, {record['elapsed_s']:.1f}s", flush=True)
    return result.mixture, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/multisurvey_lsw.json"))
    ap.add_argument("--z-step", type=float, default=None,
                    help="defaults to the sample configuration's redshift step")
    ap.add_argument("--max-background-fit", type=int, default=20000)
    ap.add_argument("--part", choices=("background", "all"), default="all")
    args = ap.parse_args(); cfg=json.loads(args.config.read_text());root=Path(cfg["data_dir"])
    if args.z_step is None:
        args.z_step=cfg["z_step"]
    if json.loads((root/"config.json").read_text()) != cfg:
        raise ValueError("training configuration differs from the cached sample selection")
    b=dict(np.load(root/"background.npz"));bands=tuple(b["bands"])
    bp=Photometry(b["flux"],b["variance"],bands)
    bfit = ~b["held"] & (bp.observed.sum(1)>=cfg["min_bands"])
    rng=np.random.default_rng(cfg["seed"]+10)
    choose,bweights=sample_patterns(bp.observed,bfit,args.max_background_fit,cfg["min_band_training"],rng)
    bfit[:]=False;bfit[choose]=True
    train_fields=np.unique(b["field"][bfit])
    for _ in range(10000):
        sel_fields=rng.choice(train_fields,max(1,round(.2*len(train_fields))),replace=False)
        btrain=bfit & ~np.isin(b["field"],sel_fields)
        if (bp.observed[btrain].sum(0)>=cfg["min_band_training"]).all():
            break
    else:
        raise ValueError("cannot reserve internal selection fields while retaining every band")
    bselect=bfit & ~btrain
    softening=[]
    for j, label in enumerate(bands):
        errors=np.sqrt(bp.variance[btrain & bp.observed[:,j],j])
        if len(errors)<cfg["min_band_training"]:
            raise ValueError(f"{label}: only {len(errors)} training measurements")
        softening.append(float(np.median(errors)))
    transform=BandLuptitudeTransform(bands,np.array(softening))
    bf=transform(bp)
    priority=reference_order(bands)
    settings=dict(config=json.loads(json.dumps(cfg)),z_step=args.z_step,max_background_fit=args.max_background_fit,
                  transform=transform.to_dict(),
                  background_sha256=hashlib.sha256((root/"background.npz").read_bytes()).hexdigest(),
                  targets_sha256=hashlib.sha256((root/"targets.npz").read_bytes()).hexdigest())
    if args.part=="all":
        settings["quasars_sha256"]=hashlib.sha256((root/"quasars.npz").read_bytes()).hexdigest()
    run_id=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    fits=root/f"fits_{run_id}";fits.mkdir(exist_ok=True)
    (fits/"settings.json").write_text(json.dumps(settings,indent=2))
    cfg["_global_train"]=btrain
    cfg["_weights"]=bweights
    # The background does not depend on the quasar redshift grid. Reuse an
    # identical completed fit when only that grid was changed.
    background_path=fits/"background.json"
    if not background_path.exists():
        background_independent={"z_step","quasars_sha256"}
        expected={k:v for k,v in settings.items() if k not in background_independent}
        for old in root.glob("fits_*/background.json"):
            previous=json.loads((old.parent/"settings.json").read_text())
            if {k:v for k,v in previous.items() if k not in background_independent}==expected:
                background_path.write_bytes(old.read_bytes())
                break
    background,brecord=fit_and_select("background",bf,btrain,bselect,cfg,
                                     background_path,priority,cfg["seed"])
    bounds=np.array([[np.min(bf.x[bfit & bf.observed[:,j],j]),
                      np.max(bf.x[bfit & bf.observed[:,j],j])] for j in range(len(bands))])
    if args.part=="background":
        return
    # Release the large field covariance before fitting the quasar sample.
    del bf
    q=dict(np.load(root/"quasars.npz"))
    if tuple(q["bands"]) != bands:
        raise ValueError("quasar/background layouts disagree")
    qp=Photometry(q["flux"],q["variance"],bands)
    qfit=~q["held"] & (qp.observed.sum(1)>=cfg["min_bands"])
    qf=transform(qp)
    rng=np.random.default_rng(cfg["seed"])
    selection_blocks=rng.choice(np.unique(q["group"][qfit]),
                                max(1,round(.2*len(np.unique(q["group"][qfit])))),replace=False)
    qselect=qfit & np.isin(q["group"],selection_blocks)
    qtrain=qfit & ~qselect
    cfg["_global_train"]=qtrain
    cfg["_weights"]=np.ones(qf.n_obs)
    edges=redshift_edges(cfg["z_min"],cfg["z_max"],args.z_step)
    centres=(edges[1:]+edges[:-1])/2
    mixtures,records,counts=[],[],[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        # Same fractional overlap as the established sliced colour model.
        selected=(q["zspec"]>=lo-.5*(hi-lo)) & (q["zspec"]<hi+.5*(hi-lo))
        tr,va=qtrain&selected,qselect&selected
        if tr.sum()<2 or va.sum()<2:
            raise ValueError(f"slice {j} lacks independent selection objects")
        mix,record=fit_and_select(f"z={centres[j]:.3f}",qf,tr,va,cfg,
                                  fits/f"qso_{j:02d}.json",priority,cfg["seed"]+j)
        mixtures.append(mix);records.append({k:v for k,v in record.items() if k!='mixture'})
        counts.append(record["n_fit"])
    qso=SlicedColourRedshiftModel(centres,mixtures,np.array(counts),f"multisurvey_{run_id}",bands,
        meta=dict(per_slice=records,heldout_blocks=sorted(np.unique(q["group"][q["held"]]).tolist()),
                  selection_blocks=selection_blocks.tolist(),n_fit=int(qfit.sum()),
                  n_holdout=int((q["held"] & (qp.observed.sum(1)>=cfg["min_bands"])).sum())))
    model=MultiSurveyModel(qso,background,transform,priority,bounds,
        meta=dict(run_id=run_id,settings=settings,background={k:v for k,v in brecord.items() if k!='mixture'},
                  background_selection_fields=sel_fields.tolist(),
                  background_sampling="observed-band-mask strata, inverse-probability weights",
                  background_holdout_fields=np.unique(b["field"][b["held"]]).tolist(),
                  fields=json.loads((root/"fields.json").read_text()),
                  magnitude_systems="native: optical AB; ALLWISE and VHS Vega",
                  reddening="observed photometry, no dereddening; high Galactic latitude training",
                  posterior="requires a matched reference-band and transform-specific prior pair"))
    model.save(cfg["model_path"])
    print(f"Saved {cfg['model_path']}",flush=True)


if __name__=="__main__":
    main()
