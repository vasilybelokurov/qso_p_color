#!/usr/bin/env python
"""Compare saved models on identical, spatially held-out Legacy DR9 grz data.

The input is a cached spectroscopic pair catalogue, used here as a source of
individual classified objects. This is not a physical-pair calibration test.
No fitting or database access occurs. Model/config/data hashes key score caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import spearmanr

from build_multisurvey_sample import deduplicate, xyz
from validate_multisurvey import auc
from qso_pcolor.background import BackgroundColourModel, galactic_healpix
from qso_pcolor.data import _save_npz, galactic_from_equatorial
from qso_pcolor.features import RelativeFluxTransform, deredden
from qso_pcolor.multisurvey import MultiSurveyModel, _ConditionalQSO, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.plotting import save_figure
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel
from qso_pcolor.score import score_candidates


def sha256(path: str) -> str:
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def redshift_diagnostics(model, features, grid, redshift, alternative):
    """Return likelihoods and flat-prior redshift summaries (densities per z).

    The grid contains all slice centres and both support edges. Since the model
    interpolates linearly in density, trapezoidal normalisation is exact.
    """
    slices = model._log_p_slices(features.x, features.cov, features.observed)
    lp = model.log_p_colour_given_z(features.x, features.cov, grid, _log_slices=slices)
    peak = lp.max(axis=1)
    density = np.exp(lp - peak[:, None])
    norm = np.trapezoid(density, grid, axis=1)
    cdf = cumulative_trapezoid(density, grid, axis=1, initial=0) / norm[:, None]
    at = model.log_p_colour_given_z(features.x, features.cov, redshift, _log_slices=slices).diagonal()
    alt = model.log_p_colour_given_z(features.x, features.cov, alternative, _log_slices=slices).diagonal()
    return dict(log_like=at, log_pz=at-peak-np.log(norm),
                alternative_log_pz=alt-peak-np.log(norm),
                z_mode=grid[np.argmax(lp, axis=1)],
                true_cdf=np.array([np.interp(z, grid, c) for z, c in zip(redshift, cdf)]))


def select_sample(cfg, old, background, new):
    raw = dict(np.load(cfg['source']))
    ra, dec = raw['comp_ra'], raw['comp_dec']
    l, b = galactic_from_equatorial(ra, dec)
    group = galactic_healpix(l, b, old.meta['holdout_nside'])
    # The new manifest lists populated held-out blocks; one original block has
    # no high-latitude training targets. Restrict to their recorded intersection.
    common_blocks = sorted(set(old.meta['holdout_blocks']) & set(new.qso.meta['heldout_blocks']))
    if not common_blocks:
        raise ValueError('models have no common recorded holdout blocks')
    flux, ivar, transmission = [np.column_stack([raw[f'comp_{prefix}_{band}'] for band in ('g','r','z')])
                                 for prefix in ('flux','flux_ivar','mw_transmission')]
    corrected, variance = deredden(flux, ivar, transmission)
    with np.errstate(invalid='ignore', divide='ignore'):
        mag = 22.5 - 2.5*np.log10(corrected[:, 1])
        snr = corrected[:, 1]/np.sqrt(variance[:, 1])
    lo, hi = cfg['reference_magnitude_range']
    sep_lo, sep_hi = cfg['separation_arcsec_range']
    kind = np.char.strip(raw['comp_spectype'].astype(str))
    isq = kind == 'QSO'
    ok = (np.isin(group, common_blocks) & (abs(b) >= cfg['min_abs_b_deg'])
          & (raw['comp_release'] == cfg['release']) & (raw['comp_maskbits'] == 0)
          & (raw['sep_arcsec'] >= sep_lo) & (raw['sep_arcsec'] <= sep_hi)
          & np.isfinite(raw['comp_fracflux_r']) & (raw['comp_fracflux_r'] <= cfg['max_fracflux_r'])
          & (snr >= cfg['min_reference_snr']) & (mag >= lo) & (mag < hi)
          & np.all(np.isfinite(corrected) & np.isfinite(variance) & (variance > 0), axis=1)
          & np.isin(kind, ['QSO','STAR','GALAXY'])
          & (~isq | (old.in_support(raw['comp_zspec']) & new.qso.in_support(raw['comp_zspec']))))
    # Exclude entire background fitting/selection cones, not just sampled rows.
    positions = xyz(ra, dec)
    cones = [(c, background.meta['radius_deg']) for c in background.meta['cones']]
    cones += [(c, new.meta['settings']['config']['field_radius_deg']) for c in new.meta['fields'] if not c['held']]
    for cone, radius in cones:
        centre = xyz([cone['ra']], [cone['dec']])[0]
        ok &= np.sum((positions-centre)**2, axis=1) > (2*np.sin(np.deg2rad(radius)/2))**2
    ids = np.flatnonzero(ok)
    # Remove repeated pair rows cheaply before positional de-duplication.
    _, unique = np.unique(raw['comp_targetid'][ids], return_index=True)
    ids = ids[np.sort(unique)]
    ids = ids[deduplicate(ra[ids], dec[ids], cfg['dedup_radius_arcsec'])]
    if cfg.get('exclude_score_cache'):
        with np.load(cfg['exclude_score_cache']) as previous:
            ids = ids[~np.isin(raw['comp_targetid'][ids], previous['targetid'])]
    rng = np.random.default_rng(cfg['seed'])
    counts = {k: int(np.sum(kind[ids] == k)) for k in ['QSO','STAR','GALAXY']}
    chosen = []
    for use_qso in [True, False]:
        pool = ids[isq[ids] == use_qso]
        chosen.extend(rng.choice(pool, min(len(pool), cfg['max_per_class']), replace=False))
    chosen = np.array(chosen)
    selected_qso = isq[chosen]
    z = raw['comp_zspec'][chosen].copy()
    zpool = raw['comp_zspec'][ids[isq[ids]]]
    z[~selected_qso] = rng.choice(zpool, np.sum(~selected_qso))
    alternative = rng.choice(zpool, len(chosen))
    for _ in range(1000):
        inside = abs(alternative-z) <= cfg['half_width_kms']/299792.458*(1+alternative)
        if not inside.any():
            break
        alternative[inside] = rng.choice(zpool, inside.sum())
    else:
        raise ValueError('cannot draw alternative redshifts outside matching window')
    data = dict(row=chosen, targetid=raw['comp_targetid'][chosen], ra=ra[chosen], dec=dec[chosen],
                l=l[chosen], b=b[chosen], group=group[chosen], kind=kind[chosen], z=z,
                alternative_z=alternative, flux=flux[chosen], ivar=ivar[chosen], transmission=transmission[chosen])
    return data, counts


def score_sample(cfg, data, old, background, new):
    n = len(data['z'])
    flux, var = deredden(data['flux'], data['ivar'], data['transmission'])
    old_features = RelativeFluxTransform(min_ref_snr=cfg['min_reference_snr'])(
        np.column_stack([flux, np.full((n,2), np.nan)]),
        np.column_stack([var, np.full((n,2), np.inf)]), ('g','r','z','w1','w2'))
    phot = Photometry(data['flux'], 1/data['ivar'], tuple(f'decals_dr9_south:{b}' for b in ('g','r','z')))
    new_features = new.transform(phot.align(new.transform.bands))
    anchor = new_features.labels.index('decals_dr9_south:r')
    conditional_qso = _ConditionalQSO(new.qso, anchor)
    low = max(old.support[0], new.qso.support[0]); high = min(old.support[1], new.qso.support[1])
    grid = np.unique(np.concatenate([np.linspace(low, high, cfg['redshift_grid_points']), old.z_centres, new.qso.z_centres]))
    grid = grid[(grid >= low) & (grid <= high)]
    out = {}
    for name, model, features in [('old', old, old_features), ('new', conditional_qso, new_features)]:
        parts = []
        for start in range(0, n, cfg['batch_size']):
            rows = np.arange(start, min(n, start+cfg['batch_size']))
            f = features.subset(rows)
            res = redshift_diagnostics(model, f, grid, data['z'][rows], data['alternative_z'][rows])
            if name == 'old':
                pb = background.log_prob(f.x, f.cov, f.ref_mag, data['l'][rows], data['b'][rows], observed=f.observed)
            else:
                pb = new.background_log_prob(f.x, f.cov, f.observed, anchor)
            res['log_bf'] = res.pop('log_like') - pb
            parts.append(res)
        out.update({f'{name}_{k}': np.concatenate([p[k] for p in parts]) for k in parts[0]})
        print(f'Scored {name}: {n} identical objects', flush=True)
    # Independent public-API cross-check, on both spectroscopic classes.
    indices = np.array([0, 1, np.flatnonzero(data['kind'] != 'QSO')[0]])
    common = dict(z_primary=data['z'][indices], l_deg=data['l'][indices], b_deg=data['b'][indices],
                  match=RedshiftMatch(half_width_kms=cfg['half_width_kms']), z_grid=grid)
    checks = [score_candidates(old_features.subset(indices), qso_model=old, background_model=background, min_bands=2, **common),
              new.score(phot.subset(indices), min_bands=3, **common)]
    for name, scores in zip(['old','new'], checks):
        np.testing.assert_allclose([s.log_bayes_factor_qz_bkg for s in scores], out[f'{name}_log_bf'][indices], atol=1e-9)
        assert all(s.status == 'no_prior_posterior_unavailable' and np.isfinite(s.p_zmatch_given_qso) for s in scores)
    return {**data, **out}


def metrics(data, rows):
    q = data['kind'][rows] == 'QSO'; qi = rows[q]; bi = rows[~q]
    result = {}
    for name in ['old','new']:
        bf, pz, alt = [data[f'{name}_{k}'] for k in ['log_bf','log_pz','alternative_log_pz']]
        cdf = data[f'{name}_true_cdf'][qi]
        result[name] = dict(auc_qso_nonqso=auc(bf[qi], bf[bi]),
                            auc_true_other_z=auc(pz[qi], alt[qi]),
                            coverage_68=float(np.mean((cdf >= .16) & (cdf <= .84))),
                            median_abs_dz_over_1pz=float(np.median(abs(data[f'{name}_z_mode'][qi]-data['z'][qi])/(1+data['z'][qi]))))
        for kind in ['STAR','GALAXY']:
            result[name][f'auc_qso_{kind.lower()}'] = auc(bf[qi], bf[rows[data['kind'][rows] == kind]])
    return result


def report(cfg, identity, data, counts):
    rows = np.arange(len(data['z'])); q = data['kind'] == 'QSO'
    values = metrics(data, rows)
    groups = np.unique(data['group']); rng = np.random.default_rng(cfg['seed']+1)
    boot = []
    for _ in range(cfg['bootstrap_replicates']):
        sampled = np.concatenate([rows[data['group'] == g] for g in rng.choice(groups, len(groups))])
        m = metrics(data, sampled)
        if all(m[n]['auc_qso_nonqso'] is not None and m[n]['auc_true_other_z'] is not None for n in ['old','new']):
            boot.append([m['new'][k]-m['old'][k] for k in ['auc_qso_nonqso','auc_true_other_z']])
    differences = {}
    for j, key in enumerate(['auc_qso_nonqso','auc_true_other_z']):
        interval = np.quantile(np.array(boot)[:,j], [.025,.975])
        differences[key] = dict(new_minus_old=values['new'][key]-values['old'][key], paired_block_bootstrap_95_interval=interval.tolist(),
                                within_equivalence_margin=bool(np.all(abs(interval) <= cfg['auc_equivalence_margin'])))
    agreement = {}
    for field, selected in [('log_bf', rows), ('log_pz', rows[q])]:
        old, new = [data[f'{name}_{field}'][selected] for name in ['old','new']]
        agreement[field] = dict(spearman=float(spearmanr(old,new).statistic), median_new_minus_old=float(np.median(new-old)),
                               median_absolute_difference=float(np.median(abs(new-old))),
                               difference_16_84=np.quantile(new-old,[.16,.84]).tolist())
    result = dict(config=cfg, hashes=identity, available=counts,
                  evaluated={k:int(np.sum(data['kind']==k)) for k in ['QSO','STAR','GALAXY']},
                  heldout_blocks=groups.tolist(), public_scorer_checks=6, metrics=values,
                  auc_differences=differences, score_agreement=agreement,
                  interpretation='Identical southern Legacy DR9 grz measurements; original dereddened relative fluxes versus new observed luptitudes. WISE excluded because Legacy forced WISE and ALLWISE are different measurements. Spectroscopically selected individual objects, not random imaging sources or a close-pair calibration. No matched new population prior; compare dimensionless Bayes factors and flat-prior redshift summaries only.')
    report_path = Path(cfg.get('report_prefix', 'docs/MODEL_COMPARISON'))
    report_path.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    labels = dict(auc_qso_nonqso='Quasar/non-quasar AUC', auc_true_other_z='True/other redshift AUC',
                  coverage_68='Central 68% redshift interval coverage', median_abs_dz_over_1pz='Median absolute dz/(1+z)',
                  auc_qso_star='Quasar/star AUC', auc_qso_galaxy='Quasar/galaxy AUC')
    lines = ['# Original versus seven-survey model: identical Legacy DR9 data', '', result['interpretation'], '',
             f"Evaluated {result['evaluated']['QSO']:,} quasars, {result['evaluated']['STAR']:,} stars, and {result['evaluated']['GALAXY']:,} galaxies in {len(groups)} common reserved sky blocks. Only blocks recorded as held out by both quasar models are admitted; every background fitting/selection cone is excluded. Objects are positionally de-duplicated; the full JSON report records all cuts, seeds, and model paths.", '',
             '| Metric | Original | Seven-survey |', '|---|---:|---:|']
    for key in values['old']:
        lines.append(f"| {labels[key]} | {values['old'][key]:.4f} | {values['new'][key]:.4f} |")
    lines += ['', 'Paired 95% intervals resample whole sky blocks, retaining each object’s old/new scores together. The configured equivalence margin is ±0.02 AUC, declared before scoring; the entire interval must lie inside it.', '']
    for key, v in differences.items():
        ci = v['paired_block_bootstrap_95_interval']
        lines.append(f"- {labels[key]}: new minus old {v['new_minus_old']:+.4f}, interval [{ci[0]:+.4f}, {ci[1]:+.4f}]; equivalence check {'passes' if v['within_equivalence_margin'] else 'does not pass'}.")
    lines += ['', 'Individual score agreement:', '']
    for key, v in agreement.items():
        lines.append(f"- {key}: Spearman {v['spearman']:.3f}; median new minus old {v['median_new_minus_old']:+.3f} nats; median absolute difference {v['median_absolute_difference']:.3f} nats.")
    lines += ['', 'Near-equal AUC, when present, does not establish interchangeable individual scores. This comparison does not refit either model. The original model remains available.', '',
              'The old model uses g/r and z/r after marginalising its missing WISE dimensions; the new model conditions the three observed luptitudes on r. Raw likelihood densities have different units and are not compared. The same true and assigned primary redshifts enter both calculations. QSO/non-QSO AUC scores quasars at their true redshifts and non-quasars at redshifts drawn from the held-out quasar population; true/other-z AUC uses each quasar at its true and one drawn redshift outside ±2000 km/s.', '',
              'All g,r,z measurements must be usable, with maskbits=0, r S/N≥5, 17≤dereddened r<22.5, |b|≥25°, release=9010, 3–30 arcsec catalogue separation, and finite fracflux_r≤0.2. Negative non-reference fluxes are retained. This defines a matched diagnostic sample, not the prevalence of classes in the sky.', '',
              f"Reproduce with `python scripts/compare_old_new_models.py --config CONFIG`, using the configuration recorded in the [full results and hashes]({report_path.name}.json). The source catalogue is local. The hash-keyed score cache records source rows, target IDs, coordinates, assigned redshifts, and both model outputs. [Other survey combinations](MULTISURVEY_VALIDATION.md)."]
    if cfg.get('exclude_score_cache'):
        lines += ['', f"This confirmation sample excludes every target in `{cfg['exclude_score_cache']}` before the random draw. It uses previously unscored objects in the common spatial holdout; the sky blocks are shared with the diagnostic sample, so this is an object-level confirmation, not a new-footprint test."]
    report_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4.2))
    for ax, field, selected, label in [(axes[0], 'log_bf', rows, f'Colour log Bayes factor ({len(rows):,} objects)'),
                                        (axes[1], 'log_pz', rows[q], f'Log redshift density at true z ({q.sum():,} QSOs)')]:
        old, new = [data[f'{name}_{field}'][selected] for name in ['old','new']]
        ax.scatter(old,new,s=5,alpha=.18,rasterized=True)
        lim = [min(old.min(),new.min()),max(old.max(),new.max())]
        ax.plot(lim,lim,'k--',lw=1)
        ax.set(xlabel='Original model', ylabel='Seven-survey model', title=label)
    fig.suptitle('Identical held-out objects and Legacy DR9 g, r, z measurements')
    fig.tight_layout(); save_figure(fig,cfg.get('figure_name','validation/old_new_grz')); plt.close(fig)
    print(json.dumps({k:result[k] for k in ['available','evaluated','metrics','auc_differences','score_agreement']},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/model_comparison.json'))
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    identity = {key: sha256(cfg[key]) for key in ['source','old_qso','old_background','new_model']}
    identity['script'] = sha256(__file__)
    identity['package'] = {p.name:sha256(str(p)) for p in sorted(Path('src/qso_pcolor').glob('*.py'))}
    if cfg.get('exclude_score_cache'):
        identity['excluded_sample'] = sha256(cfg['exclude_score_cache'])
    signature = json.dumps(dict(config=cfg, hashes=identity), sort_keys=True)
    cache = Path(cfg['score_cache'])
    if cache.exists():
        with np.load(cache) as saved:
            valid = str(saved['signature']) == signature
            data = {k:saved[k] for k in saved.files if k not in ['signature','available']} if valid else None
            counts = json.loads(str(saved['available'])) if valid else None
    else:
        data = None
    if data is None:
        old = SlicedColourRedshiftModel.load(cfg['old_qso'])
        background = BackgroundColourModel.load(cfg['old_background'])
        new = MultiSurveyModel.load(cfg['new_model'])
        sample, counts = select_sample(cfg,old,background,new)
        print('Eligible after all cuts and positional deduplication:', counts, flush=True)
        data = score_sample(cfg,sample,old,background,new)
        _save_npz(cache, **data, signature=np.array(signature), available=np.array(json.dumps(counts)))
    report(cfg,identity,data,counts)


if __name__ == '__main__':
    main()
