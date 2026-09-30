#!/usr/bin/env python
"""Evaluate the pooled grz experiment in native units with fixed diagnostic priors."""
from __future__ import annotations

from copy import deepcopy
import argparse
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as units
from scipy.stats import mannwhitneyu

from qso_pcolor import PSFMultiSurveyBaseline
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.joint_spatial import JointSpatialWeights, fit_joint_spatial_log_prob
from qso_pcolor.multisurvey import BandLuptitudeTransform, MultiSurveyModel, conditional_log_prob
from qso_pcolor.projected_xd import native_mixture
from qso_pcolor.qso_model import SlicedColourRedshiftModel
from qso_pcolor.spatial import component_log_prob

from run_pooled_optical_prototype import load_arrays, diagonal, operators_for
from validate_full_sample_release import run_scores, metrics, auc
from validate_psf_catchalls import grid_data


def read_fit(root, kind, j, mode):
    record = json.loads((root/'fits'/f'{kind}_{j:02d}_{mode}.json').read_text())
    return GaussianMixture.from_dict(record['mixture'])


def native_fit(root, calibration, kind, j, mode, hemisphere, labels=()):
    mix = read_fit(root, kind, j, 'pooled' if mode == 'pooled' else hemisphere)
    s = int(hemisphere == 'north')
    return native_mixture(mix, *operators_for(calibration, kind, mode)[s], labels=labels)


def spatial_weights(root, cfg, calibration, mode, hemisphere=None):
    name = mode if mode == 'pooled' else hemisphere
    path = root/f'spatial_{name}.json'
    if path.exists(): return JointSpatialWeights.from_dict(json.loads(path.read_text()))
    data = load_arrays(root/'stars'); use = np.isin(data['role'], cfg['fit_roles'])
    if mode != 'pooled': use &= data['system'] == int(hemisphere == 'north')
    rows = np.flatnonzero(use)
    mix = read_fit(root, 'stars', 0, name)
    lp = np.lib.format.open_memmap(root/f'spatial_logp_{name}.npy', mode='w+', dtype='f8', shape=(len(rows), mix.n_components))
    natives = {int(s): native_fit(root, calibration, 'stars', 0, mode, ('south', 'north')[int(s)]) for s in np.unique(data['system'][rows])}
    for start in range(0, len(rows), cfg['batch_size']):
        rr = rows[start:start+cfg['batch_size']]; block = np.empty((len(rr), mix.n_components))
        for s in np.unique(data['system'][rr]):
            ok = data['system'][rr] == s; ii = rr[ok]
            block[ok] = component_log_prob(natives[int(s)], data['y'][ii], diagonal(data['variance'][ii]), data['observed'][ii])
        lp[start:start+len(rr)] = block
    lp.flush()
    result = fit_joint_spatial_log_prob(mix, lp, data['l'][rows], data['b'][rows], np.ones(len(rows)),
        **cfg['spatial'], meta=dict(prototype=True, roles=cfg['fit_roles'], mode=name))
    write_json(path, result.to_dict())
    print('SPATIAL COMPLETE', name, len(rows), flush=True)
    return result


def make_bundle(root, cfg, parent, calibration, mode, hemisphere, spatial):
    """Save an explicitly diagnostic native-view bundle for the public scorer."""
    directory = root/'bundles'/f'{mode}_{hemisphere}'
    if (directory/'manifest.json').exists(): return PSFMultiSurveyBaseline.load(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = tuple(f'decals_dr9_{hemisphere}:{b}' for b in 'grz')
    indices = np.array([parent.model.transform.bands.index(b) for b in labels])
    runid = f'prototype_{root.name}_{mode}_{hemisphere}'
    qso = SlicedColourRedshiftModel(parent.model.qso.z_centres,
        [native_fit(root, calibration, 'qso', j, mode, hemisphere, labels) for j in range(len(parent.model.qso.mixtures))],
        np.array([json.loads((root/'fits'/f'qso_{j:02d}_{"pooled" if mode == "pooled" else hemisphere}.json').read_text())['n'] for j in range(len(parent.model.qso.mixtures))]),
        runid, labels, dict(prototype=True))
    model = MultiSurveyModel(qso, native_fit(root, calibration, 'stars', 0, mode, hemisphere, labels),
        BandLuptitudeTransform(labels, parent.model.transform.softening[indices]),
        tuple(b for b in parent.model.reference_priority if b in labels), parent.model.background_bounds[indices],
        meta=dict(population='psf', run_id=runid, diagnostic_only=True), spatial_background=spatial)
    model.save(directory/'model.json')
    priors = json.loads((Path(cfg['parent'])/'priors.json').read_text())
    priors['anchors'] = {k:v for k,v in priors['anchors'].items() if k in labels}
    def rebind(value):
        if isinstance(value, dict):
            if 'transform_id' in value: value['transform_id'] = model.transform_id
            if 'model_run_id' in value:
                value['inherited_model_run_id'] = value['model_run_id']; value['model_run_id'] = runid
                value['diagnostic_inherited_calibration'] = True
            for v in value.values(): rebind(v)
        elif isinstance(value, list):
            for v in value: rebind(v)
    rebind(priors); write_json(directory/'priors.json', priors)
    outlier = deepcopy(parent.outlier)
    outlier.mean = outlier.mean[indices]; outlier.cov = outlier.cov[np.ix_(indices, indices)]
    outlier.labels = labels; outlier.transform_id = model.transform_id
    outlier.fractions = {k:v for k,v in outlier.fractions.items() if k in labels or k == '*'}
    outlier.meta.update(model_run_id=runid, inherited_model_run_id=parent.model.meta['run_id'], diagnostic_inherited_calibration=True)
    outlier.__post_init__(); outlier.save(directory/'outlier.json')
    manifest = dict(parent.manifest, bundle_id=runid, diagnostic_only=True,
        scope='Optical prototype; inherited native priors and catch-all for controlled comparison, not recalibrated probabilities.',
        files={p: file_hash(directory/p) for p in ('model.json', 'priors.json', 'outlier.json')})
    write_json(directory/'manifest.json', manifest)
    return PSFMultiSurveyBaseline.load(directory)


def summary(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return dict(n=len(a), mean=float(np.mean(a)) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p05=float(np.percentile(a, 5)) if len(a) else None,
        p95=float(np.percentile(a, 95)) if len(a) else None)


def predict_all(root, cfg, vcfg, bundles, kinds=('qso', 'stars')):
    report = {}
    for kind in kinds:
        data = load_arrays(root/kind)
        for s, hemi in enumerate(('south', 'north')):
            rows = np.flatnonzero((data['role'] == cfg['test_role']) & (data['system'] == s))
            records = dict(rows=rows, reference=np.zeros(len(rows)), z=np.asarray(data['zspec'][rows]),
                n_observed=data['observed'][rows].sum(axis=1), negative=np.asarray(data['negative'][rows]))
            for mode in ('separate', 'pooled'):
                base = bundles[(mode, hemi)]; model = base.model
                path = root/f'predictive_{kind}_{hemi}_{mode}.npz'
                if path.exists():
                    d = dict(np.load(path)); records[mode] = d['logp']; records['reference'] = d['reference']; continue
                values = np.empty(len(rows)); mag = np.empty(len(rows))
                for start in range(0, len(rows), cfg['batch_size']):
                    rr = rows[start:start+cfg['batch_size']]
                    x, cov, obs = data['y'][rr], diagonal(data['variance'][rr]), data['observed'][rr]
                    order = np.array([model.transform.bands.index(b) for b in model.reference_priority])
                    anchors = order[np.argmax(obs[:, order], axis=1)]
                    idx = np.argmin(abs(data['zspec'][rr, None]-model.qso.z_centres), axis=1) if kind == 'qso' else np.zeros(len(rr), int)
                    lp = np.empty(len(rr))
                    for j in np.unique(idx):
                        for a in np.unique(anchors[idx == j]):
                            ok = (idx == j) & (anchors == a)
                            if kind == 'qso': lp[ok] = conditional_log_prob(model.qso.mixtures[j], x[ok], cov[ok], obs[ok], int(a))
                            else: lp[ok] = model.background_log_prob(x[ok], cov[ok], obs[ok], int(a), l_deg=data['l'][rr[ok]], b_deg=data['b'][rr[ok]])
                    values[start:start+len(rr)] = lp; mag[start:start+len(rr)] = x[np.arange(len(rr)), anchors]
                np.savez_compressed(path, rows=rows, logp=values, reference=mag)
                records[mode] = values; records['reference'] = mag
            delta = records['pooled']-records['separate']; informative = records['n_observed'] >= 2
            entry = dict(rows=len(rows), informative_rows=int(informative.sum()), delta_logp=summary(delta[informative]),
                negative_flux_delta=summary(delta[informative & records['negative']]), magnitude=[], redshift=[])
            for lo, hi in zip(vcfg['magnitude_edges'][:-1], vcfg['magnitude_edges'][1:]):
                use = informative & (records['reference'] >= lo) & (records['reference'] < hi)
                entry['magnitude'].append(dict(lo=lo, hi=hi, **summary(delta[use])))
            if kind == 'qso':
                zc = bundles[('pooled', hemi)].model.qso.z_centres
                nearest = np.argmin(abs(records['z'][:, None]-zc), axis=1)
                for j, z in enumerate(zc): entry['redshift'].append(dict(z=float(z), **summary(delta[informative & (nearest == j)])))
            report[f'{kind}_{hemi}'] = entry
            write_json(root/'predictive_report.json', report)
            print('PREDICTIVE', kind, hemi, entry['delta_logp'], flush=True)
    return report


def score_data(base, data, rows, kind, cfg, zprimary):
    """Reconstruct native flux exactly from saved asinh observations for public scoring."""
    x, v, obs = data['y'][rows], data['variance'][rows], data['observed'][rows]
    factor = 2.5/np.log(10); soft = base.model.transform.softening
    flux = 2*soft*np.sinh((22.5-x)/factor-np.log(soft))
    variance = v*(flux*flux+4*soft*soft)/factor**2
    return dict(flux=np.where(obs, flux, np.nan), variance=np.where(obs, variance, np.inf),
        bands=base.model.transform.bands, l=data['l'][rows], b=data['b'][rows], zprimary=zprimary)


def score_checks(root, cfg, vcfg, rcfg, bundles):
    rng = np.random.default_rng(cfg['seed']); report = dict(real={}, subsets={}, paired={}, grids={})
    selected_path = root/'score_rows.npz'
    arrays = {k: load_arrays(root/k) for k in ('qso', 'stars')}
    latitude_min = bundles[('pooled', 'south')].manifest['min_abs_b_deg']
    if selected_path.exists(): selected = dict(np.load(selected_path))
    else:
        selected = {}
        for kind, data in arrays.items():
            for s, hemi in enumerate(('south', 'north')):
                available = np.flatnonzero((data['role'] == cfg['test_role']) & (data['system'] == s) & (data['observed'].sum(axis=1) >= 2) & (abs(data['b']) >= latitude_min))
                selected[f'{kind}_{hemi}'] = rng.choice(available, min(len(available), cfg['score_rows_per_population_per_hemisphere']), replace=False)
        np.savez_compressed(selected_path, **selected)
    predictions = {}
    for hemi in ('south', 'north'):
        zprimary = arrays['qso']['zspec'][selected[f'qso_{hemi}']]
        for mode in ('separate', 'pooled'):
            base = bundles[(mode, hemi)]; entry = {}
            for kind, data in arrays.items():
                rows = selected[f'{kind}_{hemi}']; d = score_data(base, data, rows, kind, cfg, zprimary[:len(rows)])
                p = run_scores(base, d, rcfg, root/f'scores_{mode}_{hemi}_{kind}.npz')
                predictions[(mode, hemi, kind)] = p; entry[kind] = metrics(p, rcfg)
                nsub = min(vcfg['subset_rows_per_population_per_hemisphere'], len(rows))
                for bits in itertools.product((False, True), repeat=3):
                    if not any(bits): continue
                    name = ''.join(b for b, present in zip('grz', bits) if present)
                    sub = {k:(v[:nsub].copy() if k != 'bands' else v) for k,v in d.items()}
                    sub['variance'][:, ~np.array(bits)] = np.inf
                    keep = np.isfinite(sub['variance']).any(axis=1)
                    sub = {k:(v[keep] if k != 'bands' else v) for k,v in sub.items()}
                    psub = run_scores(base, sub, rcfg, root/f'subset_{mode}_{hemi}_{kind}_{name}.npz')
                    report['subsets'][f'{mode}_{hemi}_{kind}_{name}'] = metrics(psub, rcfg)
            entry['auc'] = auc(predictions[(mode, hemi, 'qso')], predictions[(mode, hemi, 'stars')])
            q, b = predictions[(mode, hemi, 'qso')], predictions[(mode, hemi, 'stars')]
            qs = np.where(q['eligible'], q['log_r_per_unit_z'], -np.inf)
            bs = np.where(b['eligible'], b['log_r_per_unit_z'], -np.inf)
            entry['qso_completeness'] = {}
            for rate in vcfg['background_acceptance_rates']:
                threshold = np.sort(bs)[min(len(bs)-1, int(np.ceil((1-rate)*len(bs)))-1)]
                entry['qso_completeness'][str(rate)] = dict(qso_fraction=float(np.mean(qs > threshold)), background_fraction=float(np.mean(bs > threshold)))
            report['real'][f'{mode}_{hemi}'] = entry
            print('SCORES', mode, hemi, 'AUC', entry['auc'], flush=True)
            write_json(root/'score_report.json', report)
    old_release = json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())
    old_cache = Path(old_release['cache'])
    for hemi in ('south', 'north'):
        for mag in rcfg['grid']['reference_magnitudes']:
            low = np.ones(rcfg['grid']['grid_size']**2, bool)
            for old in ('baseline', 'candidate'):
                saved = np.load(old_cache/f'grid_{hemi}_{mag}_{old}.npz')
                q = np.logaddexp(saved['log_lambda_sameq'], saved['log_lambda_fieldq']); b = saved['log_lambda_bkg']
                low &= (q < np.nanmax(q)+np.log(vcfg['grid_low_density_fraction'])) & (b < np.nanmax(b)+np.log(vcfg['grid_low_density_fraction']))
            for mode in ('separate', 'pooled'):
                base = bundles[(mode, hemi)]
                d = grid_data(base.model, hemi, mag, {'validation': rcfg['grid']})
                p = run_scores(base, d, rcfg, root/f'grid_{mode}_{hemi}_{mag}.npz')
                report['grids'][f'{mode}_{hemi}_{mag}'] = dict(**metrics(p, rcfg),
                    common_low_density=int(low.sum()), high_qso_low_density=int((low & p['eligible'] & (p['p_quasar'] > rcfg['high_qso_probability'])).sum()))
                old_check = json.loads(Path('configs/legacy_unification_test.json').read_text())
                if mag == old_check['probe_reference_magnitude']:
                    nodes = np.linspace(*rcfg['grid']['grid_colour_range'], rcfg['grid']['grid_size'])
                    gx, gy = np.meshgrid(nodes, nodes)
                    probe = np.flatnonzero(np.isclose(gx.ravel(), old_check['probe_colours'][0]) & np.isclose(gy.ravel(), old_check['probe_colours'][1]))[0]
                    report['grids'][f'{mode}_{hemi}_{mag}']['probe'] = dict(p_quasar=float(p['p_quasar'][probe]), eligible=bool(p['eligible'][probe]))
                print('GRID', mode, hemi, mag, report['grids'][f'{mode}_{hemi}_{mag}']['high_qso_low_density'], flush=True)
    overlap = np.load(cfg['overlap']); held = np.isin(overlap['role'], ['calib', 'test']) & overlap['observed'][:, :, :3].any(axis=2).all(axis=1)
    all_sky = SkyCoord(overlap['ra']*units.deg, overlap['dec']*units.deg).galactic
    held &= abs(all_sky.b.deg) >= latitude_min
    q = np.flatnonzero(held & np.isfinite(overlap['zspec'])); b = np.flatnonzero(held & ~np.isfinite(overlap['zspec']))
    b = rng.choice(b, min(len(b), vcfg['paired_background_rows']), replace=False)
    pairs = np.r_[q, b]; sky = SkyCoord(overlap['ra'][pairs]*units.deg, overlap['dec'][pairs]*units.deg).galactic
    np.save(root/'paired_rows.npy', pairs)
    paired_predictions = {}
    for mode in ('separate', 'pooled'):
        pair_scores = {}
        for s, hemi in enumerate(('north', 'south')):
            d = dict(flux=overlap['flux'][pairs, s, :3], variance=overlap['variance'][pairs, s, :3],
                bands=bundles[(mode, hemi)].model.transform.bands, l=sky.l.deg, b=sky.b.deg,
                zprimary=np.where(np.isfinite(overlap['zspec'][pairs]), overlap['zspec'][pairs], vcfg['paired_primary_redshift']))
            pair_scores[hemi] = run_scores(bundles[(mode, hemi)], d, rcfg, root/f'paired_{mode}_{hemi}.npz')
        n, s = pair_scores['north'], pair_scores['south']; use = n['eligible'] & s['eligible']
        paired_predictions[mode] = pair_scores
        report['paired'][mode] = dict(n=len(pairs), both_eligible=int(use.sum()),
            score_difference=summary((n['p_quasar']-s['p_quasar'])[use]),
            absolute_score_difference=summary(abs(n['p_quasar']-s['p_quasar'])[use]),
            evidence_difference=summary((n['log_bayes_factor_qz_bkg']-s['log_bayes_factor_qz_bkg'])[use]))
    common = np.ones(len(pairs), bool)
    for views in paired_predictions.values():
        for p in views.values(): common &= p['eligible']
    for mode, views in paired_predictions.items():
        report['paired'][mode]['common_eligible_absolute_score_difference'] = summary(abs(views['north']['p_quasar']-views['south']['p_quasar'])[common])
    write_json(root/'score_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--qso-only', action='store_true'); parser.add_argument('--root', type=Path); args = parser.parse_args()
    cfg = json.loads(Path('configs/pooled_optical_prototype.json').read_text())
    vcfg = json.loads(Path('configs/pooled_optical_validation.json').read_text())
    rcfg = json.loads(Path('configs/full_sample_release.json').read_text()); rcfg['batch_size'] = cfg['score_batch_size']
    root = args.root or Path(json.loads((Path(cfg['output'])/'current.json').read_text())['directory'])
    if (root/'effective_config.json').exists(): cfg = json.loads((root/'effective_config.json').read_text())
    parent = PSFMultiSurveyBaseline.load(cfg['parent']); calibration = json.loads((root/'calibration.json').read_text())
    if args.qso_only:
        views = {}
        for mode in ('separate', 'pooled'):
            for hemi in ('south', 'north'):
                labels = tuple(f'decals_dr9_{hemi}:{b}' for b in 'grz')
                mixtures = [native_fit(root, calibration, 'qso', j, mode, hemi, labels) for j in range(len(parent.model.qso.mixtures))]
                model = SimpleNamespace(qso=SimpleNamespace(z_centres=parent.model.qso.z_centres, mixtures=mixtures),
                    transform=SimpleNamespace(bands=labels), reference_priority=tuple(b for b in parent.model.reference_priority if b in labels))
                views[(mode, hemi)] = SimpleNamespace(model=model)
        predict_all(root, cfg, vcfg, views, kinds=('qso',))
        return
    if not (root/'training_complete.json').exists(): raise RuntimeError('Wait for prototype density fits to finish')
    bundles = {}
    for mode in ('separate', 'pooled'):
        for hemi in ('south', 'north'):
            spatial = spatial_weights(root, cfg, calibration, mode, hemi)
            bundles[(mode, hemi)] = make_bundle(root, cfg, parent, calibration, mode, hemi, spatial)
    predictive = predict_all(root, cfg, vcfg, bundles)
    scores = score_checks(root, cfg, vcfg, rcfg, bundles)
    fits = [json.loads(p.read_text()) for p in sorted((root/'fits').glob('*.json'))]
    checks = {}
    for k, v in predictive.items(): checks['predictive_'+k] = v['delta_logp']['mean'] >= -cfg['maximum_mean_log_density_loss']
    for hemi in ('south', 'north'):
        old, new = scores['real']['separate_'+hemi], scores['real']['pooled_'+hemi]
        checks['auc_'+hemi] = new['auc'] >= old['auc']-cfg['maximum_auc_loss']
        checks['qso_eligibility_'+hemi] = new['qso']['eligible']/new['qso']['n'] >= old['qso']['eligible']/old['qso']['n']-cfg['maximum_qso_eligibility_loss']
        checks['background_high_rate_'+hemi] = new['stars']['qso_above_half']/new['stars']['n'] <= old['stars']['qso_above_half']/old['stars']['n']+cfg['maximum_high_background_rate_increase']
        for rate in vcfg['background_acceptance_rates']:
            checks[f'qso_completeness_{hemi}_{rate}'] = new['qso_completeness'][str(rate)]['qso_fraction'] >= old['qso_completeness'][str(rate)]['qso_fraction']-vcfg['maximum_qso_completeness_loss']
    checks['northern_tail'] = all(scores['grids'][f'pooled_north_{m}']['high_qso_low_density'] == 0 for m in rcfg['grid']['reference_magnitudes'][:2])
    checks['subset_numerics'] = all(all(v[k] for k in ('finite_eligible', 'probabilities_bounded', 'window_identity', 'hard_guard')) for v in scores['subsets'].values())
    checks['paired_consistency'] = scores['paired']['pooled']['common_eligible_absolute_score_difference']['mean'] <= scores['paired']['separate']['common_eligible_absolute_score_difference']['mean']+vcfg['maximum_pair_absolute_score_increase']
    report = dict(root=str(root), config=cfg, validation_config=vcfg, calibration=calibration,
        counts=json.loads((root/'prepared.json').read_text())['counts'], predictive=predictive, scores=scores, checks=checks,
        checks_pass=all(checks.values()), training=dict(fits=len(fits), converged=sum(f['converged'] for f in fits),
            iteration_limit=sum(not f['converged'] for f in fits), maximum_likelihood_decline=min(min(np.diff(f['history']), default=0.) for f in fits)),
        active_model_changed=False, probability_calibration=False,
        limitations=['Three optical bands only', 'Native population priors/catch-all inherited for comparison', 'Fixed affine approximation in asinh space', 'Raw QSO excess scatter may include variability', 'Previously inspected examples are regression tests, not untouched data'],
        implementation={str(p):file_hash(p) for p in (Path(__file__), Path('configs/pooled_optical_validation.json'))})
    write_json(root/'report.json', report)
    write_json(Path('docs/POOLED_OPTICAL_PROTOTYPE_2026-09-30.json'), report)
    print('DONE', json.dumps(checks), flush=True)


if __name__ == '__main__': main()
