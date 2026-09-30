#!/usr/bin/env python
"""Read-only paired density audit of saved small/full-data models; never fit.

Report log predictive densities in nats per observed non-reference luptitude
volume, with identical measurements, masks and reference bands for both models.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from qso_pcolor.multisurvey import MultiSurveyModel, conditional_log_prob, _ConditionalQSO
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.background import galactic_healpix
from qso_pcolor.full_sample import file_hash, write_json

OLD = Path('models/multisurvey_psf/630f47f63b6f0694/model.json')
NEW = Path('models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b/model.json')
CONT = Path('models/multisurvey_psf/work/convergence_continuation/20260930')
INPUT = Path('models/multisurvey_psf/work/full_training_inputs/370b9a1b027c3de8')
OUT = Path('models/multisurvey_psf/work/paired_health/20260930')
REPORT = Path('docs/FULL_VS_SMALL_HEALTH_2026-09-30.json')
SEED = 20260930
MAX_ROWS = 30000  # Evaluation sample only. No training/data-acquisition cap.


def xyz(ra, dec):
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)))


def paired_qso(model, x, cov, obs, anchor, z):
    """Evaluate each row at its own z, using the scorer's density interpolation."""
    centres = model.z_centres
    left = np.clip(np.searchsorted(centres, z)-1, 0, len(centres)-2)
    fraction = np.clip((z-centres[left])/(centres[left+1]-centres[left]), 0, 1)
    values = np.empty((len(z), 2))
    for side in (0, 1):
        for j in np.unique(left+side):
            use = left+side == j
            values[use, side] = conditional_log_prob(model.mixtures[j], x[use], cov[use], obs[use], anchor)
    with np.errstate(divide='ignore'):
        return np.logaddexp(values[:, 0]+np.log1p(-fraction), values[:, 1]+np.log(fraction))


def summarize(old, new, groups):
    ok = np.isfinite(old) & np.isfinite(new)
    d = new[ok]-old[ok]
    if not len(d):
        return dict(n=0)
    ids = np.unique(groups[ok])
    sums = np.array([d[groups[ok] == g].sum() for g in ids])
    counts = np.array([(groups[ok] == g).sum() for g in ids])
    draws = np.random.default_rng(SEED).integers(0, len(ids), (1000, len(ids)))
    boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return dict(n=int(len(d)), nonfinite_pairs=int((~ok).sum()), old_mean=float(old[ok].mean()),
                new_mean=float(new[ok].mean()), mean_gain=float(d.mean()), median_gain=float(np.median(d)),
                gain_p05_p95=np.percentile(d, [5, 95]).tolist(), fraction_improved=float((d>0).mean()),
                fraction_loss_over_1_nat=float((d < -1).mean()),
                sky_groups=int(len(ids)), sky_bootstrap_mean_gain_95=np.percentile(boot, [2.5, 97.5]).tolist())


def health(model):
    mixes = [*model.qso.mixtures, model.background]
    return dict(qso_converged=int(sum(r['converged'] for r in model.qso.meta['per_slice'])),
                slices=len(model.qso.mixtures), stellar_components=len(model.background.weights),
                finite_parameters=bool(all(np.isfinite(m.means).all() and np.isfinite(m.covs).all()
                                           and np.isfinite(m.weights).all() for m in mixes)),
                positive_weights=bool(all((m.weights>0).all() for m in mixes)),
                max_weight_sum_error=float(max(abs(m.weights.sum()-1) for m in mixes)),
                minimum_covariance_eigenvalue=float(min(np.linalg.eigvalsh(m.covs).min() for m in mixes)))


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    old, new = MultiSurveyModel.load(OLD), MultiSurveyModel.load(NEW)
    assert old.transform_id == new.transform_id
    assert old.reference_priority == new.reference_priority
    assert np.array_equal(old.qso.z_centres, new.qso.z_centres)
    cont_q = type(new.qso).from_dict(new.qso.to_dict())
    cont_q.mixtures = [GaussianMixture.from_dict(json.loads((CONT/f'qso_{j:02d}.json').read_text())['mixture']) for j in range(43)]
    cont_b = GaussianMixture.from_dict(json.loads((CONT/'background.json').read_text())['mixture'])
    bands = old.transform.bands
    order = np.array([bands.index(b) for b in old.reference_priority])
    modes = dict(all_available=np.ones(len(bands), bool),
                 optical=np.array([b.split(':')[0] in ('sdss','ps1') or
                                    (b.startswith('decals_') and b.split(':')[1] in ('g','r','z')) for b in bands]),
                 legacy_optical=np.array([b.startswith('decals_') and b.split(':')[1] in ('g','r','z') for b in bands]))
    report = dict(seed=SEED, evaluation_cap_per_population=MAX_ROWS,
                  inputs=str(INPUT), models={str(p):file_hash(p) for p in (OLD, NEW)},
                  active_small=health(old), full_300=health(new),
                  training_rows=dict(small_qso=55994, full_qso=1049260, small_stars=20000, full_stars=1975894),
                  interpretation='Paired held-out conditional predictive densities, not calibrated probabilities or a ranking validation.',
                  continuation='Secondary global-shape comparison only; continued shapes have no refreshed spatial weights.', populations={})
    fields = json.loads(Path('data/multisurvey_lsw/fields.json').read_text())
    # Exclude all original field cones, including reserved ones: conservative protection
    # against old initialization/shape/spatial training overlap.
    old_centres = cKDTree(xyz([f['ra'] for f in fields], [f['dec'] for f in fields]))
    for kind in ('qso', 'stars'):
        data = {p.stem:np.load(p, mmap_mode='r', allow_pickle=False) for p in (INPUT/kind).glob('*.npy')}
        eligible = (data['role']==3) & data['eligible']
        initial = int(eligible.sum())
        if kind == 'qso':
            cells = galactic_healpix(data['l'], data['b'], 4)
            eligible &= np.isin(cells, old.qso.meta['heldout_blocks']) & old.qso.in_support(data['zspec'])
            policy = 'New test role intersected with original QSO held-out nside=4 Galactic cells and trained redshift support.'
        else:
            distance, _ = old_centres.query(xyz(data['ra'], data['dec']))
            eligible &= distance > 2*np.sin(np.deg2rad(.3)/2)
            cells = data['field']
            policy = 'New test regions, excluding every original stellar cone within 0.3 degree (original radius 0.15 degree).'
        candidates = np.flatnonzero(eligible)
        rows = np.sort(np.random.default_rng(SEED).choice(candidates, min(MAX_ROWS, len(candidates)), replace=False))
        result = dict(initial_test_rows=initial, independent_pool=len(candidates), evaluated=len(rows), selection=policy, modes={})
        save = dict(rows=rows, ra=data['ra'][rows], dec=data['dec'][rows], groups=cells[rows])
        print(kind, 'independent pool', len(candidates), 'evaluation', len(rows), flush=True)
        for mode, mask in modes.items():
            values = np.full((len(rows), 5), np.nan)
            nband = np.zeros(len(rows), int); refmag = np.full(len(rows), np.nan)
            for lo in range(0, len(rows), 512):
                rr = rows[lo:lo+512]
                variance = np.array(data['variance'][rr]); variance[:, ~mask] = np.inf
                feat = old.transform(Photometry(data['flux'][rr], variance, bands))
                obs = feat.observed; anchors = order[np.argmax(obs[:, order], axis=1)]
                nband[lo:lo+len(rr)] = obs.sum(axis=1)
                for a in np.unique(anchors):
                    ix = np.flatnonzero((anchors == a) & (obs.sum(axis=1)>=2))
                    if not len(ix): continue
                    dest = lo+ix; x,c,o = feat.x[ix],feat.cov[ix],obs[ix]
                    refmag[dest] = x[:, a]
                    if kind == 'qso':
                        z = data['zspec'][rr[ix]]
                        for col, model in enumerate((old.qso, new.qso, cont_q)):
                            values[dest, col] = paired_qso(model,x,c,o,int(a),z)
                        if lo == 0:
                            check = _ConditionalQSO(new.qso,int(a)).log_p_colour_given_z(x[:4],c[:4],z[:4],observed=o[:4]).diagonal()
                            np.testing.assert_allclose(values[dest[:4],1],check,rtol=1e-12,atol=1e-12)
                    else:
                        for col, model in enumerate((old,new)):
                            values[dest,col] = model.background_log_prob(x,c,o,int(a),l_deg=data['l'][rr[ix]],b_deg=data['b'][rr[ix]])
                            values[dest,col+3] = conditional_log_prob(model.background,x,c,o,int(a))
                        values[dest,2] = conditional_log_prob(cont_b,x,c,o,int(a))
            use = nband>=2; groups=cells[rows]
            primary = summarize(values[use,0],values[use,1],groups[use])
            primary['rows_without_colour_information'] = int((~use).sum())
            primary['strata'] = {}
            # Each stratum compares identical rows and coordinates within that stratum.
            for label, sel in [('north_dec_ge_32',data['dec'][rows]>=32),('south_dec_lt_32',data['dec'][rows]<32),
                               ('abs_b_lt_40',abs(data['b'][rows])<40),('abs_b_ge_40',abs(data['b'][rows])>=40),
                               ('reference_luptitude_lt_20',refmag<20),('reference_luptitude_ge_20',refmag>=20)]:
                s=use & sel; primary['strata'][label]=summarize(values[s,0],values[s,1],groups[s])
            if kind=='qso':
                for lower,upper in [(0,1),(1,2),(2,3),(3,5)]:
                    s=use & (data['zspec'][rows]>=lower) & (data['zspec'][rows]<upper)
                    primary['strata'][f'z_{lower}_{upper}']=summarize(values[s,0],values[s,1],groups[s])
            base = 1 if kind=='qso' else 4
            primary['continuation_vs_300_global'] = summarize(values[use,base],values[use,2],groups[use])
            if kind=='stars':
                primary['full_vs_small_global'] = summarize(values[use,3],values[use,4],groups[use])
            save[mode+'_densities']=values; save[mode+'_nbands']=nband
            result['modes'][mode]=primary
            print(kind, mode, json.dumps({k:v for k,v in primary.items() if k not in ('strata',)}), flush=True)
        np.savez_compressed(OUT/f'{kind}.npz', **save)
        report['populations'][kind]=result
        write_json(REPORT,report)
    report['elapsed_seconds']=time.monotonic()-started
    report['saved_predictions']=str(OUT)
    write_json(REPORT,report)
    print('Saved', REPORT, 'seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
