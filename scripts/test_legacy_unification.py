#!/usr/bin/env python
"""Compare published/overlap-fitted photometric corrections on cached pairs."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from qso_pcolor.full_sample import write_json, file_hash
from qso_pcolor.legacy_homogenization import affine_photometry, cubic_photometry
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.plotting import save_figure


def xyz(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)))


def paired_rows(cfg, out):
    """Unique reciprocal matches; preserve frozen cone roles and raw identity."""
    roles = {r['cone']: r['role'] for r in json.loads(Path(cfg['spatial_roles']).read_text())['regions']}
    target = np.load(cfg['qso_targets'])
    qt = cKDTree(xyz(target['ra'], target['dec']))
    collected, inventory = [], []
    seen = set()
    for p in sorted(Path(cfg['raw_stellar_root']).glob('*.npz')):
        d = np.load(p); release = d['release']
        north = np.flatnonzero(release == 9011); south = np.flatnonzero(np.isin(release, [9010, 9012]))
        if not len(north) or not len(south): continue
        cid = int(p.name.split('_')[1]); r = np.deg2rad(cfg['match_arcsec']/3600)
        xn, xs = xyz(d['ra'][north], d['dec'][north]), xyz(d['ra'][south], d['dec'][south])
        nt, st = cKDTree(xn), cKDTree(xs)
        dist, j = st.query(xn, distance_upper_bound=r)
        cn = st.query_ball_point(xn, r, return_length=True)
        cs = nt.query_ball_point(xs, r, return_length=True)
        keep = np.flatnonzero(np.isfinite(dist) & (cn == 1))
        keep = keep[cs[j[keep]] == 1]
        nn, ss = north[keep], south[j[keep]]
        unique = []
        for n, s in zip(nn, ss):
            key = tuple(int(d[k][n]) for k in ('release', 'brickid', 'objid'))
            if key in seen: unique.append(False)
            else: seen.add(key); unique.append(True)
        nn, ss = nn[unique], ss[unique]
        blend = (d['fracflux_r'][nn] < cfg['max_fracflux_r']) & (d['fracflux_r'][ss] < cfg['max_fracflux_r'])
        nn, ss = nn[blend], ss[blend]
        bands = ('g','r','z','w1','w2')
        f, v, observed = [], [], []
        for rows in (nn, ss):
            flux = np.column_stack([d['flux_'+b][rows] for b in bands])
            ivar = np.column_stack([d['flux_ivar_'+b][rows] for b in bands])
            nobs = np.column_stack([d['nobs_'+b][rows] for b in bands])
            obs = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (nobs > 0)
            var = np.divide(1., ivar, out=np.full_like(ivar, np.inf, dtype=float), where=obs)
            f.append(flux); v.append(var); observed.append(obs)
        dq, iq = qt.query(xyz(d['ra'][nn], d['dec'][nn]), distance_upper_bound=np.deg2rad(cfg['qso_match_arcsec']/3600))
        z = np.full(len(nn), np.nan); matched = np.isfinite(dq); z[matched] = target['zspec'][iq[matched]]
        collected.append(dict(flux=np.stack(f, axis=1), variance=np.stack(v, axis=1), observed=np.stack(observed, axis=1),
             zspec=z, cone=np.full(len(nn), cid), role=np.full(len(nn), roles[cid]),
             north_row=nn, south_row=ss, ra=d['ra'][nn], dec=d['dec'][nn]))
        inventory.append(dict(path=str(p),sha256=file_hash(p),cone=cid,role=roles[cid],pairs=len(nn),qso=int(matched.sum())))
    data = {k:np.concatenate([d[k] for d in collected]) for k in collected[0]}
    np.savez_compressed(out/'overlap_pairs.npz', **data)
    return data, inventory


def robust_poly(x, y, cfg):
    """Fit unweighted clipped least squares; clip residuals only in calibration."""
    design = np.polynomial.polynomial.polyvander(x, cfg['cubic_degree'])
    coeff = np.zeros((design.shape[1], y.shape[1])); kept = []
    for b in range(y.shape[1]):
        good = np.ones(len(x), bool)
        for _ in range(cfg['robust_iterations']):
            coeff[:, b] = np.linalg.lstsq(design[good], y[good, b], rcond=None)[0]
            residual = y[:, b] - design @ coeff[:, b]
            centre = np.median(residual[good]); scale = 1.4826*np.median(abs(residual[good]-centre))
            good = abs(residual-centre) <= cfg['robust_sigma']*max(scale, np.finfo(float).eps)
        coeff[:, b] = np.linalg.lstsq(design[good], y[good, b], rcond=None)[0]
        kept.append(int(good.sum()))
    return coeff, kept


def stats(residual):
    if not len(residual): return dict(n=0)
    median = np.median(residual, axis=0)
    return dict(n=len(residual), median=median.tolist(), nmad=(1.4826*np.median(abs(residual-median),axis=0)).tolist(),
                median_absolute=np.median(abs(residual),axis=0).tolist(), p95_absolute=np.percentile(abs(residual),95,axis=0).tolist())


def main():
    cfg = json.loads(Path('configs/legacy_unification_test.json').read_text())
    out = Path(cfg['output']); out.mkdir(parents=True,exist_ok=True)
    data, inventory = paired_rows(cfg,out)
    model = MultiSurveyModel.load(Path(cfg['parent_candidate'])/'model.json')
    bandnames = ('g','r','z','w1','w2')
    ix = [[model.transform.bands.index(f'decals_dr9_{h}:{b}') for b in bandnames] for h in ('north','south')]
    soft = model.transform.softening[ix]
    f, v, obs = data['flux'], data['variance'], data['observed']
    factor = 2.5/np.log(10)
    u = 22.5-factor*(np.arcsinh(f/(2*soft))+np.log(soft))
    error = factor*np.sqrt(v)/np.hypot(f,2*soft)
    snr = np.divide(f,np.sqrt(v),out=np.zeros_like(f),where=np.isfinite(v))
    optical = obs[:,:,:3].all(axis=(1,2))
    bright = optical & (snr[:,:,:3]>cfg['fit_min_snr']).all(axis=(1,2))
    qso = np.isfinite(data['zspec']); colour = u[:,0,0]-u[:,0,2]
    fit = bright & ~qso & np.isin(data['role'],cfg['fit_roles']) & (colour>=cfg['cubic_colour_range'][0]) & (colour<=cfg['cubic_colour_range'][1])
    coeff, kept = robust_poly(colour[fit],(u[:,0,:3]-u[:,1,:3])[fit],cfg)
    # Complete optical vectors only here; missing bands remain in the saved cache.
    x = u[optical,0,:3]; cv = np.zeros((len(x),3,3)); cv[:,np.arange(3),np.arange(3)] = error[optical,0,:3]**2
    corrected, cov = affine_photometry(x,cv,np.array(cfg['desi_matrix']),np.array(cfg['desi_offset']))
    cubic, jac = cubic_photometry(x,coeff,tuple(cfg['cubic_colour_range']))
    pred = {k:np.full((len(u),3),np.nan) for k in ('uncorrected','desi','overlap_cubic')}
    pred['uncorrected'][optical]=x; pred['desi'][optical]=corrected;pred['overlap_cubic'][optical]=cubic
    hold = np.isin(data['role'],cfg['evaluation_roles'])
    selections = dict(heldout_bright_stars=hold&bright&~qso,heldout_bright_qso=hold&bright&qso,
                      heldout_all_snr_stars=hold&optical&~qso,heldout_all_snr_qso=hold&optical&qso,
                      all_bright_qso=bright&qso)
    report = dict(config=cfg, inventory=inventory, n_pairs=len(u), n_qso=int(qso.sum()),
          source_hashes={k:file_hash(Path(cfg[k])) for k in ('spatial_roles','qso_targets')},
          fit=dict(n=int(fit.sum()),cones=np.unique(data['cone'][fit]).tolist(),retained_per_band=kept,
                   cubic_coefficients=coeff.tolist()), evaluations={}, wise={}, figures=[],
          interpretation='Residual = predicted southern minus measured southern native luptitude. DESI is exact in ordinary magnitudes; applying its affine form to luptitudes is an experimental faint-flux extension. Empirical cubic uses fresh DR9 overlap coefficients, not published DR8 coefficients.',
          active_model_changed=False)
    for sel, use in selections.items():
        report['evaluations'][sel] = {k:stats((value-u[:,1,:3])[use]) for k,value in pred.items()}
        report['evaluations'][sel]['cones']=np.unique(data['cone'][use]).tolist()
    for j,b in enumerate(bandnames[3:],3):
        use=hold & obs[:,:,j].all(axis=1) & (snr[:,:,j]>cfg['fit_min_snr']).all(axis=1)
        # Ordinary magnitudes isolate catalogue flux differences from different softenings.
        delta=-2.5*np.log10(f[use,0,j]/f[use,1,j])
        report['wise'][b]=stats(delta[:,None])
    # DESI implementation check in the positive-flux domain.
    ff=f[bright,0,:3]; mag=22.5-2.5*np.log10(ff)
    gm=ff[:,0]*10**(-.4*.004)*(ff[:,0]/ff[:,1])**(-.059)
    rm=ff[:,1]*10**(.4*.003)*(ff[:,1]/ff[:,2])**(-.024)
    zm=ff[:,2]*10**(.4*.013)*(ff[:,1]/ff[:,2])**(.015)
    reproduced=mag@np.array(cfg['desi_matrix']).T+np.array(cfg['desi_offset'])
    assert np.allclose(reproduced,22.5-2.5*np.log10(np.column_stack([gm,rm,zm])),atol=1e-12)
    # Also evaluate the original positive-flux DESI relation without the
    # luptitude extension, on the SAME high-S/N held-out rows.
    bright_rows=np.flatnonzero(bright)
    south_mag=22.5-2.5*np.log10(f[bright,1,:3])
    report['desi_exact_magnitude']={}
    for pop,use in [('stars',~qso),('qso',qso)]:
        take=(hold&use)[bright_rows]
        report['desi_exact_magnitude'][pop]=dict(uncorrected=stats((mag-south_mag)[take]),desi=stats((reproduced-south_mag)[take]))
    # Diagnostic native linear observation map, estimated from the same fit rows.
    a=np.array(cfg['desi_matrix']); inv=np.linalg.solve(a,np.eye(3))
    north_pred=(u[fit,1,:3]-cfg['desi_offset'])@inv.T
    rr=u[fit,0,:3]-north_pred
    centre=np.median(rr,axis=0); scale=1.4826*np.median(abs(rr-centre),axis=0)
    good=(abs(rr-centre)<cfg['robust_sigma']*scale).all(axis=1)
    # This total paired residual includes measurement noise; do not call it intrinsic.
    residual_cov=np.cov(rr[good],rowvar=False)
    report['diagnostic_forward_map']=dict(north_from_south=inv.tolist(),offset=(-inv@np.array(cfg['desi_offset'])).tolist(),
          residual_covariance=residual_cov.tolist(),n=int(good.sum()),note='Total paired residual covariance; conservative diagnostic scatter, not a deconvolved calibration.')
    colors={'uncorrected':'#777777','desi':'#246db4','overlap_cubic':'#d55e00'}
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex='row',sharey='row',layout='constrained')
    binned={};rng=np.random.default_rng(cfg['seed'])
    for row,sel in enumerate(('heldout_bright_stars','heldout_bright_qso')):
        use=selections[sel]; edges=np.array(cfg['colour_edges'] if row==0 else cfg['redshift_edges']); xx=colour if row==0 else data['zspec']
        binned[sel]={}
        for name,value in pred.items():
            rr=value-u[:,1,:3]; med=np.full((len(edges)-1,3),np.nan);counts=[]
            limits=np.full((len(edges)-1,2,3),np.nan)
            for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                ok=use&(xx>=lo)&(xx<hi);counts.append(int(ok.sum()))
                if ok.sum()>=cfg['minimum_plot_bin']:
                    med[j]=np.median(rr[ok],axis=0)
                    # Resample whole sky fields: include field-to-field offsets.
                    fields=np.unique(data['cone'][ok]);bs=[]
                    for _ in range(cfg['bootstrap_repetitions']):
                        selected=rng.choice(fields,len(fields),replace=True)
                        sample=np.concatenate([rr[ok&(data['cone']==c)] for c in selected])
                        bs.append(np.median(sample,axis=0))
                    limits[j]=np.percentile(bs,[2.5,97.5],axis=0)
            serial=lambda a: [[float(z) if np.isfinite(z) else None for z in r] for r in a]
            binned[sel][name]=dict(count=counts,median=serial(med),field_bootstrap_lower=serial(limits[:,0]),field_bootstrap_upper=serial(limits[:,1]))
            for b,ax in enumerate(axes[row]):
                centres=(edges[1:]+edges[:-1])/2
                ax.plot(centres,med[:,b],'.-',label=name.replace('_',' '),color=colors[name])
                ax.fill_between(centres,limits[:,0,b],limits[:,1,b],color=colors[name],alpha=.1)
        for b,ax in enumerate(axes[row]):
            ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.set_title(f'{bandnames[b]} | {"stellar background" if row==0 else "known QSOs"}: N={use.sum():,}')
            ax.set_xlabel('Northern g − z [mag]' if row==0 else 'Spectroscopic redshift')
        axes[row,0].set_ylabel('Median corrected North − South [mag]',labelpad=10)
    axes[0,0].legend(fontsize=9);fig.suptitle('Same objects, two instruments: corrections tested on reserved sky fields\nS/N > 10 in all grz bands; shaded: 95% field-bootstrap ranges (few independent fields)',fontsize=13)
    report['figures'].append(str(save_figure(fig,'legacy_unification/overlap_residuals')))
    report['binned']=binned
    np.savez_compressed(out/'corrections.npz',luptitudes=u,**pred,fit=fit,bright=bright,heldout=hold,qso=qso)
    write_json(out/'photometry_report.json',report);write_json(Path('docs/LEGACY_UNIFICATION_PHOTOMETRY_2026-09-30.json'),report)
    print(json.dumps({k:report[k] for k in ('n_pairs','n_qso','fit','evaluations','wise')},indent=2),flush=True)


if __name__=='__main__':main()
