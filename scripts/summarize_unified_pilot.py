#!/usr/bin/env python
"""Read saved pilot predictions, isolate guard effects, and make diagnostic plots."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

from qso_pcolor.full_sample import write_json
from qso_pcolor.plotting import save_figure
from complete_unified_pilot import get_root


def main():
    root=get_root();report=json.loads((root/'report.json').read_text());diagnostic={}
    for hemi in ('south','north'):
        samples={kind:{mode:dict(np.load(root/f'all_{hemi}_{kind}_{mode}.npz')) for mode in ('parent','raw','unified')} for kind in ('qso','stars')}
        entry={}
        for mode in ('parent','raw'):
            ranks=[]
            for kind in ('qso','stars'):
                p=samples[kind][mode];q=np.logaddexp(p['log_lambda_sameq'],p['log_lambda_fieldq'])
                rank=p['log_lambda_sameq']-logsumexp(np.stack([q,p['log_lambda_bkg'],p['log_lambda_out']]),axis=0)-np.log(p['dz_match_eff'])
                ranks.append(rank)
            entry[mode+'_diagnostic_auc_before_guards']=float(mannwhitneyu(*ranks).statistic/(len(ranks[0])*len(ranks[1])))
        raw=samples['qso']['raw'];new=samples['qso']['unified'];outside=raw['status']=='outside_both_models'
        entry['qso_rejected_by_fixed_distance']=int(outside.sum())
        entry['of_those_accepted_by_calibrated_support']=int((outside&~new['support_rejected']).sum())
        entry['qso_support_percentiles_on_fixed_distance_rejections']=new['support'][outside].tolist()
        diagnostic[hemi]=entry
    diagnostic['interpretation']='Read-only counterfactual from saved intensity terms, not a change of deployment gates. Much of the pilot full-band ranking loss is caused by fixed component-distance rejection; some northern density-ranking loss remains. The percentile cut also loses more test QSOs than its calibration target.'
    write_json(root/'guard_diagnostic.json',diagnostic);write_json(Path('docs/UNIFIED_PILOT_GUARDS_2026-09-30.json'),diagnostic)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    modes=['parent','raw','unified'];titles=['Existing full-data model','Pilot: existing guard','Pilot: both guards']
    for j,hemi in enumerate(('south','north')):
        item=report['real']['all_'+hemi]
        axes[0].bar(np.arange(3)+j*.36,[item['auc'][m] for m in modes],width=.34,label=hemi)
        axes[1].bar(np.arange(3)+j*.36,[item['qso'][m]['eligible']/item['qso'][m]['n'] for m in modes],width=.34,label=hemi)
    for ax in axes:
        ax.set_xticks(np.arange(3)+.18,titles,rotation=15,ha='right');ax.set_ylim(.8,1.01);ax.legend();ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Ranking AUC (rejected objects placed last)');axes[1].set_ylabel('Fraction of test QSOs eligible for ranking')
    fig.suptitle('Full-band pilot: integration works, rejection policy still needs adjustment\n64 QSOs and 64 PSF-background objects per hemisphere')
    save_figure(fig,'unified_pilot/ranking_and_retention');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    for ax,hemi in zip(axes,('south','north')):
        for kind,color in [('stars','#bb7134'),('qso','#3668ad')]:
            d=np.load(root/f'all_{hemi}_{kind}_unified.npz');ax.scatter(d['support'],d['p_quasar'],s=18,alpha=.6,label='PSF background' if kind=='stars' else 'Known QSOs',color=color)
        ax.axvline(report['support_policy']['threshold'],ls='--',color='black',lw=1);ax.set(xscale='log',xlabel='QSO-support percentile at target redshift',ylabel='Total-QSO diagnostic score',title=hemi.capitalize(),ylim=(-.03,1.03));ax.legend()
    fig.suptitle('Relative QSO score and absolute model support are separate diagnostics')
    save_figure(fig,'unified_pilot/score_and_support');plt.close(fig)
    nodes=np.linspace(-6,8,15);fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained');cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d9dde0')
    for ax,mode,title in zip(axes,('parent','raw','unified'),titles):
        d=np.load(root/f'grid_north_18.5_{mode}.npz');v=np.ma.array(d['p_quasar'],mask=~d['eligible']).reshape(15,15)
        im=ax.imshow(v,origin='lower',extent=[-6.5,8.5]*2,vmin=0,vmax=1,cmap=cmap);ax.set(title=title,xlabel='g − r [asinh mag]',ylabel='r − z [asinh mag]')
    fig.colorbar(im,ax=axes,label='Total-QSO diagnostic score',shrink=.8);fig.suptitle('Original northern bright tail; grey = rejected\nSynthetic stress test, not a contamination-rate measurement')
    save_figure(fig,'unified_pilot/northern_tail');plt.close(fig)
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':main()
