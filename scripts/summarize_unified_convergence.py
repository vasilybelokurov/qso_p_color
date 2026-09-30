#!/usr/bin/env python
"""Summarize saved historical convergence and bounded unified experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qso_pcolor.full_sample import write_json
from qso_pcolor.plotting import save_figure


def main():
    parent=Path('models/multisurvey_psf/work/full_training_fits/45aa8f6cdb34802b')
    audit=json.loads(Path('docs/CONVERGENCE_AUDIT_2026-09-30.json').read_text())['fits']
    items=[]
    for i in range(43):
        name=f'qso_{i:02d}';d=json.loads((parent/(name+'.json')).read_text())
        status='converged' if d['converged'] else audit[name]['action']
        items.append(dict(name=name,z=round(.15+.1*i,2),rows=d['n_fit'],k=d['selected_k'],iterations=d['n_iter'],status=status))
    groups={}
    for status in ('converged','continue','hold_for_diagnosis'):
        subset=[d for d in items if d['status']==status]
        groups[status]=dict(fits=len(subset),fraction=len(subset)/43,z=[d['z'] for d in subset],
            training_slice_memberships=sum(d['rows'] for d in subset),
            min_rows=min(d['rows'] for d in subset),max_rows=max(d['rows'] for d in subset))
    historical=dict(qso_fits=43,capped=18,capped_fraction=18/43,groups=groups,
        membership_note='Overlapping slices reuse objects; membership counts are not unique-object counts',fits=items)
    write_json(Path('docs/CONVERGENCE_FREQUENCY_2026-09-30.json'),historical)
    colors={'converged':'#718795','continue':'#c57d27','hold_for_diagnosis':'#ae4555'}
    labels={'converged':'Met stopping rule','continue':'Limit: still improving','hold_for_diagnosis':'Limit: declining'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for status in groups:
        subset=[d for d in items if d['status']==status]
        for ax,key in zip(axes,['iterations','rows']):
            ax.scatter([d['z'] for d in subset],[d[key] for d in subset],s=40,color=colors[status],label=labels[status])
    axes[0].axhline(300,color='black',ls='--',lw=.7);axes[0].set_ylabel('Iterations');axes[0].legend(fontsize=8)
    axes[1].set(yscale='log',ylabel='Training objects per overlapping slice')
    for ax in axes:ax.set_xlabel('QSO redshift-slice centre');ax.grid(alpha=.15)
    fig.suptitle('Historical full-data fits: 18/43 reached the limit; two distinct behaviours')
    save_figure(fig,'convergence/iteration_limit_frequency');plt.close(fig)
    path=Path('docs/UNIFIED_CONVERGENCE_PROBE_2026-09-30.json')
    if not path.exists():return
    report=json.loads(path.read_text());grouped={name:{r['update']:r for r in report['results'] if r['case']==name} for name in report['config']['cases']}
    for kind in ['training','prediction']:
        fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
        for ax,(name,methods) in zip(axes.flat,grouped.items()):
            for mode,r in methods.items():
                if kind=='training':x=np.arange(len(r['history']));y=np.array(r['history']);y-=y[0]
                else:
                    x=np.array(sorted(map(int,r['evaluations'])));y=np.array([r['evaluations'][str(i)]['mean'] for i in x]);y-=y[0]
                ax.plot(x,y,label='Variance floor' if mode=='eigenvalue_floor' else 'Added diagonal',color='#326ea4' if mode=='eigenvalue_floor' else '#c57d27')
            ax.set_title('Background' if name=='stars_00' else f"QSO z = {next(iter(methods.values()))['z']:.2f}")
            ax.set_xlabel('Updates');ax.set_ylabel('Mean log-density change [nat/object]');ax.grid(alpha=.15)
        axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=8)
        fig.suptitle('Small unified experiment: '+('training likelihood' if kind=='training' else 'non-training predictive density (≥5σ detections)'))
        save_figure(fig,'convergence/unified_'+kind);plt.close(fig)
    comparisons={}
    for name,m in grouped.items():
        floor=m['eigenvalue_floor'];add=m['additive'];last=str(report['config']['max_iter'])
        comparisons[name]=dict(floor_minus_additive_predictive=floor['evaluations'][last]['mean']-add['evaluations'][last]['mean'],
            additive_hits=add['tolerance_comparison'],floor_hits=floor['tolerance_comparison'],
            additive_negatives=add['negative_steps'],floor_negatives=floor['negative_steps'])
    write_json(Path('docs/UNIFIED_CONVERGENCE_COMPARISON_2026-09-30.json'),dict(historical=groups,comparison=comparisons))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
