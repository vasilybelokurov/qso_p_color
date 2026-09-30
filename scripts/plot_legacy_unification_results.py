#!/usr/bin/env python
"""Display the saved intervention results without rerunning scoring."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qso_pcolor.plotting import save_figure


def main():
    report=json.loads(Path('docs/LEGACY_UNIFICATION_SHARED_DENSITY_2026-09-30.json').read_text())
    release=json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())
    cfg=json.loads(Path('configs/legacy_unification_test.json').read_text())
    magnitude=cfg['probe_reference_magnitude'];grid=release['config']['grid']
    nodes=np.linspace(*grid['grid_colour_range'],grid['grid_size']);step=nodes[1]-nodes[0]
    extent=[nodes[0]-step/2,nodes[-1]+step/2]*2
    before=np.load(Path(release['cache'])/f'grid_north_{magnitude}_candidate.npz')
    after=np.load(Path(report['bundle'])/f'grid_north_{magnitude}.npz')
    fig,axes=plt.subplots(1,2,figsize=(10,5.9))
    fig.subplots_adjust(top=.76,bottom=.22,right=.85,wspace=.25)
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d9dde0')
    for ax,values,title in zip(axes,(before,after),('Current full-data candidate','Shared optical diagnostic')):
        p=np.ma.array(values['p_quasar'],mask=~values['eligible']).reshape(grid['grid_size'],grid['grid_size'])
        im=ax.imshow(p,origin='lower',extent=extent,interpolation='nearest',vmin=0,vmax=1,cmap=cmap)
        ax.scatter(*cfg['probe_colours'],marker='*',c='#00e8ef',edgecolors='black',s=140)
        ax.set_title(title,fontsize=11);ax.set_xlabel('g − r [luptitude mag]');ax.set_ylabel('r − z [luptitude mag]',labelpad=9)
    cax=fig.add_axes([.88,.25,.02,.45]);fig.colorbar(im,cax=cax,label='Total-QSO score (any redshift; uncalibrated)')
    fig.suptitle('Sharing the optical distribution removes the demonstrated northern tail failure',fontsize=13,y=.98)
    fig.text(.5,.88,'Bright grid: 72 → 0 high-QSO low-density points; intermediate grid: 79 → 0',ha='center',fontsize=11)
    fig.text(.5,.10,'Marked example: 99.5% → 0.7%. Grey = rejected. Same saved grids and low-density definition.\nDiagnostic uses southern shapes + DESI correction; it is not a fit to pooled North+South data.',ha='center',fontsize=10)
    print(save_figure(fig,'legacy_unification/shared_density_grid'))


if __name__=='__main__':main()
