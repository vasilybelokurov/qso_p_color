#!/usr/bin/env python
"""Plot the frozen comparison's colour-space stress tests, without refitting."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_psf_catchalls import fractions_at
from validate_psf_catchalls import reweight
from qso_pcolor import PSFMultiSurveyBaseline
from qso_pcolor.multisurvey import MultiSurveyOutlier
from qso_pcolor.plotting import save_figure


def main():
    cfg = json.loads(Path("configs/psf_catchall_comparison.json").read_text())
    root = Path(cfg["work"])
    report = json.loads((root/"validation_report.json").read_text())
    cache = root/f"validation_{report['signature']}"
    baseline = PSFMultiSurveyBaseline.load(cfg["baseline"])
    variants = {"Previous":baseline.outlier,
                "Pooled Gaussian":MultiSurveyOutlier.load(root/"gaussian.json"),
                "Pooled Student-t":MultiSurveyOutlier.load(root/"student_t.json")}
    v=cfg["validation"]; n=v["grid_size"]
    axes=np.linspace(*v["grid_colour_range"],n)
    fig,axs=plt.subplots(2,4,figsize=(13,6.7),sharex=True,sharey=True,layout="constrained")
    fraction=v["low_density_peak_fractions"][-1]
    for row,mag in enumerate(v["reference_magnitudes"][:2]):
        data=dict(np.load(cache/f"north_{mag}.npz"))
        raw={k[4:]:x for k,x in data.items() if k.startswith("raw_")}
        q=np.logaddexp(raw["log_lambda_sameq"],raw["log_lambda_fieldq"])
        b=raw["log_lambda_bkg"]
        low=(q<q.max()+np.log(fraction))&(b<b.max()+np.log(fraction))
        for col,(title,outlier) in enumerate(dict({"No catch-all":None},**variants).items()):
            if outlier is None:
                r=reweight(raw,np.zeros(n*n),np.zeros(n*n))
            else:
                name={"Previous":"previous","Pooled Gaussian":"gaussian","Pooled Student-t":"student_t"}[title]
                eta=fractions_at(outlier.fractions,data["dens_anchor"],data["dens_magnitude"],baseline.model.transform.bands)
                r=reweight(raw,data[f"dens_{name}"],eta)
            ax=axs[row,col]
            im=ax.pcolormesh(axes,axes,r["p_quasar"].reshape(n,n),vmin=0,vmax=1,cmap="magma",shading="auto")
            ax.contour(axes,axes,low.reshape(n,n).astype(float),levels=[.5],colors="cyan",linewidths=.7)
            ax.contour(axes,axes,raw["eligible"].reshape(n,n).astype(float),levels=[.5],colors="white",linewidths=.7,linestyles="dashed")
            if row==0: ax.set_title(title)
            if col==0: ax.set_ylabel(f"r = {mag} luptitudes\nr - z")
            if row==1: ax.set_xlabel("g - r (native luptitudes)")
    fig.colorbar(im,ax=axs,label="Total quasar probability before hard exclusion",shrink=.8)
    fig.suptitle("Northern optical stress test: catch-all effects at fixed stellar and quasar models\n"
                 f"Cyan: low-density boundary (both < {fraction:g} of plane peaks); white dashed: support guard",fontsize=12)
    print(save_figure(fig,"validation/psf_catchall_comparison"))


if __name__=="__main__":
    main()
