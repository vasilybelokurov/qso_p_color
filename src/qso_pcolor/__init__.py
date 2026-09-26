"""Colour evidence that a quasar's companion is itself a quasar at the same redshift.

See ``AGENTS.md`` for the working contract and ``docs/method/method.pdf`` for
the method.  The short version:

- The model is ``MultiSurveyModel`` (``models/multisurvey.json``) with its
  reference-band priors (``load_priors``) and its unmodelled term
  (``MultiSurveyOutlier``): any subset of 41 bands from seven surveys.
- It weighs four hypotheses — a quasar at the primary's redshift, a quasar at
  any other redshift, the modelled field, and a broad unmodelled share of the
  field — and returns both prior-independent evidence and, when a prior pair
  exists for the reference band, a posterior and the ranking statistic R.
- Every density is a log density; every covariance is a full matrix; bands the
  survey could not measure are marginalised out exactly rather than imputed.
"""

from .background import BackgroundColourModel, fit_background_model, tune_shrinkage
from .features import (
    AsinhColourTransform,
    FeatureSet,
    RelativeFluxTransform,
    deredden,
    flux_to_mag,
)
from .gaussmix import GaussianMixture, condition_joint
from .plotting import PLOTS_DIR, plot_path, save_figure
from .priors import BackgroundSurfaceDensity, EmpiricalQSOPrior, GridQSOPrior
from .qso_model import (
    JointColourRedshiftModel,
    RedshiftMatch,
    SlicedColourRedshiftModel,
    fit_sliced_model,
)
from .score import BlendPolicy, PairScore, score_candidates
from .multisurvey import (BandLuptitudeTransform, MultiSurveyModel, MultiSurveyOutlier,
                          MultiSurveyScore, load_priors)
from .outlier import OutlierModel
from .multisurvey_data import Photometry
from .xd import fit_xd, select_n_components

__version__ = "0.1.0"

__all__ = [
    "AsinhColourTransform",
    "BandLuptitudeTransform",
    "BlendPolicy",
    "BackgroundColourModel",
    "BackgroundSurfaceDensity",
    "EmpiricalQSOPrior",
    "FeatureSet",
    "GaussianMixture",
    "GridQSOPrior",
    "JointColourRedshiftModel",
    "MultiSurveyModel",
    "MultiSurveyOutlier",
    "MultiSurveyScore",
    "OutlierModel",
    "load_priors",
    "PLOTS_DIR",
    "PairScore",
    "Photometry",
    "RedshiftMatch",
    "RelativeFluxTransform",
    "SlicedColourRedshiftModel",
    "condition_joint",
    "deredden",
    "fit_background_model",
    "fit_sliced_model",
    "fit_xd",
    "flux_to_mag",
    "plot_path",
    "save_figure",
    "score_candidates",
    "select_n_components",
    "tune_shrinkage",
    "__version__",
]
