"""Colour evidence that a quasar's companion is itself a quasar at the same redshift.

See ``AGENTS.md`` for the working contract and ``docs/REVIEW_OF_PLAN.md`` for
the reasoning behind the design.  The short version:

- ``score_candidates`` weighs three hypotheses — a quasar at the primary's
  redshift, a quasar at any other redshift, and everything else in the imaging
  catalogue — and returns both prior-independent evidence and, when a defensible
  surface-density prior exists, a posterior.
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
from .score import PairScore, score_candidates
from .multisurvey import BandLuptitudeTransform, MultiSurveyModel, MultiSurveyScore
from .multisurvey_data import Photometry
from .xd import fit_xd, select_n_components

__version__ = "0.1.0"

__all__ = [
    "AsinhColourTransform",
    "BandLuptitudeTransform",
    "BackgroundColourModel",
    "BackgroundSurfaceDensity",
    "EmpiricalQSOPrior",
    "FeatureSet",
    "GaussianMixture",
    "GridQSOPrior",
    "JointColourRedshiftModel",
    "MultiSurveyModel",
    "MultiSurveyScore",
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
