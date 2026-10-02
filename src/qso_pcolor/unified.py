"""Shared latent Legacy photometry, native observation views, and QSO support.

Photometric vectors are asinh magnitudes; covariances are mag squared. Native
conditional likelihoods are normalized over the observed non-reference bands.
Support percentiles are diagnostics, not class probabilities.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import galactic_from_equatorial
from .gaussmix import GaussianMixture, condition_joint
from .legacy import hemisphere_of
from .multisurvey import BandLuptitudeTransform
from .multisurvey_baseline import PSFMultiSurveyBaseline
from .multisurvey_data import Photometry
from .projected_xd import native_mixture


def catalogue_photometry(values, errors, bands, *, ra_deg, dec_deg,
                         measurement: str = 'flux', legacy_hemisphere=None) -> Photometry:
    """Convert catalogue arrays to labelled native photometry.

    Flux inputs/errors are nanomaggies. Magnitudes/errors must be AB mag.
    ``legacy:g`` (and r/z/w1/w2) uses the position-based default system unless
    a per-row north/south provenance is supplied. Explicit native labels win.
    Missing entries are NaN; weak and negative fluxes remain valid.
    """
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    if values.ndim != 2 or values.shape != errors.shape or values.shape[1] != len(bands):
        raise ValueError('values/errors must have shape (objects, bands)')
    if measurement == 'abmag':
        flux = 10**((22.5-values)/2.5)
        error = np.log(10)/2.5*flux*errors
    elif measurement == 'flux':
        flux, error = values, errors
    else:
        raise ValueError('measurement must be flux or abmag')
    n = len(values)
    hemi = hemisphere_of(np.broadcast_to(ra_deg,(n,)), np.broadcast_to(dec_deg,(n,))) if legacy_hemisphere is None else np.broadcast_to(legacy_hemisphere,(n,))
    if not np.isin(hemi,['north','south']).all():
        raise ValueError('native Legacy provenance must be north or south')
    labels=[];columns=[]
    for j,label in enumerate(bands):
        if label.startswith('legacy:'):
            for h in ('south','north'):
                labels.append('decals_dr9_'+h+':'+label.split(':')[1]);columns.append((j,hemi==h))
        else:
            labels.append(label);columns.append((j,np.ones(n,bool)))
    f=np.full((n,len(labels)),np.nan);v=np.full_like(f,np.inf)
    for k,(j,use) in enumerate(columns):
        f[use,k]=flux[use,j];v[use,k]=np.where(error[use,j]>0,error[use,j]**2,np.inf)
    return Photometry(f,v,tuple(labels))


def observation_layout(transform: BandLuptitudeTransform, calibration: dict,
                       *, native_variance_floor: float) -> dict:
    """Create a 36-latent/41-native layout with common Legacy WISE softening.

    The configured northern WISE residual variance makes the exported joint
    native covariance nonsingular. It is recorded and included in fitting,
    not a floor on a density. Other survey systems remain separate.
    """
    if not np.isfinite(native_variance_floor) or native_variance_floor <= 0:
        raise ValueError("positive native residual variance required")
    labels = transform.bands
    keep = [j for j, b in enumerate(labels) if not b.startswith('decals_dr9_north:')]
    latent = tuple(labels[j].replace('decals_dr9_south:', 'legacy:') for j in keep)
    h = np.zeros((len(labels), len(keep)))
    for i, label in enumerate(labels):
        target = label.replace('decals_dr9_north:', 'legacy:').replace('decals_dr9_south:', 'legacy:')
        h[i, latent.index(target)] = 1.
    north = [labels.index('decals_dr9_north:'+b) for b in ('g', 'r', 'z')]
    optical = [latent.index('legacy:'+b) for b in ('g', 'r', 'z')]
    h[np.ix_(north, optical)] = calibration['matrix']
    soft = transform.softening.copy()
    nw = [labels.index('decals_dr9_north:'+b) for b in ('w1', 'w2')]
    sw = [labels.index('decals_dr9_south:'+b) for b in ('w1', 'w2')]
    soft[nw] = soft[sw]
    operators = {}
    for kind in ('qso', 'stars'):
        pop = calibration['populations'][kind]
        b = np.zeros(len(labels)); t = np.zeros((len(labels), len(labels)))
        b[north] = pop['offset']; t[np.ix_(north, north)] = pop['extra_covariance']
        t[nw, nw] = native_variance_floor
        operators[kind] = dict(matrix=h.tolist(), offset=b.tolist(), covariance=t.tolist())
    return dict(latent_labels=latent, native_labels=labels, canonical_indices=keep,
                softening=soft.tolist(), operators=operators,
                native_variance_floor=native_variance_floor)


def operator(layout: dict, kind: str) -> tuple:
    """Observation matrix, offset (mag), and extra covariance (mag squared)."""
    op = layout['operators'][kind]
    return tuple(np.array(op[k]) for k in ('matrix', 'offset', 'covariance'))


def native_view(mix: GaussianMixture, layout: dict, kind: str) -> GaussianMixture:
    """Expand one shared latent mixture into all labelled observation channels."""
    return native_mixture(mix, *operator(layout, kind), labels=tuple(layout['native_labels']))


def conditional_predictive_mixture(qso, z: float, x: np.ndarray,
                                   covariance: np.ndarray, observed: np.ndarray,
                                   anchor: int) -> tuple[GaussianMixture, np.ndarray]:
    """Observed non-reference mixture at z, with noise added once before conditioning.

    Interpolate normalized conditional slice densities, matching the scorer.
    Returned indices identify the observed non-reference input coordinates.
    """
    if not bool(qso.in_support(z)):
        raise ValueError('redshift outside trained support')
    dims = np.flatnonzero(observed)
    if anchor not in dims:
        raise ValueError('reference must be observed')
    a = int(np.flatnonzero(dims == anchor)[0]); other = dims[dims != anchor]
    idx = int(np.clip(np.searchsorted(qso.z_centres, z)-1, 0, len(qso.z_centres)-2))
    fraction = float(np.clip((z-qso.z_centres[idx])/(qso.z_centres[idx+1]-qso.z_centres[idx]), 0, 1))
    parts = []
    for j, weight in ((idx, 1-fraction), (idx+1, fraction)):
        if weight == 0: continue
        mix = qso.mixtures[j].marginal(dims)
        noisy = GaussianMixture(mix.weights, mix.means,
            mix.covs+covariance[np.ix_(dims, dims)], mix.labels)
        with np.errstate(divide='ignore'):
            _, conditional = condition_joint(noisy, [x[anchor]], np.array([a]))
        parts.append((weight, conditional))
    return GaussianMixture(np.concatenate([w*m.weights for w,m in parts]),
        np.concatenate([m.means for _,m in parts]), np.concatenate([m.covs for _,m in parts])), other


def qso_support(model, photometry: Photometry, z_primary, *, draws: int,
                seed: int, flux_covariance=None, l_deg=None, b_deg=None) -> dict:
    """Noise-aware QSO conditional-density percentile at the specified redshift.

    Monte Carlo ranks use (1 + count)/(draws + 1). The deterministic row seed
    depends on the measured inputs, making results invariant to batch ordering.
    Single-band data have no colour support information and return NaN.
    """
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('positive integer support draws required')
    if 'extinction' in model.meta:
        # Catalogue photometry must be corrected exactly as at training.
        if l_deg is None or b_deg is None:
            raise ValueError('this model is extinction-corrected: Galactic l_deg and b_deg are required')
        from .extinction import deredden
        photometry, flux_covariance, _ = deredden(photometry, l_deg, b_deg, model.meta['extinction'],
                                                  flux_covariance=flux_covariance)
    f = model.transform(photometry, flux_covariance=flux_covariance)
    anchors = model.reference_indices(photometry)
    z = np.broadcast_to(np.asarray(z_primary, float), (len(f.x),))
    value = np.full(len(z), np.nan); logp = value.copy()
    status = np.full(len(z), 'ok', dtype='<U40')
    for i in range(len(z)):
        if f.observed[i].sum() < 2:
            status[i] = 'no_colour_information'; continue
        if not model.qso.in_support(z[i]):
            status[i] = 'redshift_outside_support'; continue
        mix, dims = conditional_predictive_mixture(model.qso, float(z[i]), f.x[i],
            f.cov[i], f.observed[i], int(anchors[i]))
        payload = f.x[i, f.observed[i]].tobytes()+f.cov[i].tobytes()+f.observed[i].tobytes()+np.float64(z[i]).tobytes()
        rng = np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')]))
        sample = mix.sample(draws, rng)
        logp[i] = mix.log_prob(f.x[i, dims])[0]
        value[i] = (1+np.count_nonzero(mix.log_prob(sample) <= logp[i]))/(draws+1)
    return dict(percentile=value, log_density=logp, status=status, anchor=anchors,
                n_colour=f.observed.sum(axis=1)-1)


class UnifiedPSFModel:
    """Native-band PSF scorer with RA/Dec convenience and saved support policy."""
    def __init__(self, baseline: PSFMultiSurveyBaseline, support: dict):
        self.base, self.support = baseline, support
        if not 0 <= support['threshold'] <= 1:
            raise ValueError('support threshold must be in [0,1]')

    @classmethod
    def load(cls, path: str | Path):
        """Load a bundle directory or a pointer file such as ``models/multisurvey_psf/current``."""
        root = Path(path)
        if root.is_file():
            root = root.parent/json.loads(root.read_text())['bundle']
        base = PSFMultiSurveyBaseline.load(root)
        for name, expected in base.manifest['unified_files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('unified metadata hash mismatch: '+name)
        return cls(base, json.loads((root/'support.json').read_text()))

    def score(self, photometry: Photometry, *, ra_deg, dec_deg, z_primary,
              apply_support: bool = True, flux_covariance=None, **kwargs):
        """Use candidate ICRS degrees and primary-QSO redshift; return support separately.

        Explicit survey-labelled bands identify actual native provenance even
        in overlap fields. The coordinate hemisphere is an informational default.
        """
        n = len(photometry.flux)
        ra = np.broadcast_to(np.asarray(ra_deg, float), (n,))
        dec = np.broadcast_to(np.asarray(dec_deg, float), (n,))
        if not np.isfinite(ra+dec).all() or (abs(dec)>90).any():
            raise ValueError('finite ICRS coordinates with valid declination required')
        l, b = galactic_from_equatorial(ra, dec)
        scores, decision = self.base.score(photometry, z_primary=z_primary,
            l_deg=l, b_deg=b, flux_covariance=flux_covariance, **kwargs)
        support = qso_support(self.base.model, photometry, z_primary,
            draws=self.support['draws'], seed=self.support['seed'], flux_covariance=flux_covariance,
            l_deg=l, b_deg=b)
        rejected = np.isfinite(support['percentile']) & (support['percentile'] < self.support['threshold'])
        if apply_support:
            for i in np.flatnonzero(rejected & decision['eligible']):
                scores[i] = replace(scores[i], status='qso_support_rejected',
                    quality_flags=tuple(scores[i].quality_flags)+('qso_support_rejected',),
                    log_r_per_unit_z=np.nan, p_sameq=np.nan, p_sameq_vs_bkg=np.nan,
                    p_outlier=np.nan, p_zmatch_given_qso=np.nan)
                decision['eligible'][i] = False; decision['reason'][i] = 'qso_support_rejected'
        decision.update(qso_support=support, support_rejected=rejected,
                        coordinate_hemisphere=hemisphere_of(ra, dec, b))
        return scores, decision

    def score_catalogue(self, values, errors, bands, *, ra_deg, dec_deg,
                        z_primary, measurement='flux', legacy_hemisphere=None, **kwargs):
        """Score fluxes or AB magnitudes with errors, candidate RA/Dec and primary z."""
        phot=catalogue_photometry(values,errors,bands,ra_deg=ra_deg,dec_deg=dec_deg,
            measurement=measurement,legacy_hemisphere=legacy_hemisphere)
        return self.score(phot,ra_deg=ra_deg,dec_deg=dec_deg,z_primary=z_primary,**kwargs)

    def fit_local(self, **kwargs):
        """Refit local contaminant weights/counts; QSO support calibration is unchanged."""
        return type(self)(self.base.fit_local(**kwargs), dict(self.support))
