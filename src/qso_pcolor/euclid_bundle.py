"""Euclid DR1 photometry for the resolved-mode bundles (docs/euclid/PLAN.md).

- Catalogue rows (``qso_pcolor.euclid_data.match`` output) -> native Euclid photometry, cleaning, point rule.
- Extended bundle assembly: the parent bundle's files with every mixture extended by the fitted Euclid block.
  Every parent number (native old-block means/covariances, priors, outlier moments, spatial weights) is copied, not
  recomputed, so 41-band scores are unchanged under ``selection='legacy_psf'`` (test T-U1).
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .euclid_data import UJY_TO_NATIVE
from .euclid_extension import Extension, extend_mixture
from .gaussmix import GaussianMixture
from .multisurvey_data import EXTENSION_BANDS, Photometry

EUCLID_BANDS = EXTENSION_BANDS
COLUMNS = {'euclid:vis': 'vis_to_y_templfit', 'euclid:y': 'y_templfit', 'euclid:j': 'j_templfit',
           'euclid:h': 'h_templfit'}
# A_band / E(B-V)_SFD, SF11 recipe (F99, R_V = 3.1, 7000 K, scale 0.8787 fitted to SF11 Table 6, rms 0.98%),
# SVO Euclid/VIS.vis, Euclid/NISP.{Y,J,H}; qso_pcolor.band_extinction. VIS for a 3000 K SED: 1.702.
EXTINCTION_R = {'euclid:vis': 1.917, 'euclid:y': 0.921, 'euclid:j': 0.629, 'euclid:h': 0.420}
A = 2.5/np.log(10.)


def photometry(match: dict) -> Photometry:
    """Native (not dereddened) Euclid fluxes and variances; unusable entries NaN / inf."""
    n = len(match[f"flux_{COLUMNS['euclid:vis']}"])
    matched = np.asarray(match.get('matched', np.ones(n, bool)), bool)       # catalogue rows: all matched
    f = np.full((n, 4), np.nan); v = np.full_like(f, np.inf)
    for j, band in enumerate(EUCLID_BANDS):
        c = COLUMNS[band]
        flux = np.asarray(match['flux_'+c], float)*UJY_TO_NATIVE
        err = np.asarray(match['fluxerr_'+c], float)*UJY_TO_NATIVE
        ok = matched & np.isfinite(flux) & np.isfinite(err) & (err > 0)
        f[ok, j] = flux[ok]; v[ok, j] = err[ok]**2
    return Photometry(f, v, EUCLID_BANDS)


def dereddened(phot: Photometry, ebv) -> Photometry:
    """flux * 10**(0.4 R E), variance * 10**(0.8 R E)."""
    r = np.array([EXTINCTION_R[b] for b in phot.bands])
    fac = 10**(0.4*np.outer(np.asarray(ebv, float), r))
    return Photometry(phot.flux*fac, phot.variance*fac**2, phot.bands)


def luptitudes(flux, variance, softening):
    """Luptitude and its variance (mag^2), as BandLuptitudeTransform for diagonal errors; NaN/0 where unusable."""
    f = np.asarray(flux, float); s = np.asarray(softening, float)
    ok = np.isfinite(f) & np.isfinite(variance) & (np.asarray(variance) > 0)
    with np.errstate(invalid='ignore'):
        y = np.where(ok, 22.5 - A*(np.arcsinh(f/(2*s)) + np.log(s)), np.nan)
        n = np.where(ok, (A*np.sqrt(variance)/np.hypot(f, 2*s))**2, 0.)
    return y, n


def clean(match: dict, *, neighbours=None, bad_bits: int = 0, vis_bright: float | None = None) -> np.ndarray:
    """Plan 2.1 cleaning: matched, vis_det = 1, spurious_flag = 0, no isolation neighbour, no ``bad_bits`` in
    det_quality_flag, VIS template-fit magnitude fainter than ``vis_bright``. Sentinels count as failing."""
    ok = np.asarray(match['matched'], bool).copy()
    ok &= np.asarray(match['vis_det']) == 1
    ok &= np.asarray(match['spurious_flag']) == 0
    dq = np.asarray(match['det_quality_flag']).astype(np.int64)
    ok &= (dq >= 0) & ((dq & bad_bits) == 0)
    if neighbours is not None:
        ok &= np.asarray(neighbours) == 0
    if vis_bright is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            vis = 23.9 - 2.5*np.log10(np.asarray(match['flux_vis_to_y_templfit'], float))
        ok &= np.isfinite(vis) & (vis >= vis_bright)
    return ok


def point_like(match: dict, p_min: float) -> np.ndarray:
    """Euclid point rule: point_like_prob >= p_min (NaN fails). point_like_flag is never used (shear stars)."""
    p = np.asarray(match['point_like_prob'], float)
    with np.errstate(invalid='ignore'):
        return np.isfinite(p) & (p >= p_min)


# ---------------------------------------------------------------- bundle assembly

def extend_layout(layout: dict, softening_euclid) -> dict:
    """Latent 36 -> 40 and native 41 -> 45 with identity observation rows for the Euclid bands (no extra variance)."""
    out = deepcopy(layout)
    nd, nl = len(layout['native_labels']), len(layout['latent_labels'])
    out['latent_labels'] = list(layout['latent_labels']) + list(EUCLID_BANDS)
    out['native_labels'] = list(layout['native_labels']) + list(EUCLID_BANDS)
    out['canonical_indices'] = list(layout['canonical_indices']) + list(range(nd, nd + 4))
    out['softening'] = list(layout['softening']) + [float(s) for s in softening_euclid]
    for kind, op in out['operators'].items():
        h = np.zeros((nd + 4, nl + 4)); h[:nd, :nl] = op['matrix']; h[nd:, nl:] = np.eye(4)
        b = np.r_[op['offset'], np.zeros(4)]
        t = np.zeros((nd + 4, nd + 4)); t[:nd, :nd] = op['covariance']
        op.update(matrix=h.tolist(), offset=b.tolist(), covariance=t.tolist())
    if 'extinction' in out:
        e = out['extinction']
        e['labels'] = list(e['labels']) + list(EUCLID_BANDS)
        e['coefficients'] = list(e['coefficients']) + [EXTINCTION_R[b] for b in EUCLID_BANDS]
        e['euclid'] = 'Euclid R from qso_pcolor.band_extinction (SF11 recipe, SVO filter curves)'
    if 'qso_coordinates' in out:
        q = out['qso_coordinates']; r = q['index']
        mat = np.zeros((nl + 4, nl + 4)); mat[:nl, :nl] = q['matrix']
        for j in range(4):
            mat[nl + j, nl + j] = 1.; mat[nl + j, r] = -1.
        q['matrix'] = mat.tolist()
        q['labels'] = list(q['labels']) + [f'colour:{b}-legacy:r' for b in EUCLID_BANDS]
    return out


def native_extension(parent_native: GaussianMixture, latent_ext: GaussianMixture, layout_ext: dict,
                     kind: str) -> GaussianMixture:
    """Native 45-band mixture: parent native block copied exactly; Euclid rows from the extended latent mixture."""
    h = np.asarray(layout_ext['operators'][kind]['matrix'])
    nd = parent_native.n_dim; nl = latent_ext.n_dim - 4
    hold = h[:nd, :nl]
    k = latent_ext.n_components
    means = np.concatenate([parent_native.means, latent_ext.means[:, nl:]], axis=1)
    covs = np.zeros((k, nd + 4, nd + 4))
    covs[:, :nd, :nd] = parent_native.covs
    cross = hold[None] @ latent_ext.covs[:, :nl, nl:]                           # (K, nd, 4)
    covs[:, :nd, nd:] = cross; covs[:, nd:, :nd] = cross.swapaxes(-1, -2)
    covs[:, nd:, nd:] = latent_ext.covs[:, nl:, nl:]
    return GaussianMixture(parent_native.weights, means, covs, tuple(layout_ext['native_labels']))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_bundle(parent_root, out_root, *, qso_ext: list[Extension], bkg_ext: Extension, t_d_qso, t_d_bkg,
                 softening_euclid, bounds_euclid, selection: dict | None = None, c_q: dict | None = None,
                 outlier_fractions: dict | None = None, support: dict | None = None, meta: dict | None = None,
                 euclid_band_counts=None, fallback: dict | None = None):
    """Write an extended bundle. ``selection=None``: parent deployment selection (legacy_psf), parent priors,
    catch-all fractions and support. Otherwise ``selection = dict(name, thinning)`` and ``c_q`` (per Legacy-r
    anchor: array (n_z, n_mag) multiplying Sigma_Q), ``outlier_fractions`` and ``support`` for that selection.
    ``euclid_band_counts`` (n_slices, 4): Euclid training QSOs per slice and band (the scorer flags bands trained on
    fewer than ``min_band_training``); default zero. ``fallback``: model.meta['fallback'] (bands, windows, n_grid) for
    scoring without Legacy r (qso_pcolor.fallback_reference); default: the parent rule (PS1/NSC/SkyMapper r)."""
    from .multisurvey import MultiSurveyModel, MultiSurveyOutlier
    from .multisurvey import BandLuptitudeTransform
    from .qso_model import SlicedColourRedshiftModel
    from .outlier import mixture_moments
    parent_root, out_root = Path(parent_root), Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    latent = json.loads((parent_root/'latent.json').read_text())
    parent = MultiSurveyModel.load(parent_root/'model.json')
    lay = extend_layout(latent['layout'], softening_euclid)
    nl = len(latent['layout']['latent_labels']); r_index = latent['layout']['latent_labels'].index('legacy:r')
    q_lat = [extend_mixture(GaussianMixture.from_dict(m), e, t_d_qso, r_index, EUCLID_BANDS)
             for m, e in zip(latent['qso'], qso_ext)]
    b_lat = extend_mixture(GaussianMixture.from_dict(latent['background']), bkg_ext, t_d_bkg, r_index, EUCLID_BANDS)
    q_nat = [native_extension(pm, lm, lay, 'qso') for pm, lm in zip(parent.qso.mixtures, q_lat)]
    b_nat = native_extension(parent.background, b_lat, lay, 'stars')
    bands = tuple(lay['native_labels'])
    transform = BandLuptitudeTransform(bands, np.array(lay['softening']))
    qmeta = deepcopy(parent.qso.meta)
    counts = np.zeros((len(parent.qso.z_centres), 4), int) if euclid_band_counts is None else np.asarray(euclid_band_counts, int)
    for sl, c in zip(qmeta.get('per_slice', []), counts):
        if 'band_counts' in sl:
            sl['band_counts'] = list(sl['band_counts']) + c.tolist()
    qso = SlicedColourRedshiftModel(parent.qso.z_centres, q_nat, parent.qso.n_train, parent.qso.system, bands, qmeta)
    bounds = np.concatenate([parent.background_bounds, np.asarray(bounds_euclid, float)])
    run_id = parent.meta['run_id'] + '+euclid_' + hashlib.sha256(json.dumps(
        [e.to_dict() for e in qso_ext] + [bkg_ext.to_dict(), selection], default=float).encode()).hexdigest()[:12]
    mmeta = deepcopy(parent.meta); mmeta['run_id'] = run_id
    if 'extinction' in mmeta:
        mmeta['extinction'] = lay['extinction']
    mmeta['euclid'] = dict(meta or {}, parent_run_id=parent.meta['run_id'], parent_bundle=str(parent_root))
    if selection is not None:
        mmeta['selection'] = selection
    if fallback is not None:
        mmeta['fallback'] = fallback
    model = MultiSurveyModel(qso, b_nat, transform, tuple(parent.reference_priority) + EUCLID_BANDS, bounds,
                             mmeta, spatial_background=parent.spatial_background)
    model.save(out_root/'model.json')
    # priors: parent numbers, re-keyed to the new transform; Sigma_Q x c_Q for a non-parent selection
    pri = json.loads((parent_root/'priors.json').read_text())
    pri['transform_id'] = model.transform_id; pri['model_run_id'] = run_id
    for label, e in pri['anchors'].items():
        for part in ('qso_prior', 'background_density'):
            e[part]['meta'].update(transform_id=model.transform_id, model_run_id=run_id)
        if c_q is not None and label in c_q:
            e['qso_prior']['sigma'] = (np.asarray(e['qso_prior']['sigma'])*np.asarray(c_q[label])).tolist()
            e['qso_prior']['meta']['selection'] = selection['name']
    (out_root/'priors.json').write_text(json.dumps(pri))
    # catch-all: parent moments for the old block (copied), extended moments for the Euclid rows/columns
    po = json.loads((parent_root/'outlier.json').read_text())
    mean, cov = mixture_moments([b_nat]); nd = parent.background.n_dim
    cov = po['kappa']**2*cov                                   # stored scale = kappa^2 x field moments (parent rule)
    mean[:nd] = po['mean']; cov[:nd, :nd] = po['cov']
    fr = po['fractions'] if outlier_fractions is None else outlier_fractions
    out = MultiSurveyOutlier(mean, cov, po['kappa'], po['kappa_min'],
                             bands, model.transform_id, fr, meta=dict(po['meta'], model_run_id=run_id,
                             euclid_rows='moments of the extended background; old block copied from the parent'),
                             family=po['family'], nu=po['nu'], noise=po['noise'])
    out.save(out_root/'outlier.json')
    sup = json.loads((parent_root/'support.json').read_text()) if support is None else support
    (out_root/'support.json').write_text(json.dumps(sup, indent=1))
    lat = dict(layout=lay, qso=[m.to_dict() for m in q_lat], background=b_lat.to_dict(),
               extension=dict(qso=[e.to_dict() for e in qso_ext], background=bkg_ext.to_dict(),
                              t_d_qso=np.asarray(t_d_qso).tolist(), t_d_bkg=np.asarray(t_d_bkg).tolist()),
               parent=str(parent_root), scope=latent.get('scope', ''))
    (out_root/'latent.json').write_text(json.dumps(lat))
    pman = json.loads((parent_root/'manifest.json').read_text())
    man = dict(pman, files={n: _sha(out_root/n) for n in ('model.json', 'priors.json', 'outlier.json')},
               unified_files={n: _sha(out_root/n) for n in ('latent.json', 'support.json')},
               bundle_id=out_root.name, source_bundle=pman['bundle_id'],
               selection=(selection or {}).get('name', 'legacy_psf'),
               status='candidate (Euclid extension, docs/euclid/PLAN.md); not promoted')
    (out_root/'manifest.json').write_text(json.dumps(man, indent=1))
    return out_root


def build_joint_bundle(stage1_root, out_root, *, qso_latent: list[GaussianMixture], bkg_latent: GaussianMixture,
                       selection: dict | None, c_q: dict | None = None, meta: dict | None = None):
    """Stage-2 bundle: refitted 40-d latent mixtures (old blocks changed), everything else from the stage-1 bundle.

    Native views come from the extended observation operator (no copying, since the old blocks were refitted).
    Kept from stage 1 / the parent: sky-gate weights (component order preserved by the warm start), Sigma_B,
    Sigma_Q (x c_Q for ``euclid_point``), catch-all fraction and scale rule, support threshold, Euclid fallback.
    ``selection=None`` builds the parent deployment selection (no thinning, parent priors).
    """
    from .multisurvey import MultiSurveyModel, MultiSurveyOutlier
    from .outlier import mixture_moments
    from .projected_xd import native_mixture
    from .qso_model import SlicedColourRedshiftModel
    st1, out_root = Path(stage1_root), Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    lat1 = json.loads((st1/'latent.json').read_text()); lay = lat1['layout']
    m1 = MultiSurveyModel.load(st1/'model.json')
    nat = lambda m, kind: native_mixture(m, *[np.asarray(lay['operators'][kind][k]) for k in ('matrix', 'offset', 'covariance')],
                                         labels=tuple(lay['native_labels']))
    qso = SlicedColourRedshiftModel(m1.qso.z_centres, [nat(m, 'qso') for m in qso_latent], m1.qso.n_train,
                                    m1.qso.system, tuple(lay['native_labels']), deepcopy(m1.qso.meta))
    bnat = nat(bkg_latent, 'stars')
    meta1 = deepcopy(m1.meta)
    run_id = meta1['euclid']['parent_run_id'] + '+euclid_joint_' + hashlib.sha256(
        json.dumps([m.to_dict() for m in qso_latent] + [bkg_latent.to_dict(), selection], default=float).encode()).hexdigest()[:12]
    meta1['run_id'] = run_id; meta1['euclid'] = dict(meta1['euclid'], stage=2, **(meta or {}))
    meta1.pop('selection', None)
    if selection is not None:
        meta1['selection'] = selection
    model = MultiSurveyModel(qso, bnat, m1.transform, m1.reference_priority, m1.background_bounds, meta1,
                             spatial_background=m1.spatial_background)
    model.save(out_root/'model.json')
    parent_root = Path(meta1['euclid']['parent_bundle'])
    pri = json.loads((parent_root/'priors.json').read_text())
    pri['transform_id'] = model.transform_id; pri['model_run_id'] = run_id
    for label, e in pri['anchors'].items():
        for part in ('qso_prior', 'background_density'):
            e[part]['meta'].update(transform_id=model.transform_id, model_run_id=run_id)
        if c_q is not None and label in c_q:
            e['qso_prior']['sigma'] = (np.asarray(e['qso_prior']['sigma'])*np.asarray(c_q[label])).tolist()
            e['qso_prior']['meta']['selection'] = selection['name']
    (out_root/'priors.json').write_text(json.dumps(pri))
    po = json.loads((parent_root/'outlier.json').read_text())
    mean, cov = mixture_moments([bnat])
    out = MultiSurveyOutlier(mean, po['kappa']**2*cov, po['kappa'], po['kappa_min'], tuple(lay['native_labels']),
                             model.transform_id, po['fractions'], meta=dict(po['meta'], model_run_id=run_id,
                             rule='moments of the stage-2 background; parent kappa and fractions'),
                             family=po['family'], nu=po['nu'], noise=po['noise'])
    out.save(out_root/'outlier.json')
    (out_root/'support.json').write_text((st1/'support.json').read_text())
    (out_root/'latent.json').write_text(json.dumps(dict(layout=lay, qso=[m.to_dict() for m in qso_latent],
                                                         background=bkg_latent.to_dict(), stage=2, stage1=str(st1))))
    pman = json.loads((st1/'manifest.json').read_text())
    man = dict(pman, files={n: _sha(out_root/n) for n in ('model.json', 'priors.json', 'outlier.json')},
               unified_files={n: _sha(out_root/n) for n in ('latent.json', 'support.json')}, bundle_id=out_root.name,
               selection=(selection or {}).get('name', 'legacy_psf'),
               status='candidate (Euclid stage 2 joint refit, docs/euclid/PLAN.md); not promoted')
    (out_root/'manifest.json').write_text(json.dumps(man, indent=1))
    return out_root
