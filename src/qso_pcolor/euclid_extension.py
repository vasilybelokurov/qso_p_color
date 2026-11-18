"""Frozen extension of a latent Gaussian mixture by new bands (stage 1 of the Euclid plan, docs/euclid/PLAN.md).

Every component k keeps its old block (mean mu_k, covariance V_k over the parent latent coordinates) exactly.
The new coordinates x_E (here Euclid VIS, Y, J, H luptitudes) enter as colours relative to a magnitude coordinate
x_r, linear-Gaussian in a few old regressors d = T_d x_old:

    c_E = x_E - x_r 1 = a_k + B_k (d - mu_k,d) + eps,   eps ~ N(0, W_k)

so the joint (x_old, x_E) of component k is Gaussian with

    E[x_E] = mu_k,r 1 + a_k,   Cov(x_E, x_old) = L_k V_k,   Cov(x_E) = L_k V_k L_k^T + W_k,   L_k = 1 e_r^T + B_k T_d.

Fitting is EM with the old block frozen: the E-step is the exact XD posterior of the latent vector given each row's
observed native bands (any subset, each with its own noise), the M-step a weighted multivariate regression of
E[c_E] on E[d] using full posterior second moments (errors in the regressors are handled exactly). Components can
share B and W within a group, with per-component offsets shrunk to the group regression (``kappa_*`` pseudo-counts;
``np.inf`` ties a parameter to its group).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .gaussmix import GaussianMixture, log_gauss_batch


@dataclass
class Extension:
    """Parameters of the new block for every component: a (K, m), B (K, m, p), W (K, m, m)."""
    a: np.ndarray
    b: np.ndarray
    w: np.ndarray
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(a=self.a.tolist(), b=self.b.tolist(), w=self.w.tolist(), meta=self.meta)

    @classmethod
    def from_dict(cls, d: dict) -> "Extension":
        return cls(np.array(d['a'], float), np.array(d['b'], float), np.array(d['w'], float), d.get('meta', {}))


def regressor_matrix(labels, colours: tuple[str, ...], magnitude: str, include_magnitude: bool) -> np.ndarray:
    """T_d (p x D): rows are colours ``c - magnitude`` for each label c in ``colours`` [+ the magnitude itself]."""
    labels = list(labels); r = labels.index(magnitude)
    rows = []
    for c in colours:
        t = np.zeros(len(labels)); t[labels.index(c)] = 1.; t[r] -= 1.; rows.append(t)
    if include_magnitude:
        t = np.zeros(len(labels)); t[r] = 1.; rows.append(t)
    return np.array(rows)


def extend_mixture(mix: GaussianMixture, ext: Extension, t_d: np.ndarray, r_index: int,
                   new_labels: tuple[str, ...]) -> GaussianMixture:
    """The (D + m)-dimensional mixture; the old block is copied exactly (no arithmetic on it)."""
    k, d = mix.means.shape; m = ext.a.shape[1]
    e_r = np.zeros(d); e_r[r_index] = 1.
    lmat = np.ones((m, 1))*e_r[None, :] + ext.b @ t_d                          # (K, m, D)
    mean_e = mix.means[:, [r_index]] + ext.a
    cross = lmat @ mix.covs                                                     # (K, m, D)
    cov_e = cross @ lmat.swapaxes(-1, -2) + ext.w
    means = np.concatenate([mix.means, mean_e], axis=1)
    covs = np.zeros((k, d + m, d + m))
    covs[:, :d, :d] = mix.covs
    covs[:, d:, :d] = cross; covs[:, :d, d:] = cross.swapaxes(-1, -2)
    covs[:, d:, d:] = 0.5*(cov_e + cov_e.swapaxes(-1, -2))
    labels = tuple(mix.labels) + tuple(new_labels) if mix.labels else ()
    return GaussianMixture(mix.weights, means, covs, labels)


def _estep(mix_ext: GaussianMixture, y, noise, observed, h, offset, extra, proj, log_weights):
    """Responsibilities and posterior moments of z = proj @ latent for each row and component.

    y, noise: (N, n_native) observed luptitudes and per-band noise variances (NaN/inf where unobserved);
    h (n_native, D+m), offset (n_native,), extra (n_native, n_native): observation operator.
    Returns log-likelihood per row, responsibilities (N, K), E[z] (N, K, q), Cov[z] (N, K, q, q).
    """
    n, kk = len(y), mix_ext.n_components; q = proj.shape[0]
    ll = np.full(n, np.nan); resp = np.zeros((n, kk)); ez = np.zeros((n, kk, q)); cz = np.zeros((n, kk, q, q))
    keys, inverse = np.unique(observed, axis=0, return_inverse=True)
    pc = proj @ mix_ext.covs                                                   # (K, q, D+m)
    pm = mix_ext.means @ proj.T                                                # (K, q)
    for g, mask in enumerate(keys):
        rows = np.flatnonzero(inverse.ravel() == g)
        if not mask.any():
            continue
        ho = h[mask]
        mean = mix_ext.means @ ho.T + offset[mask]                             # (K, o)
        cross = mix_ext.covs @ ho.T                                            # (K, D+m, o)
        base = ho @ cross + extra[np.ix_(mask, mask)]                          # (K, o, o)
        yo = y[np.ix_(rows, np.flatnonzero(mask))]
        tot = base[None] + np.einsum('no,op->nop', noise[np.ix_(rows, np.flatnonzero(mask))], np.eye(mask.sum()))[:, None]
        lp = log_gauss_batch(yo[:, None, :], mean[None], tot) + log_weights[None]
        lse = logsumexp(lp, axis=1); ll[rows] = lse; resp[rows] = np.exp(lp - lse[:, None])
        pco = proj @ cross                                                     # (K, q, o) = proj C H^T
        chol = np.linalg.cholesky(tot)                                         # (n, K, o, o)
        resid = (yo[:, None, :] - mean[None])[..., None]
        a1 = np.linalg.solve(chol, resid)                                      # L^-1 r
        a2 = np.linalg.solve(chol, np.broadcast_to(pco.swapaxes(-1, -2)[None], (len(rows),) + pco.swapaxes(-1, -2).shape))
        ez[rows] = pm[None] + (a2.swapaxes(-1, -2) @ a1)[..., 0]
        cz[rows] = (pc @ proj.T)[None] - a2.swapaxes(-1, -2) @ a2
    return ll, resp, ez, cz


def fit_extension(mix: GaussianMixture, y, noise, observed, *, h, offset, extra, t_d, r_index: int,
                  new_labels: tuple[str, ...], groups: list[np.ndarray], pooling: np.ndarray | None = None,
                  kappa_a: float = 10., kappa_b: float = np.inf, kappa_w: float = np.inf,
                  init: Extension | None = None, w_floor: float = 1e-4, n_iter: int = 100, tol: float = 1e-4,
                  row_weights=None, log=None) -> tuple[Extension, list[float]]:
    """EM for the new block with the old block frozen (see module docstring).

    groups: component index arrays; each group has one regression (alpha_g, B_g, W_g) fitted on its pooled
    statistics (``pooling[g, h]`` weights the statistics of group h in group g's fit; default identity).
    Per component: a_k = alpha_g + B_g mu_k,d + shrunk offset (kappa_a), B_k -> B_g (kappa_b), W_k -> W_g (kappa_w).
    ``w_floor`` (mag^2) is added to the diagonal of each W as a weak inverse-Wishart prior scale.
    Returns the extension and the mean log-likelihood per row at each iteration (non-decreasing up to the priors).
    """
    k, d = mix.means.shape; m = len(new_labels); p = t_d.shape[0]
    y = np.asarray(y, float); noise = np.asarray(noise, float); observed = np.asarray(observed, bool)
    rw = np.ones(len(y)) if row_weights is None else np.asarray(row_weights, float)
    mu_d = mix.means @ t_d.T                                                    # (K, p)
    ext = init or Extension(np.zeros((k, m)), np.zeros((k, m, p)), np.tile(np.eye(m)*0.1, (k, 1, 1)))
    # z = (d, c_E): projections from the extended latent (D + m)
    proj = np.zeros((p + m, d + m)); proj[:p, :d] = t_d
    proj[p:, d:] = np.eye(m); proj[p:, r_index] -= 1.
    ng = len(groups); pool = np.eye(ng) if pooling is None else np.asarray(pooling, float)
    history = []
    logw = np.log(mix.weights)
    for it in range(n_iter):
        mx = extend_mixture(mix, ext, t_d, r_index, new_labels)
        ll, resp, ez, cz = _estep(mx, y, noise, observed, h, offset, extra, proj, logw)
        good = np.isfinite(ll)
        history.append(float(np.average(ll[good], weights=rw[good])))
        if log:
            log(f'extension EM {it}: mean lnL {history[-1]:.5f}')
        r = resp*rw[:, None]*good[:, None]
        # per-component sufficient statistics of the augmented regressor u = (1, d) and target c
        s0 = r.sum(0)
        ed, ec = ez[..., :p], ez[..., p:]
        su = np.einsum('nk,nkp->kp', r, ed)
        sc = np.einsum('nk,nkm->km', r, ec)
        sdd = np.einsum('nk,nkp,nkq->kpq', r, ed, ed) + np.einsum('nk,nkpq->kpq', r, cz[..., :p, :p])
        scd = np.einsum('nk,nkm,nkp->kmp', r, ec, ed) + np.einsum('nk,nkmp->kmp', r, cz[..., p:, :p])
        scc = np.einsum('nk,nkm,nkl->kml', r, ec, ec) + np.einsum('nk,nkml->kml', r, cz[..., p:, p:])
        grp = np.empty(k, int)
        for gi, comp in enumerate(groups):
            grp[comp] = gi
        new_a = np.zeros_like(ext.a); new_b = np.zeros_like(ext.b); new_w = np.zeros_like(ext.w)

        def coef_stats(kk):
            g1 = np.block([[np.array([[s0[kk]]]), su[kk][None]], [su[kk][:, None], sdd[kk]]])
            r1 = np.concatenate([sc[kk][:, None], scd[kk]], axis=1)
            return g1, r1

        def resid(coef_k, kk):
            g1, r1 = coef_stats(kk)
            return scc[kk] - coef_k @ r1.T - r1 @ coef_k.T + coef_k @ g1 @ coef_k.T

        for gi, comp in enumerate(groups):
            # pooled components and their weights (kernel over groups)
            om = pool[gi, grp]; use = np.flatnonzero(om > 0); om = om[use]
            bg = ext.b[comp[0]].copy() if it > 0 else np.zeros((m, p))
            for _ in range(20):                       # coordinate ascent: exact for shared B, per-component a0
                alpha = (om[:, None]*(sc[use] - np.einsum('mp,kp->km', bg, su[use]))).sum(0)/max((om*s0[use]).sum(), 1e-12)
                if np.isinf(kappa_a):
                    a0 = np.tile(alpha, (len(use), 1))
                else:
                    a0 = (sc[use] - np.einsum('mp,kp->km', bg, su[use]) + kappa_a*alpha)/(s0[use] + kappa_a)[:, None]
                num = np.einsum('k,kmp->mp', om, scd[use] - a0[:, :, None]*su[use][:, None, :])
                den = np.einsum('k,kpq->pq', om, sdd[use]) + 1e-9*np.eye(p)
                bg = np.linalg.solve(den.T, num.T).T
            a0_of = dict(zip(use, a0))
            t0 = (om*s0[use]).sum()
            wsum = sum(o*resid(np.concatenate([a0_of[kk][:, None], bg], axis=1), kk) for o, kk in zip(om, use))
            wg = (0.5*(wsum + wsum.T) + w_floor*np.eye(m))/(t0 + 1.)
            gmat = sum(o*coef_stats(kk)[0] for o, kk in zip(om, use))/max(t0, 1e-12)
            for kk in comp:
                if np.isinf(kappa_b):
                    coef_k = np.concatenate([a0_of[kk][:, None], bg], axis=1)
                else:
                    # kappa_b pseudo-rows distributed like the group's regressors, centred on the group fit
                    centre = np.concatenate([a0_of[kk][:, None], bg], axis=1)
                    g1, r1 = coef_stats(kk); lam = kappa_b*gmat
                    coef_k = np.linalg.solve(g1 + lam + 1e-9*np.eye(p + 1), (r1 + centre @ lam).T).T
                if np.isinf(kappa_w):
                    wk = wg
                else:
                    rk = resid(coef_k, kk)
                    wk = (0.5*(rk + rk.T) + kappa_w*wg + w_floor*np.eye(m))/(s0[kk] + kappa_w + 1e-12)
                new_b[kk] = coef_k[:, 1:]
                new_a[kk] = coef_k[:, 0] + coef_k[:, 1:] @ mu_d[kk]           # a_k: mean of c at d = mu_k,d
                new_w[kk] = 0.5*(wk + wk.T)
        ext = Extension(new_a, new_b, new_w)
        if it > 0 and abs(history[-1] - history[-2]) < tol:
            break
    ext.meta.update(iterations=len(history), mean_loglike=history[-1], kappa_a=kappa_a, kappa_b=kappa_b,
                    kappa_w=kappa_w, w_floor=w_floor, n_rows=int(len(y)))
    return ext, history
