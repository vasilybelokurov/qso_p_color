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


def _chunk_stats(mix_ext: GaussianMixture, y, noise, observed, h, offset, extra, proj, p, log_weights, rw):
    """E-step for one chunk of rows against one set of components, reduced to sufficient statistics.

    y, noise: (N, n_native) luptitudes and noise variances (unobserved entries ignored); log_weights (N, K) or (K,).
    Returns (sum of row-weighted log-likelihoods, total row weight, s0, su, sc, sdd, scd, scc) for the K components,
    where z = proj @ latent = (d, c) and moments are posterior moments given each row's observed bands.
    """
    kk = mix_ext.n_components; q = proj.shape[0]; m = q - p
    s0 = np.zeros(kk); su = np.zeros((kk, p)); sc = np.zeros((kk, m))
    sdd = np.zeros((kk, p, p)); scd = np.zeros((kk, m, p)); scc = np.zeros((kk, m, m))
    lw_all = np.broadcast_to(log_weights, (len(y), kk))
    tot_ll, tot_w = 0., 0.
    keys, inverse = np.unique(observed, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    pc = proj @ mix_ext.covs                                                   # (K, q, D+m)
    pm = mix_ext.means @ proj.T                                                # (K, q)
    pcp = pc @ proj.T                                                          # (K, q, q)
    for g, mask in enumerate(keys):
        rows = np.flatnonzero(inverse == g)
        if not mask.any() or not len(rows):
            continue
        cols = np.flatnonzero(mask); ho = h[cols]
        mean = mix_ext.means @ ho.T + offset[cols]                             # (K, o)
        cross = mix_ext.covs @ ho.T                                            # (K, D+m, o)
        base = ho @ cross + extra[np.ix_(cols, cols)]                          # (K, o, o)
        pco_t = (proj @ cross).swapaxes(-1, -2)                                # (K, o, q)
        yo = y[np.ix_(rows, cols)]; no = noise[np.ix_(rows, cols)]
        tot = base[None] + no[:, None, :, None]*np.eye(len(cols))[None, None]  # (n, K, o, o)
        chol = np.linalg.cholesky(tot)
        resid = yo[:, None, :] - mean[None]
        a1 = np.linalg.solve(chol, resid[..., None])[..., 0]                   # (n, K, o)
        maha = np.einsum('nko,nko->nk', a1, a1)
        logdet = 2*np.log(np.einsum('nkii->nki', chol)).sum(-1)
        lp = -0.5*(len(cols)*np.log(2*np.pi) + logdet + maha) + lw_all[rows]
        lse = logsumexp(lp, axis=1); resp = np.exp(lp - lse[:, None])*rw[rows, None]
        tot_ll += float(np.dot(rw[rows], lse)); tot_w += float(rw[rows].sum())
        a2 = np.linalg.solve(chol, np.broadcast_to(pco_t[None], (len(rows),) + pco_t.shape))   # (n, K, o, q)
        ez = pm[None] + np.einsum('nkoq,nko->nkq', a2, a1)
        cz = pcp[None] - np.einsum('nkoq,nkor->nkqr', a2, a2)
        ed, ec = ez[..., :p], ez[..., p:]
        s0 += resp.sum(0)
        su += np.einsum('nk,nkp->kp', resp, ed); sc += np.einsum('nk,nkm->km', resp, ec)
        sdd += np.einsum('nk,nkp,nkq->kpq', resp, ed, ed) + np.einsum('nk,nkpq->kpq', resp, cz[..., :p, :p])
        scd += np.einsum('nk,nkm,nkp->kmp', resp, ec, ed) + np.einsum('nk,nkmp->kmp', resp, cz[..., p:, :p])
        scc += np.einsum('nk,nkm,nkl->kml', resp, ec, ec) + np.einsum('nk,nkml->kml', resp, cz[..., p:, p:])
    return tot_ll, tot_w, s0, su, sc, sdd, scd, scc


def fit_extension(mix: GaussianMixture, y, noise, observed, *, h, offset, extra, t_d, r_index: int,
                  new_labels: tuple[str, ...], groups: list[np.ndarray], pooling: np.ndarray | None = None,
                  kappa_a: float = 10., kappa_b: float = np.inf, kappa_w: float = np.inf,
                  init: Extension | None = None, w_floor: float = 1e-4, n_iter: int = 100, tol: float = 1e-4,
                  row_weights=None, blocks=None, row_log_weights=None, memory: float = 3e8, b_zero: bool = False,
                  log=None) -> tuple[Extension, list[float]]:
    """EM for the new block with the old block frozen (see module docstring).

    groups: component index arrays; each group has one regression (alpha_g, B_g, W_g) fitted on its pooled
    statistics (``pooling[g, h]`` weights the statistics of group h in group g's fit; default identity).
    Per component: a_k = alpha_g + B_g mu_k,d + shrunk offset (kappa_a), B_k -> B_g (kappa_b), W_k -> W_g (kappa_w).
    ``w_floor`` (mag^2) is added to the diagonal of each W as a weak inverse-Wishart prior scale.
    ``blocks``: list of (row indices, component indices); each row is scored only against its block's components
    (a row may appear in several blocks, e.g. two neighbouring redshift slices, with ``row_weights`` splitting it).
    Default: one block, all rows and components. ``row_log_weights`` (N, K) replaces log mixture weights per row
    (e.g. sky-dependent background weights). ``memory`` bounds the bytes of one E-step chunk. ``b_zero``: B = 0
    (new colours independent of the regressors within a component; candidate (i) of the plan).
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
    with np.errstate(divide='ignore'):
        logw = np.log(mix.weights)
    blocks_ = [(np.arange(len(y)), np.arange(k))] if blocks is None else [(np.asarray(r), np.asarray(c)) for r, c in blocks]
    for it in range(n_iter):
        mx = extend_mixture(mix, ext, t_d, r_index, new_labels)
        s0 = np.zeros(k); su = np.zeros((k, p)); sc = np.zeros((k, m))
        sdd = np.zeros((k, p, p)); scd = np.zeros((k, m, p)); scc = np.zeros((k, m, m))
        tot_ll = tot_w = 0.
        for rows_b, comp_b in blocks_:
            sub = GaussianMixture(np.full(len(comp_b), 1/len(comp_b)), mx.means[comp_b], mx.covs[comp_b])
            o = max(int(observed[rows_b].sum(1).max()), 1)
            step = max(16, int(memory/(8*len(comp_b)*(o*o + 2*o*(p + m) + (p + m)**2))))
            for lo in range(0, len(rows_b), step):
                rr = rows_b[lo:lo + step]
                lw = (logw[comp_b] if row_log_weights is None else row_log_weights[np.ix_(rr, comp_b)])
                res = _chunk_stats(sub, y[rr], noise[rr], observed[rr], h, offset, extra, proj, p, lw, rw[rr])
                tot_ll += res[0]; tot_w += res[1]
                for acc, val in zip((s0, su, sc, sdd, scd, scc), res[2:]):
                    acc[comp_b] += val
        history.append(tot_ll/max(tot_w, 1e-300))
        if log:
            log(f'extension EM {it}: mean lnL {history[-1]:.5f}')
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
            ka = kappa_a[gi] if np.ndim(kappa_a) else kappa_a
            for _ in range(20):                       # coordinate ascent: exact for shared B, per-component a0
                alpha = (om[:, None]*(sc[use] - np.einsum('mp,kp->km', bg, su[use]))).sum(0)/max((om*s0[use]).sum(), 1e-12)
                if np.isinf(ka):
                    a0 = np.tile(alpha, (len(use), 1))
                else:
                    a0 = (sc[use] - np.einsum('mp,kp->km', bg, su[use]) + ka*alpha)/(s0[use] + ka)[:, None]
                if b_zero:
                    bg = np.zeros((m, p)); continue
                num = np.einsum('k,kmp->mp', om, scd[use] - a0[:, :, None]*su[use][:, None, :])
                den = np.einsum('k,kpq->pq', om, sdd[use]) + 1e-9*np.eye(p)
                bg = np.linalg.solve(den.T, num.T).T
            a0_of = dict(zip(use, a0))
            t0 = (om*s0[use]).sum()
            wsum = sum(o*resid(np.concatenate([a0_of[kk][:, None], bg], axis=1), kk) for o, kk in zip(om, use))
            wg = (0.5*(wsum + wsum.T) + w_floor*np.eye(m))/(t0 + 1.)
            gmat = sum(o*coef_stats(kk)[0] for o, kk in zip(om, use))/max(t0, 1e-12)
            for kk in comp:
                if np.isinf(kappa_b) or b_zero:
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
    ext.meta.update(b_zero=bool(b_zero), iterations=len(history), mean_loglike=history[-1], kappa_a=np.asarray(kappa_a).tolist(), kappa_b=kappa_b,
                    kappa_w=kappa_w, w_floor=w_floor, n_rows=int(len(y)))
    return ext, history
