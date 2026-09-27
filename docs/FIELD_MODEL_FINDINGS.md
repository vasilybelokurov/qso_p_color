# The field ("non-quasar") model: findings and what to do (2026-09-27)

This collects what the health checks, the ChatGPT and Codex reviews, and the
follow-up diagnostics found about the field model of the one multi-survey model,
and what to do about it. The quasar model, the scorer and the ranking statistic
are not the problem; the field model is.

## Summary

- **The ranking works** where it has been validated (Legacy DR9 south
  g, r, z, W1, W2): held-out same-z vs wrong-z AUC 0.827, quasar/star 0.981,
  quasar/galaxy 0.960, matching or beating the retired Legacy model.
- **The field model is the weak component.** It is one joint 8-component
  mixture over 41 bands, fitted to every source in 24 cones of radius 0.15°.
  76 % of those sources are fainter than any candidate we score (reference
  magnitude > 22.5), and six of the eight components, 72 % of the weight, sit at
  Legacy r ≈ 23.4–24.8. The whole range that matters, 17 ≤ r < 22.5, is
  described essentially by **one broad component** (weight 0.22, mean r = 21.4,
  σ ≈ 1.5 mag in every band).
- **Consequences, measured:** its magnitude distribution is wrong in the bands
  few field sources have (SDSS r: 3–4× too many sources predicted at r = 19–21,
  where SDSS is complete), which makes scores depend on the reference band by
  ~1 nat; its tails are the wrong shape; and it has **no sky dependence at all**,
  although the design called for one and the retired model had it.
- **Fix:** refit the field model over the candidate magnitude range only, with
  enough components (chosen by held-out likelihood and run to convergence),
  restore the sky dependence of the original design, match its source selection
  to the candidates', and keep the Student-t catch-all. Details at the end.

## 1. What the field model is today

| property | current state |
|---|---|
| training population | union of all sources detected by any of 7 surveys in 24 cones (r = 0.15°, 0.071 deg² each) at \|b\| ≥ 25°, known quasars removed; 174,567 sources, 6 cones reserved |
| magnitude range | none imposed; 76 % fainter than the candidate limit, 32 % fainter than 24 |
| morphology | none imposed (stars, galaxies, anything); photometry is PSF for SDSS, PS1, SkyMapper, AllWISE; model fluxes for Legacy; aperture for VHS |
| mixture | one joint XD mixture over 41 luptitudes, K = 8 (chosen from 4, 8, 12), fitted to 20,000 pattern-weighted rows, stopped at the 150-iteration cap without converging |
| extra fits | Legacy south g,r,z and g,r,z,W1,W2, K = 32 each, used only when every observed band is in those sets |
| sky dependence | none: field density and Σ_B are global; the scorer accepts l, b and ignores them |
| catch-all | separate Student-t "unmodelled" term, ν = 2, scale 1.5 × field covariance, η ≈ 10⁻³–5×10⁻² by band |

Component allocation (Legacy south r):

| component | weight | mean r | σ(r) |
|---|---|---|---|
| 7 | 0.22 | 21.4 | 1.55 |
| 0 | 0.06 | 23.0 | 1.17 |
| 5 | 0.31 | 23.4 | 0.59 |
| 6 | 0.08 | 23.9 | 0.67 |
| 1 | 0.25 | 24.3 | 0.37 |
| 3 | 0.02 | 24.5 | 0.32 |
| 4 | 0.03 | 24.6 | 0.74 |
| 2 | 0.03 | 24.8 | 0.16 |

## 2. Findings

### 2.1 Reference-band dependence (~1 nat for SDSS-reference candidates)

Scoring the same reserved objects with SDSS r, Legacy r or PS1 r as the reference
band: quasars change ln R by a median 0.03–0.04; **field sources shift by a
constant +0.9 nats with SDSS r, +0.1 with PS1 r**, at every SDSS magnitude from
14 to 23. It is not the unmodelled term (0.96 with it, 0.96 without) and not the
redshift interpolation (0.95 at slice centres).

The cause is the field model's marginal distribution in each reference band.
The intensity is λ = Σ_B,a(u_a) · J/m_a, with J the model's joint density and
m_a its marginal in band a; it is invariant only if Σ_B,a/m_a is the same for
every band. Counted sources against the model's prediction, per deg² per mag:

| magnitude | SDSS r counted | SDSS r model | ratio | Legacy r ratio |
|---|---|---|---|---|
| 19–20 | 1,056 | 4,076 | 0.26 | 0.61 |
| 20–21 | 2,057 | 8,613 | 0.24 | 0.66 |
| 21–22 | 4,393 | 13,951 | 0.31 | 0.98 |

SDSS is complete there (95 % point-source limit r = 22.2;
https://classic.sdss.org/dr7/), so the discrepancy is the model's: only a
minority of the field sources carry SDSS photometry, and the one broad bright
component spreads far too much density over SDSS r = 14–21. ln(0.63/0.25) ≈ 0.9
nats is the offset. SDSS r is first in the reference priority, so every
candidate with SDSS photometry is affected; the validated Legacy case is not.

### 2.2 Wrong tail shape

Bright field sources (σ < 0.03 in g and z) counted where the field model's
density is a given fraction of its peak, against its prediction:

| u_r | below 10⁻⁴ of peak: observed / model | 10⁻³–10⁻¹: observed / model |
|---|---|---|
| 19.5 | 2 / 0.3 | 89 / 140 |
| 20.0 | 4 / 0.2 | 50 / 158 |
| 20.5 | 5 / 0.2 | 51 / 185 |

Too thin far out, too fat in between. The Student-t term fixes the far tail
(predicts 4.4–5.6); it cannot remove the excess in between, which makes the
Bayes factor conservative there.

### 2.3 No sky dependence

The original design had it and the retired Legacy model implemented it: a
HEALPix hierarchy (cell → parent → global, shrinkage with pooling constant n₀),
Σ_B(m, l, b) per cell, and a local refit in the candidate's own 0.5° cone
(`fit_local_background`). Measured then, in 30 cones: at |b| > 45° the local
model was indistinguishable from the global one; at 20–45° it described the
field better by 0.1–0.2 nats per object but moved no decision; the plane was
never tested. The multi-survey model was built without any of this, and when it
became the only model the capability was lost. **This should have been flagged
when the models were consolidated, and was not.**

### 2.4 Source population not matched to the candidates

The field model contains every source type. For the retired model this was a
deliberate, measured choice: a background fitted to PSF sources only made
spectroscopic galaxies look like quasar contaminants (quasar/galaxy AUC 0.80,
10.7 % above the same-z median) because the *candidates* were not
morphology-cut; with an all-source background it was 0.96 and 0.3 %. The rule
is not "all sources" but **"the same selection as the candidates"**.

### 2.5 Smaller items

- Σ_B is built from all 24 field cones, including the six reserved ones (the
  quasar prior has been fixed; the field prior has not).
- The completeness constant C changes the ranking a little (AUC ±0.01 for a
  factor 2) and the quasar/non-quasar boundary a lot (±50 %); it is a declared
  provisional normalisation, not a measurement.
- Non-detections in catalogue surveys (AllWISE, VHS, SkyMapper, SDSS) are
  treated as uninformative; Legacy (forced photometry) is not affected.
- The luptitude error approximation changes ln BF by a median 0.02 nats at
  S/N > 10 and 0.25 below S/N 3: acceptable for ranking.

## 3. Questions answered

### Should the background contain all sources, or point sources only?

**It should contain exactly the population the candidates are drawn from.** If
candidates are restricted to point sources — reasonable, since quasars at the
redshifts of interest are unresolved and many science questions only care
about point-like contaminants — the background must be restricted the same way,
and it will then be a much better model: a sharp stellar locus plus compact
galaxies, instead of a huge faint galaxy population soaking up components. What
must not happen is the mismatch that bit the first validation: a point-source
background scored against candidates that include extended objects.

Two caveats before choosing the point-source route:

- **Morphology is not a clean quasar criterion.** In the retired validation
  91–94 % of spectroscopic quasar companions were Legacy `TYPE = PSF`, so a PSF
  cut loses 6–9 % of quasars (low-z hosts, faint objects, deblending next to a
  bright primary). 3 % of the DESI-targeted galaxies were also PSF.
- **Morphology is survey-specific and degrades at the faint end**; each survey
  needs its own definition (Legacy `TYPE`, SDSS `type`, PS1 PSF−Kron), and a
  candidate without any resolved-morphology survey cannot be cut.

Recommendation: support both, as an explicit declared selection applied
identically to the training field sample, the Σ_B counts and the candidates —
`point` (default for companion searches) and `all` (for completeness studies).

### What is the optimal catch-all background model?

The catch-all has two jobs: describe the field well where it has data, and
refuse to be decisive where it has none. The current design does the second
well (Student-t) and the first badly. A good background, in order of impact:

1. **Fit only the relevant population**: reference magnitude in the candidate
   range with a margin (e.g. 16 < u_ref < 23.5), and the declared morphology
   selection. This alone returns the components to where candidates live.
2. **Enough components, converged**: choose K by held-out likelihood over a
   grid that is not capped (e.g. 8, 16, 32, 64); several initialisations; run
   until the held-out likelihood and the validation scores stop changing (the
   current fit stops at 150 iterations unconverged).
3. **Accurate in each survey's own bands**: either a joint fit with enough data
   in every band, or dedicated fits for the common band sets (as already done
   for Legacy): SDSS, PS1, Legacy+AllWISE, … — selected by which bands are
   observed, never by values.
4. **Sky dependence** (below).
5. **A heavy-tailed catch-all** for what no component describes — the
   Student-t term, refitted after every change of the field model.
6. **Validated on the tails, not only the core**: the tail-count test of §2.2
   (observed vs predicted in bins of model density), held-out likelihood by
   magnitude and band set, and reference-band invariance (§2.1) as a
   regression test.

### Does the model depend on position on the sky?

**The design says yes; the current model does not.** Everything in §2.3 was
built for the retired Legacy model; the multi-survey model has a single global
field density and a single global Σ_B. To restore it, the choices are the ones
already made and measured once:

- **Global plus local refit** (cheapest): keep a global field model, and for a
  specific candidate refit the field density and Σ_B in its own cone. Needs the
  multi-survey equivalent of `fit_local_background` (cone queries for all
  surveys, the same selection, the same transform).
- **HEALPix hierarchy** (for survey-wide scans): field mixtures and Σ_B per
  (cell, magnitude bin) with shrinkage towards parent and global fits, as in the
  retired model — which also needs its pooling constant tuned properly (the old
  tuning routine could not see it).

At |b| > 45° the retired model's measurement says global is adequate; between
20° and 45° local helps a little; nearer the plane it is untested and expected
to matter.

## 4. What to do, in order

| priority | action | why | cost |
|---|---|---|---|
| 1 | Legacy r first in the reference priority | removes the ~1 nat SDSS-reference offset for everything with Legacy photometry; keeps the validated case | minutes; rerun subset validation and examples |
| 2 | Σ_B without the reserved cones | end-to-end held-out validation | minutes |
| 3 | Refit the field model: candidate magnitude range, declared morphology selection, K by held-out likelihood up to 32–64, converged; refit priors, Legacy fits and the Student-t term; re-validate (pairs, 127 subsets, tail counts, reference invariance) | the root cause of §2.1 and §2.2; likely improves star rejection in every survey combination | a few hours, including one WSDB pull if morphology columns are needed |
| 4 | Restore sky dependence (local refit first, hierarchy later) | design requirement; matters at low latitude | a day |
| 5 | Detection model for catalogue surveys | non-detections carry information | a day or more |
| 6 | Measured spectroscopic completeness for Σ_Q | only when probabilities, not rankings, are quoted | later |

Clustering is parked and not part of this list.
