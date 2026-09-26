# Health checks of the model (2026-09-26)

Three cheap consistency checks recommended after an external review. Scripts
and raw outputs are listed at the end; all use the shipped model, priors and
unmodelled term, and reserved objects only.

## 1. Luptitude error propagation (delta method)

The model treats each measured luptitude as Gaussian with
σ_u = |g′(f_obs)| σ_f. The truth, for Gaussian flux errors, is Gaussian in flux.

**Per band** (`scripts/check_luptitude_errors.py`): median |Δ ln L| 0.02–0.05 nats
at low S/N, 0.003–0.01 at S/N > 10; 99th percentile 1–2 nats below S/N 10, with
rare much larger values below S/N 1.

**On the Bayes factor** (`scripts/check_luptitude_bf.py`), which is what matters —
much of the per-band error is common to both hypotheses. 300 reserved objects,
Legacy south g, r, z, W1, W2, ln BF computed with the exact flux-space noise and
with the delta method by the same importance-sampling integrator (which agrees
with the closed form to 0.005 nats median, 0.02 max):

| faintest band S/N | n | median \|Δ ln BF\| | 90th percentile |
|---|---|---|---|
| < 3 | 124 | 0.25 | 1.5 |
| 3–10 | 85 | 0.10 | 0.37 |
| > 10 | 91 | 0.02 | 0.07 |

All objects: median Δ −0.05, median |Δ| 0.09, max 6.4; the sign of ln BF changes
for 3 of 300. **Verdict:** negligible for bright objects, a few tenths of a nat
typical for faint ones, occasionally ~1 nat. Acceptable for ranking; a
flux-space likelihood would remove it if faint objects become the focus.

## 2. Reference-band invariance

In an exact model the joint intensity does not depend on which measured band is
the reference. `scripts/check_reference_invariance.py` scores 400 reserved
quasars (at their own z) and 400 reserved field sources (random z) that have
SDSS, Legacy south and PS1 r, with each as reference.

| | quasars: median \|Δ ln R\| | field: median Δ ln R | ρ(ln R) |
|---|---|---|---|
| SDSS r − Legacy r | 0.04 | **+0.94** | ≥ 0.99 |
| PS1 r − Legacy r | 0.03 | +0.14 | ≥ 0.99 |
| PS1 r − SDSS r | 0.04 | −0.81 | ≥ 0.99 |

ln BF moves by 0.2–0.7 in both populations; that is expected, because the Bayes
factor is conditional on the reference and is not meant to be invariant.

**Quasars are invariant; field sources are not, by ~1 nat when SDSS r is the
reference.** The offset is constant in magnitude (+0.9 to +1.1 from SDSS r = 14 to
23), so it is not a detection-limit effect. The model itself is coherent — its
joint field density agrees across references to 0.04 nats. The mismatch is in the
per-band priors: Σ_B,a counts field sources *measured in band a* (SDSS measures
12,303 deg⁻² of them for 17 ≤ r < 22.5 against Legacy's 18,945 — quality flag,
partial cone coverage, depth), while p(rest | u_a, B) is the joint model of *all*
field sources. The two agree only if every band measures the same fraction of the
field. Within the field population the ranking is unchanged (ρ = 1.00); what
shrinks is the quasar–field gap, by ~0.9 nats for SDSS-reference candidates.

**Verdict:** a real, moderate inconsistency affecting candidates whose reference
is SDSS r (first in the current priority order). The pair validation (Legacy
reference) is unaffected. Cheap mitigation: put Legacy r first in the priority.
Principled fix: Σ_B,a(u_a) = (N_field/A) p_model(u_a | B), and likewise for Σ_Q,
which is exactly invariant — but it needs the detection model of the
non-detection item, so both belong together.

## 3. Sensitivity to the completeness constant

The pair validation rescored with Σ_Q scaled by 0.5 and 2 (baseline C = 2.37),
and with its redshift shape tilted by (1+z)^±1 at fixed total:

| variant | AUC same/wrong z by ln R | AUC by p(z∈W\|Q) | quasar/non-quasar | ρ(ln R) vs baseline | median Δ log₁₀ p_same (same-z) | non-quasars called quasar |
|---|---|---|---|---|---|---|
| baseline | 0.827 | 0.864 | 0.965 | 1 | 0 | 134 |
| × 0.5 | 0.816 | 0.864 | 0.965 | 0.999 | −0.08 | 85 |
| × 2 | 0.837 | 0.864 | 0.965 | 0.999 | +0.05 | 222 |
| tilt (1+z)^+1 | 0.829 | 0.866 | 0.965 | 0.999 | −0.00 | 138 |
| tilt (1+z)^−1 | 0.823 | 0.859 | 0.965 | 0.999 | −0.01 | 135 |

**Verdict:** the ranking is robust (ρ ≥ 0.999, AUC ±0.01 for a factor of two).
Absolute quantities are not: the number of objects with P(quasar) > ½ changes by
±50 %. C matters for probabilities and class boundaries, not for ranking.

## Scripts and outputs

- `scripts/check_luptitude_errors.py` → `docs/HEALTH_luptitude.json`
- `scripts/check_luptitude_bf.py` → `docs/HEALTH_luptitude_bf.json`
- `scripts/check_reference_invariance.py` → `docs/HEALTH_reference.json`
- completeness variants: `scripts/validate_pairs_multisurvey.py --priors <scaled>`
  → `docs/HEALTH_completeness.json`
