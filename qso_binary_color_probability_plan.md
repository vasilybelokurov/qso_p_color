# QSO companion color-probability project: implementation specification

## 0. Purpose

Build a reproducible Python package that scores a photometric companion to a spectroscopically confirmed primary QSO.

The package must answer three different questions and must not conflate them:

1. **QSO-locus compatibility at the primary redshift**

   Estimate the likelihood

   \[
   p(\mathbf{x}_{\rm obs}\mid Q,z_0)
   \]

   and the quasar-only redshift posterior

   \[
   p(z\mid \mathbf{x}_{\rm obs},Q),
   \]

   where \(Q\) means "the object is a QSO", \(z_0\) is the primary spectroscopic redshift, and \(\mathbf{x}_{\rm obs}\) contains the companion photometric shape information.

   This is **not** yet a probability that the object is a QSO. It is a class-conditional likelihood or a quasar-only redshift posterior.

2. **QSO-vs-star posterior including the observed stellar population**

   Estimate a posterior of the form

   \[
   P(Q,z\sim z_0\mid \mathbf{x}_{\rm obs},m,l,b)
   \]

   by comparing the QSO model with a stellar/background model that depends on color, apparent magnitude, and Galactic position.

3. **Recommended full same-redshift posterior**

   For a binary-QSO search, also include field QSOs at other redshifts as a competing hypothesis:

   \[
   H_{\rm sameQ},\quad H_{\rm fieldQ},\quad H_{\rm star},
   \]

   with an optional compact-galaxy/background class. Otherwise, a perfectly real QSO at the wrong redshift can be incorrectly counted as evidence for a same-redshift companion.

The package must report both prior-independent evidence measures (likelihoods and Bayes factors) and prior-dependent posteriors.

Do not hard-code scientific thresholds, a redshift window, a velocity window, HEALPix resolution, magnitude bins, number of mixture components, or posterior selection cuts. Those must be configuration parameters or chosen by cross-validation.

---

## 1. Scientific definitions

### 1.1 Inputs for one candidate

Required:

- primary spectroscopic redshift `z_primary`;
- candidate sky coordinates;
- candidate Galactic longitude and latitude `(l, b)`, derived from sky coordinates;
- candidate photometry in one explicitly identified photometric system;
- flux or magnitude uncertainties;
- a reference magnitude or reference-band flux;
- metadata sufficient to identify survey, release, and photometric system.

Strongly recommended:

- primary redshift uncertainty;
- angular separation from the primary;
- morphology / PSF-vs-extended information;
- image-quality and blending diagnostics;
- Gaia astrometry when available;
- WISE W1/W2 forced photometry when scientifically appropriate and not compromised by blending.

### 1.2 Main hypotheses

Define:

- `H_sameQ`: candidate is a QSO with redshift consistent with the primary according to the configured redshift-match definition;
- `H_fieldQ`: candidate is a QSO, but outside the configured redshift-match interval/kernel;
- `H_star`: candidate is a star;
- `H_gal`: optional compact galaxy / non-QSO extragalactic contaminant;
- `H_bad`: optional artifact / pathological photometry class, or otherwise a quality flag rather than a probabilistic class.

The minimum requested two-class calculation is `H_sameQ` versus `H_star`.

The recommended binary-QSO calculation is `H_sameQ` versus `H_fieldQ + H_star`, optionally plus `H_gal`.

### 1.3 Redshift-match definition

Do not bake in an arbitrary `Delta z`.

Support at least these configurable definitions:

1. `mode: dz_window`
   - user supplies `dz_half_width`;

2. `mode: velocity_window`
   - user supplies a rest-frame velocity half-width;
   - for small velocity differences use the configured conversion consistently, e.g.
     \[
     \Delta v \simeq c\,\frac{\Delta z}{1+z_0};
     \]

3. `mode: kernel`
   - user supplies a normalized kernel \(K_{\rm pair}(z\mid z_0)\);
   - the kernel may include primary redshift uncertainty and any desired physical pair velocity model.

The code must store the selected definition in every output table and model manifest.

---

## 2. Core statistical factorization

The cleanest implementation separates:

1. a **shape likelihood**, describing colors/relative fluxes at fixed redshift and magnitude; and
2. a **surface-density prior/intensity**, describing how many objects of each class exist at the relevant magnitude and sky position.

Let \(\mathbf{c}\) denote color/SED-shape features, \(m\) a reference magnitude, and \(h(l,b)\) a sky cell or smooth sky coordinate representation.

Use

\[
p(\mathbf{c}_{\rm obs}\mid Q,z,m)
\]

for the QSO color likelihood and

\[
p(\mathbf{c}_{\rm obs}\mid S,m,h)
\]

for the stellar color likelihood.

Let

\[
\Sigma_Q(z,m)
\]

be the expected QSO surface density per redshift and magnitude element after applying the relevant survey selection/completeness model, and

\[
\Sigma_S(m,h)
\]

the stellar surface density at magnitude \(m\) and sky location \(h\).

Then define the same-redshift QSO intensity

\[
\lambda_{\rm sameQ}
=
\int_{\mathcal{W}(z_0)}
\Sigma_Q(z,m)\,
p(\mathbf{c}_{\rm obs}\mid Q,z,m)\,dz,
\]

where \(\mathcal{W}(z_0)\) is the configured redshift-match region or an equivalent kernel-weighted integral.

Define

\[
\lambda_{\rm star}
=
\Sigma_S(m,h)\,
p(\mathbf{c}_{\rm obs}\mid S,m,h).
\]

If field QSOs are included,

\[
\lambda_{\rm fieldQ}
=
\int_{z\notin \mathcal{W}(z_0)}
\Sigma_Q(z,m)\,
p(\mathbf{c}_{\rm obs}\mid Q,z,m)\,dz.
\]

The requested star-only posterior is

\[
P_{\rm sameQ,star}
=
\frac{\lambda_{\rm sameQ}}
{\lambda_{\rm sameQ}+\lambda_{\rm star}}.
\]

The recommended posterior is

\[
P_{\rm sameQ}
=
\frac{\lambda_{\rm sameQ}}
{\lambda_{\rm sameQ}
+\lambda_{\rm fieldQ}
+\lambda_{\rm star}
+\lambda_{\rm gal}},
\]

omitting terms for classes that are not modeled.

Also report the prior-independent Bayes factor

\[
{\rm BF}_{Qz/S}
=
\frac{p(\mathbf{c}_{\rm obs}\mid Q,z\sim z_0,m)}
{p(\mathbf{c}_{\rm obs}\mid S,m,h)}.
\]

This separation is essential because the posterior can change when the QSO or stellar number-density prior changes, while the color evidence does not.

---

## 3. Stage 1: QSO color-redshift model

### 3.1 Recommended baseline: XDQSOz-like generative model

Use an error-deconvolved Gaussian mixture model inspired by XDQSO/XDQSOz.

For each photometric system and reference-magnitude regime, fit a mixture to the latent vector

\[
\mathbf{v} =
\begin{bmatrix}
\mathbf{c} \\
z
\end{bmatrix}
\]

for spectroscopically confirmed QSOs:

\[
p(\mathbf{v}\mid Q,m)
=
\sum_{k=1}^{K}
\alpha_k\,
\mathcal{N}(\mathbf{v}\mid \boldsymbol{\mu}_k,\mathbf{V}_k).
\]

Fit the intrinsic mixture using Extreme Deconvolution (XD), supplying each training object's heteroscedastic photometric covariance and redshift uncertainty.

The number of components `K` must be selected by held-out predictive likelihood, not fixed in source code.

Magnitude can be handled in v1 by overlapping magnitude bins with interpolation between neighboring models. A future implementation may replace this with a continuously conditional mixture model.

### 3.2 Why XD is the preferred v1 baseline

The QSO locus is multimodal and changes strongly with redshift. Photometric uncertainties are object-dependent and can be correlated after converting bands into colors. XD provides a direct generative density whose evaluation can be analytically convolved with candidate errors.

A major practical benefit is that a fitted Gaussian mixture permits analytic:

- \(p(\mathbf{c}_{\rm obs}\mid Q,z_0,m)\);
- \(p(z\mid \mathbf{c}_{\rm obs},Q,m)\);
- integration over a redshift interval;
- marginalization over missing photometric dimensions.

### 3.3 Analytic conditional likelihood from the fitted mixture

For component \(k\), partition the intrinsic mean and covariance as

\[
\boldsymbol{\mu}_k
=
\begin{bmatrix}
\boldsymbol{\mu}_{c,k}\\
\mu_{z,k}
\end{bmatrix},
\qquad
\mathbf{V}_k
=
\begin{bmatrix}
\mathbf{V}_{cc,k} & \mathbf{V}_{cz,k}\\
\mathbf{V}_{zc,k} & V_{zz,k}
\end{bmatrix}.
\]

Let the candidate color covariance be \(\mathbf{S}_c\).

At fixed redshift \(z_0\), component \(k\) gives

\[
\boldsymbol{\mu}_{c\mid z,k}
=
\boldsymbol{\mu}_{c,k}
+
\mathbf{V}_{cz,k}V_{zz,k}^{-1}
(z_0-\mu_{z,k}),
\]

\[
\mathbf{V}_{c\mid z,k}
=
\mathbf{V}_{cc,k}
-
\mathbf{V}_{cz,k}V_{zz,k}^{-1}\mathbf{V}_{zc,k}.
\]

The observed-color covariance is

\[
\mathbf{C}_{c\mid z,k}
=
\mathbf{V}_{c\mid z,k}+\mathbf{S}_c.
\]

The component weights at fixed redshift are

\[
w_k(z_0)
\propto
\alpha_k\,
\mathcal{N}(z_0\mid\mu_{z,k},V_{zz,k}),
\]

normalized over components.

Then

\[
p(\mathbf{c}_{\rm obs}\mid Q,z_0,m)
=
\sum_k
w_k(z_0)\,
\mathcal{N}
(
\mathbf{c}_{\rm obs}
\mid
\boldsymbol{\mu}_{c\mid z,k},
\mathbf{C}_{c\mid z,k}
).
\]

Implement this directly with `scipy.linalg` and `scipy.special.logsumexp`. All public likelihood functions must work in log space.

### 3.4 Analytic QSO-only photometric redshift PDF

For a candidate with observed colors \(\mathbf{c}_{\rm obs}\), define

\[
\mathbf{C}_{c,k}
=
\mathbf{V}_{cc,k}+\mathbf{S}_c.
\]

The posterior component weight given the observed colors is

\[
w_k(\mathbf{c}_{\rm obs})
\propto
\alpha_k
\mathcal{N}
(
\mathbf{c}_{\rm obs}
\mid
\boldsymbol{\mu}_{c,k},
\mathbf{C}_{c,k}
).
\]

For each component,

\[
\mu_{z\mid c,k}
=
\mu_{z,k}
+
\mathbf{V}_{zc,k}
\mathbf{C}_{c,k}^{-1}
(
\mathbf{c}_{\rm obs}-\boldsymbol{\mu}_{c,k}
),
\]

\[
V_{z\mid c,k}
=
V_{zz,k}
-
\mathbf{V}_{zc,k}
\mathbf{C}_{c,k}^{-1}
\mathbf{V}_{cz,k}.
\]

Therefore

\[
p(z\mid\mathbf{c}_{\rm obs},Q,m)
=
\sum_k
w_k(\mathbf{c}_{\rm obs})
\mathcal{N}
(
z\mid\mu_{z\mid c,k},V_{z\mid c,k}
).
\]

This makes the probability of a redshift match under the QSO hypothesis analytic:

\[
P(z\in\mathcal{W}(z_0)\mid\mathbf{c}_{\rm obs},Q,m)
=
\int_{\mathcal{W}(z_0)}
p(z\mid\mathbf{c}_{\rm obs},Q,m)\,dz.
\]

For a top-hat redshift window, evaluate the integral with the Gaussian CDF component by component.

For a custom pair kernel, integrate numerically in one dimension or use a closed form when the configured kernel permits it.

### 3.5 Stage-1 outputs

For every candidate write:

- `loglike_qso_at_zprimary`;
- `p_zmatch_given_qso`;
- a compact representation of `p(z | photometry, Q)` or a saved grid;
- `z_phot_mode_qso`;
- `z_phot_summary` fields chosen by the implementation;
- QSO-model out-of-distribution score;
- photometric quality flags.

Do **not** name `loglike_qso_at_zprimary` a "QSO probability".

---

## 4. Photometric representation and errors

### 4.1 Never mix survey systems silently

Train separate models for distinct photometric systems unless an explicitly validated transformation is applied.

Examples:

- SDSS `ugriz`;
- Legacy Surveys / DECaLS-like systems;
- Legacy North versus South when filter systems differ;
- optical-only versus optical+WISE models.

The model manifest must record:

- survey/release;
- bands;
- filter-system identifier;
- extinction treatment;
- feature transform;
- magnitude definition;
- training catalog provenance.

### 4.2 Preferred inputs: calibrated fluxes plus covariance

Keep raw calibrated flux and inverse-variance columns in the canonical internal table.

For Legacy Surveys, use the catalog `flux_*`, `flux_ivar_*`, and `mw_transmission_*` columns. Build dereddened fluxes consistently, and propagate the transmission correction into the flux variance.

Do not discard a band only because the measured flux is negative. Negative noisy flux values are valid measurements in linear flux space.

### 4.3 Color/shape feature options

Implement a pluggable `FeatureTransform` interface.

#### Option A: relative linear fluxes

Given a reference band \(a\),

\[
r_j = \frac{f_j}{f_a}.
\]

Propagate the full covariance with the Jacobian

\[
\mathbf{C}_r
=
\mathbf{J}\mathbf{C}_f\mathbf{J}^{T}.
\]

This is close in spirit to XDQSO-like flux-space modeling.

Because the denominator can become unstable at low reference-band signal-to-noise, the code must detect this and either:

- route the object to a low-S/N-safe transform;
- use a direct flux-space model;
- or return a quality state showing that the configured feature transform is not valid.

Do not silently clip or replace the reference flux.

#### Option B: asinh/luptitude colors

Support a fixed, survey-specific asinh transform so non-detections and negative fluxes remain finite.

The softening/scaling parameters must be stored in the model configuration and derived from a declared procedure, not hidden constants.

#### Option C: direct flux-space model

Support a future model in which a latent amplitude is marginalized rather than dividing by a noisy reference flux.

This is the most rigorous low-S/N extension but need not block the v1 implementation.

### 4.4 Color covariance

If ordinary magnitudes are used, never assume color errors are independent.

For magnitude vector \(\mathbf{m}\) with covariance \(\mathbf{C}_m\), and color transform

\[
\mathbf{c} = \mathbf{A}\mathbf{m},
\]

use

\[
\mathbf{C}_c
=
\mathbf{A}\mathbf{C}_m\mathbf{A}^{T}.
\]

Shared bands produce off-diagonal terms automatically.

### 4.5 Missing bands

A Gaussian-mixture likelihood can marginalize missing dimensions exactly.

At inference time:

1. construct the index set of observed/usable feature dimensions;
2. subset every mixture mean and covariance to those dimensions;
3. subset the candidate covariance identically;
4. evaluate the marginalized density.

Do not impute missing colors with a survey median for the main probability calculation.

### 4.6 Blending is a first-class issue for close companions

Close QSO companions are exactly where catalog photometry can be least trustworthy.

For Legacy Survey data, preserve and inspect available diagnostics such as:

- source morphology/type;
- `fracflux_*`;
- `fracin_*`;
- `fracmasked_*`;
- `rchisq_*`;
- relevant mask bits.

The scoring code should produce a photometric-quality flag when deblending contamination is suspicious.

For the closest systems, provide an extension point for image-level or forced multi-source photometry. A high posterior from contaminated colors must not be treated as equivalent to a high posterior from clean photometry.

WISE information can be highly useful for QSO/star separation, but its much broader PSF makes close-pair blending especially important. Keep an optical-only model available even when an optical+WISE model exists.

---

## 5. Training data

### 5.1 QSO sample

Use spectroscopically confirmed QSOs, crossmatched into the exact imaging data release used for the candidates.

Potential sources include:

- SDSS DR16Q for a large confirmed QSO sample;
- DESI DR1 spectroscopic QSOs, crossmatched to the desired Legacy Surveys release;
- other spectroscopic QSO catalogs with known provenance.

Requirements:

- deduplicate by astrophysical object, not observation;
- keep the best available spectroscopic/systemic redshift and its uncertainty;
- record the original selection/targeting channel;
- prevent the same astrophysical object from appearing in both train and validation folds;
- remove or flag clearly bad photometry;
- retain a reproducible table of all cuts.

### 5.2 QSO training-selection bias

Spectroscopic QSO catalogs are not automatically representative training sets for intrinsic QSO colors because QSO targeting itself often uses color, magnitude, morphology, variability, radio, or IR information.

Therefore:

- use the spectroscopic sample primarily to learn the conditional color-redshift locus;
- use targeting/completeness weights when reliable selection functions are available;
- combine multiple QSO selection channels when possible;
- validate against QSO subsets selected by substantially different methods;
- consider simulated/synthetic QSO photometry to test color regions with poor spectroscopic coverage.

Do not estimate the QSO class prior by taking the raw QSO fraction of a spectroscopic targeting catalog.

### 5.3 Stellar sample

Implement two selectable strategies.

#### Strategy A: explicit stellar model

Build a high-purity star sample using one or more of:

- spectroscopically confirmed stars;
- Gaia astrometric evidence;
- carefully defined point-source control samples.

A Gaia-selected sample can be very pure but is not automatically representative of the faint/distant stellar population. A spectroscopic stellar sample can have strong targeting selection. The code must preserve selection metadata and validation by magnitude and sky position.

#### Strategy B: empirical point-source background model

As a robust alternative, model all ordinary point-like background sources in the imaging catalog:

\[
p(\mathbf{c},m\mid {\rm background},l,b).
\]

This naturally includes stars and any remaining compact contaminants. It may be more directly useful for candidate ranking when a pure, representative stellar training set is difficult to construct.

Implement both modes behind a common interface.

### 5.4 Optional galaxy/background class

For faint imaging, compact galaxies can matter.

The architecture must allow an optional `H_gal` density and prior without rewriting the inference layer.

---

## 6. Stage 2: stellar color-magnitude-sky model

### 6.1 Baseline model

For each photometric system, model

\[
p(\mathbf{c}\mid S,m,l,b).
\]

Use:

- reference-magnitude bins;
- coarse HEALPix sky cells or another spherical partition;
- an XD-GMM in color/shape space within each local cell.

Do not pick the sky resolution or magnitude-bin edges arbitrarily in code. Supply candidate grids in configuration and choose the final complexity with spatially blocked validation.

### 6.2 Partial pooling / hierarchical fallback

Many local sky-magnitude cells will be sparse.

Implement a hierarchy:

1. local sky + magnitude model;
2. parent/coarser sky + magnitude model;
3. global magnitude-dependent stellar model.

A simple implementation is

\[
p_{\rm local}
=
a\,p_{\rm cell}
+
(1-a)\,p_{\rm parent},
\]

where the shrinkage weight `a` is learned or tuned from held-out predictive performance, not hard-coded.

The output must record which hierarchy level contributed to each candidate score.

### 6.3 Stellar surface-density prior

Estimate

\[
\Sigma_S(m,l,b)
\]

from the imaging catalog itself over the same footprint and with the same photometric-quality definition used by the candidate sample.

Correct or condition on:

- usable survey area;
- masks;
- depth/completeness;
- extinction treatment;
- point-source definition.

Use area-corrected counts in sky and magnitude cells, then smooth/hierarchically pool them.

Do not derive \(\Sigma_S\) from the spectroscopic star sample unless a defensible spectroscopic selection function is included.

### 6.4 Galactic longitude and latitude

A HEALPix representation automatically handles the periodicity of Galactic longitude and the geometry of the sphere.

If a smooth conditional model is implemented later, never feed longitude as a raw discontinuous scalar alone. Use a spherical representation such as:

- unit Cartesian sky vector;
- spherical harmonics;
- a suitable periodic embedding.

---

## 7. QSO surface-density prior

The QSO posterior needs a prior/intensity term:

\[
\Sigma_Q(z,m).
\]

Support multiple interchangeable modes:

### Mode 1: selection-corrected empirical prior

Estimate \(\Sigma_Q(z,m)\) from a spectroscopic QSO sample only if the relevant targeting completeness and area are known well enough to correct the counts.

### Mode 2: luminosity-function prior

Use a published QSO luminosity function, K-correction/SED model, and the exact survey selection function to predict apparent \((z,m)\) counts.

This is scientifically attractive but substantially more implementation work.

### Mode 3: calibrated effective prior

Treat the QSO prior as a calibration function learned on a representative labeled validation sample.

This can be practical for target ranking, but it must be labeled clearly as an empirically calibrated prior for that selection function.

### Required rule

Always save and report the Bayes factor independently of the chosen prior mode.

---

## 8. Recommended full posterior for a binary-QSO search

For a companion to a known QSO, compute:

\[
\lambda_{\rm sameQ}
=
\int_{\mathcal{W}(z_0)}
\Sigma_Q(z,m)
p(\mathbf{c}_{\rm obs}\mid Q,z,m)\,dz,
\]

\[
\lambda_{\rm fieldQ}
=
\int_{z\notin\mathcal{W}(z_0)}
\Sigma_Q(z,m)
p(\mathbf{c}_{\rm obs}\mid Q,z,m)\,dz,
\]

\[
\lambda_{\rm star}
=
\Sigma_S(m,l,b)
p(\mathbf{c}_{\rm obs}\mid S,m,l,b).
\]

Then

\[
P_{\rm sameQ}
=
\frac{\lambda_{\rm sameQ}}
{\lambda_{\rm sameQ}
+\lambda_{\rm fieldQ}
+\lambda_{\rm star}}.
\]

If an explicit galaxy class is available, add \(\lambda_{\rm gal}\) to the denominator.

### 8.1 Separate "same-redshift QSO" from "physically associated binary"

`P_sameQ` is a photometric/classification probability under the adopted population priors.

It is not yet the posterior probability of a physically associated binary.

A physical-pair model should additionally condition the QSO prior on angular/projected separation from a known QSO and on small-scale QSO clustering:

\[
\Sigma_{Q\mid {\rm primary}}(z,m,\theta)
\neq
\Sigma_Q(z,m).
\]

Implement this only as a later optional layer. The pair/clustering prior must come from a measurement or explicitly specified model matched to the selection function. Do not invent an enhancement factor.

---

## 9. Calibration and validation

### 9.1 Data splitting

Use spatially blocked cross-validation, preferably with HEALPix groups, so nearby sky regions do not leak into both train and validation sets.

Also stratify or diagnose performance by:

- redshift;
- reference magnitude;
- Galactic latitude;
- survey depth;
- photometric S/N;
- morphology;
- blend/quality state.

### 9.2 Stage-1 validation

For confirmed QSOs:

- held-out log predictive density;
- coverage of the QSO-only redshift PDF;
- probability integral transform diagnostics;
- fraction of held-out QSOs whose spectroscopic redshift lies inside nominal credible regions;
- performance versus redshift and magnitude.

### 9.3 Stage-2 validation

On a representative labeled test sample:

- Brier score;
- log loss;
- reliability/calibration curves;
- precision-recall curves;
- ROC only as a supplementary metric;
- completeness/purity at user-selected thresholds;
- calibration versus magnitude and \(|b|\).

Because the true class prevalence changes strongly with magnitude and sky position, validate posterior calibration in those dimensions rather than only globally.

### 9.4 Prior-shift tests

Write synthetic tests that hold class-conditional likelihoods fixed while changing class priors.

The posterior must move exactly according to Bayes' theorem, while the Bayes factor remains unchanged.

### 9.5 Selection-function stress tests

Construct validation subsets with:

- different QSO targeting channels;
- different stellar selection routes;
- different sky regions;
- different survey depths.

A model that works only on the original targeting distribution is not sufficient.

---

## 10. Suggested package layout

Create:

```text
qso-pair-prob/
|-- pyproject.toml
|-- README.md
|-- configs/
|   |-- example_legacy.yaml
|   `-- example_sdss.yaml
|-- src/
|   `-- qso_pair_prob/
|       |-- __init__.py
|       |-- config.py
|       |-- schema.py
|       |-- coordinates.py
|       |-- photometry/
|       |   |-- base.py
|       |   |-- legacy.py
|       |   |-- sdss.py
|       |   |-- extinction.py
|       |   `-- features.py
|       |-- density/
|       |   |-- base.py
|       |   |-- xdgmm.py
|       |   |-- gaussian_math.py
|       |   `-- hierarchy.py
|       |-- models/
|       |   |-- qso_redshift.py
|       |   |-- star_sky.py
|       |   |-- background_sky.py
|       |   `-- priors.py
|       |-- inference/
|       |   |-- hypotheses.py
|       |   |-- pair_score.py
|       |   `-- redshift_match.py
|       |-- training/
|       |   |-- build_qso_sample.py
|       |   |-- build_star_sample.py
|       |   |-- fit_qso.py
|       |   |-- fit_star.py
|       |   `-- fit_priors.py
|       |-- validation/
|       |   |-- splits.py
|       |   |-- calibration.py
|       |   `-- diagnostics.py
|       |-- io/
|       |   |-- catalogs.py
|       |   |-- models.py
|       |   `-- outputs.py
|       `-- cli.py
|-- tests/
|   |-- test_gaussian_conditionals.py
|   |-- test_feature_covariance.py
|   |-- test_missing_bands.py
|   |-- test_qso_redshift_pdf.py
|   |-- test_star_hierarchy.py
|   |-- test_posterior.py
|   |-- test_prior_shift.py
|   `-- test_survey_adapters.py
`-- notebooks/
    |-- 01_qso_locus_diagnostics.ipynb
    |-- 02_star_sky_diagnostics.ipynb
    |-- 03_probability_calibration.ipynb
    `-- 04_candidate_examples.ipynb
```

Use standard scientific Python dependencies:

- `numpy`
- `scipy`
- `pandas` or `polars`
- `astropy`
- `astropy-healpix` or `healpy`
- `scikit-learn`
- `astroML` for the baseline XD implementation
- `pyarrow`
- `pydantic`
- `pyyaml`
- `typer`
- `joblib`
- `matplotlib`
- `pytest`

Do not pin versions until the environment is inspected. Add tested bounds only after CI has established compatibility.

---

## 11. Core Python interfaces

### 11.1 Photometry adapter

```python
class PhotometryAdapter(Protocol):
    system_name: str
    bands: tuple[str, ...]

    def canonicalize(self, table):
        """Return calibrated fluxes, variances, transmissions, and quality metadata."""

    def feature_vector(self, row, config):
        """Return features, full covariance, observed-dimension mask, and quality flags."""
```

### 11.2 QSO model

```python
class QSOColorRedshiftModel:
    def fit(self, features, covariances, redshift, redshift_var, ref_mag, sample_weight=None):
        ...

    def loglike_at_redshift(self, features, covariance, z, ref_mag, observed_mask=None):
        """log p(features_obs | Q, z, ref_mag)."""

    def redshift_mixture(self, features, covariance, ref_mag, observed_mask=None):
        """Return analytic 1D Gaussian-mixture representation of p(z | features_obs, Q)."""

    def p_zmatch_given_qso(self, features, covariance, z_primary, ref_mag, match_definition, observed_mask=None):
        ...
```

### 11.3 Stellar/background model

```python
class StarSkyModel:
    def fit(self, features, covariances, ref_mag, l_deg, b_deg, sample_weight=None):
        ...

    def loglike(self, features, covariance, ref_mag, l_deg, b_deg, observed_mask=None):
        """log p(features_obs | star, ref_mag, l, b)."""

    def surface_density(self, ref_mag, l_deg, b_deg):
        """Sigma_star(ref_mag, l, b) in documented units."""
```

### 11.4 QSO prior

```python
class QSOPriorModel:
    def surface_density(self, z, ref_mag):
        """Sigma_Q(z, ref_mag) in documented units."""

    def integrate_surface_density(self, z_interval_or_kernel, ref_mag):
        ...
```

### 11.5 Scorer

```python
@dataclass
class PairScore:
    loglike_qso_zprimary: float
    p_zmatch_given_qso: float
    loglike_star: float | None
    log_bayes_factor_qz_star: float | None
    lambda_sameq: float | None
    lambda_fieldq: float | None
    lambda_star: float | None
    p_sameq_vs_star: float | None
    p_sameq_full: float | None
    quality_flags: tuple[str, ...]
    model_manifest_id: str
```

---

## 12. Numerically stable Gaussian-mixture code

Do not rely on repeated calls to generic multivariate-normal objects inside large loops.

Implement batched log-density evaluation using:

- Cholesky decompositions;
- triangular solves;
- log determinants from the Cholesky factor;
- `scipy.special.logsumexp`.

Never explicitly invert a covariance matrix in production code. The equations above use inverse notation only mathematically.

Regularize fitted covariance matrices through the XD fitting configuration and verify positive definiteness after serialization/deserialization.

---

## 13. Model fitting workflow

### 13.1 Build canonical QSO training table

Command target:

```bash
qso-pair-prob prepare-qso --config configs/example_legacy.yaml
```

Tasks:

1. read confirmed-QSO catalogs;
2. crossmatch to the selected imaging release;
3. deduplicate objects;
4. apply reproducible photometric quality rules;
5. deredden consistently;
6. construct flux covariance;
7. compute Galactic coordinates;
8. save target-selection metadata and sample weights;
9. assign spatial cross-validation groups;
10. write Parquet.

### 13.2 Fit QSO color-redshift model

```bash
qso-pair-prob fit-qso --config configs/example_legacy.yaml
```

Tasks:

1. split by photometric system;
2. split or smoothly weight by reference magnitude;
3. for each candidate mixture complexity, fit XD on training folds;
4. evaluate held-out predictive log density;
5. choose complexity using the configured selection rule;
6. refit on the full training subset;
7. serialize mixture parameters, feature metadata, and training manifest.

### 13.3 Build stellar/background control table

```bash
qso-pair-prob prepare-stars --config configs/example_legacy.yaml
```

Tasks:

1. create explicit-star or empirical-background sample;
2. apply the same photometric canonicalization as for candidates;
3. compute Galactic coordinates and HEALPix cell;
4. save area/depth/mask metadata needed for number-density estimation;
5. assign spatial validation groups.

### 13.4 Fit stellar/background color model

```bash
qso-pair-prob fit-stars --config configs/example_legacy.yaml
```

Tasks:

1. train local sky-magnitude XD-GMMs;
2. train parent/global fallback models;
3. tune hierarchy/shrinkage on held-out sky regions;
4. save local-model coverage and fallback metadata.

### 13.5 Fit priors

```bash
qso-pair-prob fit-priors --config configs/example_legacy.yaml
```

Produce:

- `Sigma_star(m,l,b)` or `Sigma_background(m,l,b)`;
- `Sigma_qso(z,m)`;
- optional galaxy density;
- explicit provenance and units.

### 13.6 Score candidates

```bash
qso-pair-prob score --config configs/example_legacy.yaml --input candidates.parquet --output scores.parquet
```

For each candidate:

1. canonicalize photometry;
2. compute feature vector and full covariance;
3. choose/interpolate the appropriate magnitude model;
4. evaluate QSO likelihood at `z_primary`;
5. compute the full QSO-only `p(z | photometry, Q)`;
6. integrate `p_zmatch_given_qso`;
7. evaluate local stellar/background likelihood;
8. compute color Bayes factor;
9. evaluate QSO and star surface densities;
10. compute the requested two-class posterior;
11. compute the recommended full same-redshift posterior if the field-QSO model is enabled;
12. attach quality, OOD, and blending flags;
13. write all intermediate evidence terms, not only the final posterior.

---

## 14. Configuration schema

Create a Pydantic schema and YAML config.

Do not place scientific defaults in code when they change the meaning of the probability.

Example structure:

```yaml
survey:
  name: legacy_dr10_south
  bands: [g, r, i, z]
  reference_band: r
  use_wise: false
  deredden: true
  photometric_system_id: null

features:
  mode: relative_flux
  low_snr_fallback: null
  required_bands: null

qso_training:
  catalogs: []
  imaging_catalog: null
  targeting_weight_column: null
  quality_query: null

star_training:
  mode: explicit_star
  catalogs: []
  imaging_catalog: null
  quality_query: null

qso_model:
  magnitude_edges: null
  magnitude_overlap: null
  n_components_grid: null
  max_iter: null
  tolerance: null

star_model:
  magnitude_edges: null
  healpix_nside_grid: null
  n_components_grid: null
  hierarchy_mode: local_parent_global

priors:
  qso_mode: null
  star_mode: imaging_counts
  qso_prior_source: null
  area_map: null
  depth_map: null

redshift_match:
  mode: velocity_window
  half_width_kms: null
  dz_half_width: null
  kernel_config: null

inference:
  include_field_qso: true
  include_galaxy: false
  use_astrometry: false
  use_morphology: false

validation:
  spatial_cv_nside: null
  n_folds: null
  calibration_method: null

output:
  save_redshift_pdf: true
  redshift_grid: null
```

Validation must fail loudly when a scientifically required field is `null`.

---

## 15. Pseudocode for inference

```python
def score_candidate(candidate, models, cfg):
    phot = models.photometry_adapter.canonicalize_one(candidate)

    feat = models.photometry_adapter.feature_vector(
        phot,
        config=cfg.features,
    )

    qso = models.qso_model

    log_lq_z0 = qso.loglike_at_redshift(
        features=feat.x,
        covariance=feat.cov,
        z=candidate.z_primary,
        ref_mag=feat.ref_mag,
        observed_mask=feat.mask,
    )

    p_zmatch_q = qso.p_zmatch_given_qso(
        features=feat.x,
        covariance=feat.cov,
        z_primary=candidate.z_primary,
        ref_mag=feat.ref_mag,
        match_definition=cfg.redshift_match,
        observed_mask=feat.mask,
    )

    log_ls = models.star_model.loglike(
        features=feat.x,
        covariance=feat.cov,
        ref_mag=feat.ref_mag,
        l_deg=candidate.l_deg,
        b_deg=candidate.b_deg,
        observed_mask=feat.mask,
    )

    log_bf = log_lq_z0 - log_ls

    lambda_star = models.star_model.surface_density(
        ref_mag=feat.ref_mag,
        l_deg=candidate.l_deg,
        b_deg=candidate.b_deg,
    ) * exp(log_ls)

    lambda_sameq = integrate_qso_intensity_over_match_region(
        qso_model=models.qso_model,
        qso_prior=models.qso_prior,
        candidate=candidate,
        features=feat,
        match_definition=cfg.redshift_match,
    )

    p_same_vs_star = lambda_sameq / (lambda_sameq + lambda_star)

    if cfg.inference.include_field_qso:
        lambda_fieldq = integrate_qso_intensity_outside_match_region(...)
        denom = lambda_sameq + lambda_fieldq + lambda_star
        p_same_full = lambda_sameq / denom
    else:
        lambda_fieldq = None
        p_same_full = None

    return PairScore(
        loglike_qso_zprimary=log_lq_z0,
        p_zmatch_given_qso=p_zmatch_q,
        loglike_star=log_ls,
        log_bayes_factor_qz_star=log_bf,
        lambda_sameq=lambda_sameq,
        lambda_fieldq=lambda_fieldq,
        lambda_star=lambda_star,
        p_sameq_vs_star=p_same_vs_star,
        p_sameq_full=p_same_full,
        quality_flags=feat.quality_flags,
        model_manifest_id=models.manifest_id,
    )
```

In production, perform all intensity arithmetic in log space and normalize with `logsumexp`.

---

## 16. Required tests

### 16.1 Gaussian conditional identities

Generate a synthetic joint Gaussian in \((\mathbf{c},z)\).

Verify numerically that:

- analytic `p(c | z)` matches direct evaluation of `p(c,z)/p(z)`;
- analytic `p(z | c)` matches direct evaluation;
- the mixture redshift PDF integrates to one.

### 16.2 Photometric covariance propagation

Draw Monte Carlo flux realizations from a known covariance matrix.

Verify that the empirical covariance of transformed features agrees with the Jacobian-propagated covariance in the regime where the linear approximation is expected to hold.

### 16.3 Missing-band marginalization

For a known Gaussian mixture:

- evaluate the full-dimensional density and integrate out a missing dimension numerically;
- compare against direct subspace Gaussian marginalization.

### 16.4 Error convolution

Generate latent samples from a mixture, add known Gaussian measurement noise, and verify that evaluation with `V_component + S_observation` reproduces the noisy density.

### 16.5 Posterior normalization

For arbitrary positive intensities, verify:

```text
P_sameQ + P_fieldQ + P_star (+ P_gal) = 1
```

to numerical tolerance.

### 16.6 Prior-shift invariance of Bayes factor

Change only `Sigma_Q` and `Sigma_star`.

Verify:

- posterior changes;
- `log_bayes_factor_qz_star` is unchanged.

### 16.7 Spatial leakage test

Ensure no HEALPix cross-validation group appears in both train and validation partitions.

### 16.8 Serialization regression

Save and reload every model.

Likelihoods before and after serialization must agree to a documented numerical tolerance.

### 16.9 Survey mismatch protection

Attempt to score SDSS features with a Legacy model.

The code must fail with a clear photometric-system mismatch error.

---

## 17. Diagnostics to generate automatically

### QSO model

- color-color slices in narrow spectroscopic-redshift intervals;
- fitted mixture contours over held-out QSOs;
- `p(z | photometry, Q)` examples;
- calibration/coverage of redshift PDFs;
- held-out log likelihood versus redshift and magnitude.

### Stellar model

- color-color diagrams in several Galactic sky regions;
- residuals of predicted versus observed star counts by sky and magnitude;
- local-versus-parent fallback fraction;
- held-out log likelihood versus \(|b|\) and magnitude.

### Classification

- distribution of `log BF_Qz/S`;
- posterior calibration curves;
- precision-recall curves;
- confusion matrix at user-selected operating points;
- metrics in redshift, magnitude, latitude, and blend-quality bins.

### Candidate report

For each interesting companion, generate a compact diagnostic page or notebook cell containing:

- observed colors with errors;
- primary redshift;
- QSO model color locus near the primary redshift;
- star/background density near the candidate;
- quasar-only redshift PDF;
- Bayes factor;
- posterior(s);
- quality/blending warnings.

---

## 18. Optional astrometric and morphology evidence

Keep these separate from the core color likelihood so the scientific contribution of each evidence source remains inspectable.

### Gaia astrometry

Add

\[
p(\mathbf{a}_{\rm Gaia}\mid S,m,l,b)
\]

and

\[
p(\mathbf{a}_{\rm Gaia}\mid Q)
\]

as another likelihood factor, where \(\mathbf{a}_{\rm Gaia}\) can include parallax and proper motion with their covariance.

Do not use a simple hard cut as the only implementation.

### Morphology

Add a probabilistic morphology likelihood if desired.

Remember that low-redshift QSOs can have resolved host contribution, so "not PSF-like" is not equivalent to "not QSO".

---

## 19. Close-pair / physical-binary extension

After the photometric classifier is validated, add an optional pair prior.

Let \(\theta\) be the angular separation from the primary.

The field-QSO hypothesis uses the ordinary QSO surface density.

The physical-pair hypothesis uses a conditional QSO intensity around an existing QSO:

\[
\lambda_{\rm pair}
\propto
\Sigma_Q(z,m)\,
[1+w_{QQ}(\theta,z)]
\]

or a more complete model in projected separation and line-of-sight velocity.

This layer must be based on an externally supplied clustering model and must be switchable off.

Never use a historical clustering measurement outside its documented redshift, luminosity, separation, and selection regime without explicit justification.

---

## 20. Development milestones for Claude Code / Codex CLI

### Milestone 1: repository and schemas

Deliver:

- package skeleton;
- Pydantic configuration;
- canonical candidate/training table schema;
- survey adapter interfaces;
- unit tests for config and schema validation.

Acceptance:

- `pytest` passes;
- invalid survey/model combinations fail clearly.

### Milestone 2: photometry and covariance

Deliver:

- Legacy adapter;
- SDSS adapter;
- extinction handling;
- feature transforms;
- full covariance propagation;
- missing-band masks;
- blend-quality flags.

Acceptance:

- covariance Monte Carlo tests pass;
- negative fluxes are preserved at canonical-table level;
- no hidden magnitude clipping.

### Milestone 3: XD-GMM mathematics

Deliver:

- XD fit wrapper;
- serialized mixture representation;
- analytic Gaussian conditioning;
- candidate-error convolution;
- missing-dimension marginalization;
- unit tests against direct Gaussian identities.

Acceptance:

- all synthetic analytic tests pass.

### Milestone 4: QSO color-redshift model

Deliver:

- QSO sample builder;
- spatial CV;
- magnitude-conditioned XD training;
- `loglike_at_redshift`;
- analytic `p(z | photometry, Q)`;
- redshift-window integration;
- diagnostic plots.

Acceptance:

- held-out QSO likelihood and redshift coverage reports are generated automatically.

### Milestone 5: stellar/background model

Deliver:

- explicit-star and empirical-background modes;
- sky/magnitude hierarchy;
- surface-density map;
- local fallback logic;
- diagnostics.

Acceptance:

- sky-block validation works;
- sparse cells fall back deterministically and record the fallback level.

### Milestone 6: priors and posterior

Deliver:

- QSO prior interface;
- star prior/intensity;
- star-only posterior;
- field-QSO competing hypothesis;
- Bayes factors;
- posterior normalization tests.

Acceptance:

- prior-shift tests pass;
- every score output contains both evidence and posterior quantities.

### Milestone 7: calibration and candidate scoring

Deliver:

- calibration report;
- candidate batch scorer;
- Parquet output;
- diagnostic plots/notebook;
- model manifests and provenance hashes.

Acceptance:

- a candidate can be traced to exact model files, training-data manifests, config, and photometric system.

### Milestone 8: optional physical-pair prior

Only after Milestones 1-7 are stable.

Deliver:

- separation-aware prior interface;
- configurable quasar clustering model;
- outputs clearly separating `P_sameQ_photometric` from any physical-pair posterior.

---

## 21. Coding rules for the implementation agent

1. Do not hard-code empirical thresholds unless they are explicitly present in the user configuration.
2. Do not silently drop non-detections.
3. Do not silently replace negative fluxes.
4. Do not assume independent color errors.
5. Do not combine SDSS and Legacy photometric systems without an explicit transformation/model boundary.
6. Do not use raw spectroscopic class fractions as population priors.
7. Do not call a class-conditional likelihood a posterior probability.
8. Keep all likelihood calculations in log space.
9. Store the full model provenance.
10. Make every random procedure reproducible from an explicit seed in the run config.
11. Use type annotations for public functions.
12. Use docstrings that state units and probability normalization.
13. Prefer vectorized array operations over Python loops in scoring.
14. Add tests before optimizing.
15. Treat close-pair deblending failures as a scientific validity issue, not merely a software warning.
16. Expose Bayes factors even when posterior calibration is unavailable.
17. If the QSO prior is not defensible, return likelihoods/Bayes factors and mark posterior fields unavailable rather than fabricating a prior.
18. If a candidate is outside the training support, emit an OOD flag and do not conceal it behind a high posterior.

---

## 22. Practical v1 recommendation

Implement in this order:

1. one photometric system first, preferably the one containing most candidates;
2. optical bands only;
3. QSO XD model in color/relative-flux plus redshift, conditioned on reference magnitude;
4. empirical stellar/background XD model by magnitude and sky;
5. Bayes factor `QSO at z_primary` versus star/background;
6. only then add population priors and calibrated posterior;
7. add field-QSO redshift contamination;
8. add WISE, Gaia, morphology, and physical-pair separation priors as modular extensions.

This ordering produces a scientifically interpretable result early: first "does this object look like a QSO at the primary redshift?", then "is that explanation more plausible than the actual local background population?", and finally "what is the posterior under an explicit population prior?"

---

## 23. References and implementation precedents

### QSO probability and redshift density modeling

- Bovy et al., "Think Outside the Color Box: Probabilistic Target Selection and the SDSS-XDQSO Quasar Targeting Catalog", arXiv:1011.6392
  https://arxiv.org/abs/1011.6392

- Bovy et al., "Photometric redshifts and quasar probabilities from a single, data-driven generative model", arXiv:1105.3975
  https://arxiv.org/abs/1105.3975

- Bovy, Hogg, and Roweis, "Extreme deconvolution: Inferring complete distribution functions from noisy, heterogeneous and incomplete observations", arXiv:0905.2979
  https://arxiv.org/abs/0905.2979

- DiPompeo et al., "Quasar Probabilities and Redshifts from WISE mid-IR through GALEX UV Photometry", arXiv:1507.02884
  https://arxiv.org/abs/1507.02884

- astroML `XDGMM` documentation
  https://www.astroml.org/modules/generated/astroML.density_estimation.XDGMM.html

### Spectroscopic QSO training catalogs

- SDSS DR16Q catalog description
  https://www.sdss4.org/dr17/algorithms/qso_catalog/

- Lyke et al., "The Sloan Digital Sky Survey Quasar Catalog: Sixteenth Data Release", arXiv:2007.09001
  https://arxiv.org/abs/2007.09001

- DESI DR1 documentation
  https://data.desi.lbl.gov/doc/releases/dr1/

### Legacy Surveys photometry

- Legacy Surveys DR10 catalog columns and photometric quantities
  https://www.legacysurvey.org/dr10/catalogs/

- Legacy Surveys DR10 description
  https://www.legacysurvey.org/dr10/description/

### Direct precedent for close QSO-pair targeting

- SDSS quasar-pair ancillary program, which used KDE/XDQSOz-based QSO selection for close companions
  https://www.sdss4.org/dr15/algorithms/ancillary/boss/smallscaleqso/

- Hennawi et al., "Binary Quasars in the Sloan Digital Sky Survey: Evidence for Excess Clustering on Small Scales", arXiv:astro-ph/0504535
  https://arxiv.org/abs/astro-ph/0504535

- Eftekharzadeh et al., "Clustering on very small scales from a large sample of confirmed quasar pairs", arXiv:1702.03491
  https://arxiv.org/abs/1702.03491

---

## 24. Final scientific output contract

The final scorer must never return only one opaque probability.

At minimum return:

```text
candidate_id
primary_id
z_primary
ref_mag
photometric_system
loglike_qso_zprimary
p_zmatch_given_qso
loglike_star_or_background
log_bayes_factor_qz_star
sigma_qso_match
sigma_star_or_background
p_sameq_vs_star
lambda_fieldq
p_sameq_full
qso_ood_score
star_ood_score
photometry_quality_flags
blend_flags
model_manifest_id
config_hash
```

Any quantity that cannot be supported by the available prior/selection information should be `null` with an explicit status code, not estimated from an arbitrary class fraction.
