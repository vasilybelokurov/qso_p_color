"""The Legacy baseline bundle: provenance checks and scoring through the selection."""
import json

import numpy as np
import pytest

from qso_pcolor.baseline import LegacyBaseline, file_sha256
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.legacy import LegacySelection, hemisphere_labels
from qso_pcolor.multisurvey import BandLuptitudeTransform, MultiSurveyModel, MultiSurveyOutlier
from qso_pcolor.priors import BackgroundSurfaceDensity, GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel


def write_bundle(root, bundle_id="b1", selection=LegacySelection("point")):
    labels = hemisphere_labels("south"); d = 5
    cov = .2 * np.eye(d) + .3 * np.ones((d, d))
    art = dict(bundle_id=bundle_id, selection_id=selection.identity)
    qso = SlicedColourRedshiftModel(np.array([.5, 2.5]),
        [GaussianMixture(np.ones(1), np.full((1, d), m), cov[None], labels) for m in (20., 21.)],
        np.array([100, 100]), "legacy_psf_south", labels)
    bkg = GaussianMixture(np.ones(1), np.full((1, d), 20.5), (2 * cov)[None], labels)
    tr = BandLuptitudeTransform(labels, np.full(d, .05))
    order = (labels[1],) + tuple(b for b in labels if b != labels[1])
    model = MultiSurveyModel(qso, bkg, tr, order, np.tile([10., 30.], (d, 1)), meta=dict(art))
    m_edges = np.arange(16, 24.01, .5); z_edges = np.array([0.1, 1.5, 3.0])
    meta = dict(art, reference_band=labels[1], transform_id=model.transform_id)
    gq = GridQSOPrior(np.array([.8, 2.25]), .5 * (m_edges[1:] + m_edges[:-1]),
                      np.full((2, m_edges.size - 1), 5.), dict(meta), z_edges=z_edges, mag_edges=m_edges)
    bd = BackgroundSurfaceDensity(1, 1, m_edges, {(0, i): 1000. for i in range(m_edges.size - 1)},
                                  {0: 1.0}, meta=dict(meta))
    priors = dict(kind="multisurvey_priors", transform_id=model.transform_id, **art,
                  anchors={labels[1]: {"qso_prior": gq.to_dict(), "background_density": bd.to_dict()}})
    out = MultiSurveyOutlier(np.full(d, 20.5), 2 * cov, 1.0, 0.0, labels, model.transform_id,
                             {labels[1]: (np.array([17., 22.5]), np.array([.01]))},
                             meta=dict(art), family="student_t", nu=2.0)
    root.mkdir(parents=True, exist_ok=True)
    model.save(root / "south_model.json")
    (root / "south_priors.json").write_text(json.dumps(priors))
    out.save(root / "south_outlier.json")
    files = {"model": "south_model.json", "priors": "south_priors.json", "outlier": "south_outlier.json"}
    man = dict(kind="legacy_baseline_bundle", version=1, bundle_id=bundle_id,
               selection=selection.to_dict(), selection_id=selection.identity,
               domain=dict(candidate_ref_range=[17.0, 22.5]), hemispheres={"south": files},
               files={f: file_sha256(root / f) for f in files.values()})
    (root / "manifest.json").write_text(json.dumps(man))
    return root


def rows(n=4):
    r = dict(ra=np.full(n, 150.0), dec=np.full(n, 2.0), release=np.full(n, 9010),
             type=np.array(["PSF", "PSF", "REX", "PSF"])[:n], maskbits=np.zeros(n, int))
    for b, f in zip(("g", "r", "z", "w1", "w2"), (4., 5., 6., 8., 8.)):
        r[f"flux_{b}"] = np.full(n, f); r[f"flux_ivar_{b}"] = np.full(n, 100.)
        r[f"nobs_{b}"] = np.full(n, 3)
    r["flux_r"] = np.array([5., 5., 5., 1e-3])[:n]                   # last: far too faint
    return r


def test_load_and_score_through_the_selection(tmp_path):
    bl = LegacyBaseline.load(write_bundle(tmp_path / "b"))
    scores, dec = bl.score_rows(rows(), z_primary=1.0, match=RedshiftMatch(2000.0))
    assert dec["eligible"].tolist() == [True, True, False, False]
    assert dec["reason"].tolist() == ["", "", "morphology", "outside_domain"]
    assert scores[2] is None and scores[3] is None
    assert scores[0].reference_band == "decals_dr9_south:r"
    assert np.isfinite(scores[0].log_r_per_unit_z)


def test_pointer_file_resolves(tmp_path):
    write_bundle(tmp_path / "b1")
    (tmp_path / "current").write_text(json.dumps({"bundle": "b1"}))
    assert LegacyBaseline.load(tmp_path / "current").bundle_id == "b1"


def test_tampered_file_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b")
    p = root / "south_priors.json"
    d = json.loads(p.read_text()); d["completeness_constant"] = 3.0; p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="does not match the manifest"):
        LegacyBaseline.load(root)


def test_mixed_bundle_is_refused_even_with_matching_hashes(tmp_path):
    a, b = write_bundle(tmp_path / "a", "A"), write_bundle(tmp_path / "b", "B")
    # same transform in both; move B's priors into A and re-record the hash
    (a / "south_priors.json").write_bytes((b / "south_priors.json").read_bytes())
    man = json.loads((a / "manifest.json").read_text())
    man["files"]["south_priors.json"] = file_sha256(a / "south_priors.json")
    (a / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="different bundle"):
        LegacyBaseline.load(a)


def test_selection_mismatch_is_refused(tmp_path):
    root = write_bundle(tmp_path / "b")
    man = json.loads((root / "manifest.json").read_text())
    man["selection"]["morphology"] = "all"          # declared selection no longer the fitted one
    (root / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="identity"):
        LegacyBaseline.load(root)
