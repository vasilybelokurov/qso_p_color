#!/usr/bin/env python
"""Experiment manifest for the score-all work (docs/euclid/PLAN_score_all.md, Phase A).

Records, per Galactic NESTED nside-16 Euclid cell of docs/euclid/sky_roles.json, what has already looked at it, so
that any new evaluation can be shown to use untouched sky. Nothing is allocated here: choosing new evaluation
cells is PI decision D3. The manifest only states which cells are spent and what is left.

Uses recorded (from the code that consumed each role; file:line in the 'evidence' block):
  fit     matched QSO training rows of stage 1 and 2 (fit_stage1.py:151, fit_stage2.py:38) and the fallback window
          (build_stage1.py:93,111)
  select  QSO select rows (fit_stage1.py:151) and the select cones (build_stage1.py:72): model and policy choices
  calib   calib QSOs and cones (validate_stage1.py:114): thresholds and calibration
  test    stage-1 test and the stage-2 diagnosis (validate_stage1.py:114, compare_stages.py): SPENT for decisions
plus the parent background cones that overlap DR1 (rows of the background table, fit_stage1.py:9), whose cells are
counted from that table.

Output: docs/score_all/manifest.json with a sha256 of its own content (key order fixed), and a summary on stdout.

    python scripts/score_all/manifest.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
ROLES = ROOT/'docs/euclid/sky_roles.json'
OUT = ROOT/'docs/score_all/manifest.json'
USES = dict(fit=['stage1_fit_qso', 'stage2_fit_qso', 'stage1_fallback_window'],
            select=['stage1_model_selection', 'select_cones_policy'],
            calib=['calibration_thresholds'],
            test=['stage1_test', 'stage2_test_and_diagnosis'])


def cell_counts(table: str, settings: dict) -> dict:
    """Clean rows per Euclid cell in a local table (the rows the fits could use)."""
    import sys
    sys.path.insert(0, str(ROOT/'scripts/euclid'))
    from fit_stage1 import load_rows
    d = load_rows(table, settings)
    cells, n = np.unique(np.asarray(d['euclid_cell'])[d['clean']], return_counts=True)
    return {int(c): int(k) for c, k in zip(cells, n)}


def main():
    roles = json.loads(ROLES.read_text())
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    qso = cell_counts('qso', settings)
    tables = sorted(p.stem for p in (BASE/'tables').glob('*.npz'))
    bkg_table = 'background' if 'background' in tables else None
    bkg = cell_counts(bkg_table, settings) if bkg_table else {}
    cones = {}
    for c in roles['cones']:
        cones.setdefault(int(c['cell']), []).append(dict(role=c['role'], id=int(c['id']), ra=c['ra'], dec=c['dec'],
                                                       radius_deg=c['radius_deg']))
    cells = []
    for c in roles['cells']:
        k = int(c['cell']); uses = list(USES[c['role']])
        if bkg.get(k):
            uses.append('parent_background_rows')
        cells.append(dict(cell=k, role=c['role'], coverage=c['coverage'], l=c['l'], b=c['b'],
                          n_qso_clean=qso.get(k, 0), n_parent_background_clean=bkg.get(k, 0),
                          cones=cones.get(k, []), uses=uses))
    summary = {}
    for role in USES:
        sel = [c for c in cells if c['role'] == role]
        summary[role] = dict(cells=len(sel), cones=sum(len(c['cones']) for c in sel),
                             qso_clean=sum(c['n_qso_clean'] for c in sel),
                             cells_with_parent_background=sum(c['n_parent_background_clean'] > 0 for c in sel))
    untouched = [c['cell'] for c in cells if not c['cones'] and c['n_qso_clean'] == 0 and c['n_parent_background_clean'] == 0]
    out = dict(
        purpose='Which Euclid DR1 cells have been used for fitting or decisions (PLAN_score_all.md Phase A). '
                'New evaluation cells are PI decision D3; this file allocates nothing.',
        sky_roles=dict(path=str(ROLES.relative_to(ROOT)), sha256=hashlib.sha256(ROLES.read_bytes()).hexdigest(),
                       seed=roles['seed'], nside=roles['nside'], frame=roles['frame'], ordering=roles['ordering']),
        background_table=bkg_table,
        spent=dict(test='stage-2 diagnosis looked at test results; test cells cannot judge new choices',
                   select='used to choose models and policies', calib='used to set thresholds',
                   fit='matched QSOs trained the models'),
        summary=summary,
        cells_with_no_recorded_use=dict(n=len(untouched), cells=untouched,
                                        note='no cone, no clean matched QSO, no parent background row; role as in '
                                             'sky_roles.json. Mostly low-coverage fit cells (coverage < 0.5).'),
        cells=cells)
    body = json.dumps(out, indent=1, sort_keys=True)
    out['sha256_of_content'] = hashlib.sha256(body.encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(dict(summary=summary, untouched=len(untouched), sha256=out['sha256_of_content']), indent=1))


if __name__ == '__main__':
    main()
