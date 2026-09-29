"""Compute accounting must weight large fits and respect convergence/selection."""
import json
from pathlib import Path
import runpy

import pytest


def test_compute_progress_weights_rows_components_and_actual_finished_iterations(tmp_path):
    compute = runpy.run_path(str(Path(__file__).parents[1]/'scripts/monitor_full_training.py'))['compute_progress']
    cfg = dict(background_k_candidates=[1, 2], qso_k_candidates=[1, 2],
               fit_roles=['fit'], final_shape_roles=['fit', 'select'], selection_max_iter=4, max_iter=6)
    preflight = dict(roles=dict(stars=dict(fit=dict(rows=10), select=dict(rows=5))),
                     slices=[dict(index=0, rows=dict(fit=100, select=20))])

    def save(name, value):
        (tmp_path/name).write_text(json.dumps(value))

    assert compute(tmp_path, cfg, preflight)['populations']['overall']['percent'] == 0
    save('background.select_k1.checkpoint.json', dict(iteration=2))
    assert compute(tmp_path, cfg, preflight)['populations']['overall']['percent'] == pytest.approx(100*20/2940)

    # Early convergence and selecting K=1 both reduce the projected remaining work.
    save('background.select_k1.json', dict(n_iter=2, score=2))
    save('background.select_k2.json', dict(n_iter=4, score=1))
    estimate = compute(tmp_path, cfg, preflight)['populations']
    assert estimate['stars'] == dict(done=100, remaining=90, percent=pytest.approx(100*100/190))
    assert estimate['qso']['remaining'] == 2640

    # Stale checkpoints do not double count a completed fit.
    save('background.json', dict(n_iter=3, selected_k=1, mixture=dict(weights=[1])))
    for k in (1, 2):
        save(f'qso_00.select_k{k}.json', dict(n_iter=4, score=k))
    save('qso_00.json', dict(n_iter=3, selected_k=2, mixture=dict(weights=[.5, .5])))
    estimate = compute(tmp_path, cfg, preflight)['populations']
    assert estimate['overall'] == dict(done=2065, remaining=0, percent=100)
