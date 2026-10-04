#!/usr/bin/env python
"""Promote the 3 October 2026 bundles (binned, QSO-cleaned background; faint limit). PI decision, 4 October.

Pointers after promotion (models/multisurvey_psf/):
  current            magnitude-DEPENDENT QSO colours (the PI's choice for `current`)
  current_magdep     same bundle as `current` (kept so existing code keeps working)
  current_magindep   magnitude-INDEPENDENT QSO colours: a first-class model that can and should be used
                     alongside `current` (score candidates with both)
  previous           unchanged: the 28 September bundle (rollback used by tests)
  previous_20261002_magindep, previous_20261002_magdep   the 2 October bundles (rollback)
Bundles are copied unchanged except the manifest status; the candidate directories are not modified.

Usage: python scripts/promote_stellar_binned.py
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('models/multisurvey_psf')
SOURCE = ROOT/'work'/'stellar_binned'/'20261003'/'bundles'
TAGS = {'magdep': ('dependent', '20261003_magdep'), 'magindep': ('independent', '20261003_magindep')}
COMMON = dict(extinction='SFD98', method_note='docs/method_unified/method_unified.tex',
              comparison='docs/method_unified/stellar_binned_comparison.json',
              model_difference='docs/method_unified/model_difference_bootstrap.json',
              faint_limit='Legacy r (South, else North) S/N >= 10; Legacy r is the reference band',
              scope='PSF sources with Legacy r S/N >= 10; any subset of 41 bands containing it; extinction-corrected; '
                    'binned QSO-cleaned background with smooth sky gate; Student-t catch-all; calibrated QSO-support cut. '
                    'Load with UnifiedPSFModel.load.',
              full_probability_calibration=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for key, (name, tag) in TAGS.items():
        dest = ROOT/tag
        if not dest.exists():
            shutil.copytree(SOURCE/name, dest)
        label = 'magnitude-dependent QSO colours' if key == 'magdep' else 'magnitude-independent QSO colours'
        manifest = json.loads((dest/'manifest.json').read_text())
        for n, h in manifest['files'].items():
            assert sha(dest/n) == h, n
        manifest.update(bundle_id=tag, status=f'promoted 4 October 2026: {label}; binned QSO-cleaned background, faint limit; '
                        'not full probability calibration')
        (dest/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pointers = {
        'current': dict(bundle=TAGS['magdep'][1], qso_colours='magnitude-dependent QSO colours',
                        companion='current_magindep: magnitude-independent QSO colours. Use both; see the method note, Section 6.2.'),
        'current_magdep': dict(bundle=TAGS['magdep'][1], qso_colours='magnitude-dependent QSO colours (same bundle as current)'),
        'current_magindep': dict(bundle=TAGS['magindep'][1], qso_colours='magnitude-independent QSO colours (XDQSO assumption)',
                                 companion='current: magnitude-dependent QSO colours. Use both; see the method note, Section 6.2.'),
        'previous_20261002_magindep': dict(bundle='13866e45ef794059', qso_colours='magnitude-independent', note='2 October bundle; rollback'),
        'previous_20261002_magdep': dict(bundle='5f4002492dbb849c', qso_colours='magnitude-dependent', note='2 October bundle; rollback'),
    }
    for name, d in pointers.items():
        body = dict(d, **(COMMON if name.startswith('current') else {}))
        (ROOT/name).write_text(json.dumps(body, indent=2) + '\n')
        print(name, '->', d['bundle'])


if __name__ == '__main__':
    main()
