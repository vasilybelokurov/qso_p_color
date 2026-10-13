#!/usr/bin/env python
"""Write the method note's Legacy-r fallback validation table from docs/fallback_reference/validation_*.json.

Output: docs/method_unified/fallback_table.tex (\\input by method_unified.tex, Section sec:nolegacy).
Usage:  python scripts/fallback_reference/note_table.py
"""
import json
from pathlib import Path

DOC = Path('docs/fallback_reference')
OUT = Path('docs/method_unified/fallback_table.tex')
BANDS = (('ps1', 'PS1 $r$'), ('nsc', 'NSC $r$'), ('skymapper', 'SkyMapper $r$'))
MODELS = ('current_magindep', 'current')          # first number: magnitude-independent, as elsewhere in the note


def pair(values, fmt):
    return ' / '.join(fmt.format(v).replace('-', '$-$') for v in values)


def main():
    rows = []
    for key, name in BANDS:
        r = [json.loads((DOC/f'validation_resolved_{key}_{m}.json').read_text()) for m in MODELS]
        b = [json.loads((DOC/f'validation_blend_{key}_{m}.json').read_text()) for m in MODELS]
        rows.append(' & '.join([
            name,
            f"{r[0]['n']['qso']:,}".replace(',', '{,}'),
            pair([x['schemes']['legacy_r']['auc_common'] for x in r], '{:.4f}'),
            pair([x['auc_loss']['value'] for x in r], '{:.4f}'),
            pair([100*x['recall_loss']['value'] for x in r], '{:.1f}'),
            pair([100*x['bkg_p05_increase']['value'] for x in r], '{:+.2f}'),
            pair([x['legacy_r_recovery']['qso_median'] for x in r], '{:+.2f}') + ' (' +
            pair([x['legacy_r_recovery']['qso_nmad'] for x in r], '{:.2f}') + ')',
            pair([x['auc_loss'] for x in b], '{:+.3f}'),
            pair([x['dlogbf']['median'] for x in b], '{:+.1f}'),
            pair([100*x['dlogbf']['sign_agreement'] for x in b], '{:.0f}'),
        ]) + r' \\')
    OUT.write_text('\n'.join([
        r'\begin{tabular}{lrcccccccc}', r'\hline',
        r'& \multicolumn{6}{c}{Resolved mode} & \multicolumn{3}{c}{Blend mode} \\',
        r'Fallback & $N_Q$ & AUC$_{\rm true\,r}$ & AUC loss & recall loss & $\Delta$bkg & $\Delta r$ (NMAD) '
        r'& AUC loss & $\Delta\log{\rm BF}$ & sign \\',
        r'& & & & (pp) & (pp) & (mag) & & (nat) & (\%) \\', r'\hline', *rows, r'\hline', r'\end{tabular}', '']))
    print(OUT.read_text())


if __name__ == '__main__':
    main()
