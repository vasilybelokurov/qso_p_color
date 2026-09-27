"""Local store of reusable catalogues, so analyses read files instead of re-querying WSDB.

Root: ``$QSO_PCOLOR_STORE`` or ``~/data/qso_p_color`` (outside Dropbox).

    catalogues/<name>.npz    one master table per catalogue, all columns we use
    catalogues/<name>.json   provenance: SQL, date, rows, elapsed time, notes
    meta/                    Legacy brick tables and per-brick mask images
    project_data/            the project's former ``data/`` (symlinked back)

Masters are fetched once by ``scripts/build_store.py``. :func:`load` never
queries: a missing file or column raises and names the command that builds it.

    from qso_pcolor.store import load
    desi = load("desi_dr1_qso", columns=("zspec", "flux_r", "type"))
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CATALOGUES = {
    "desi_dr1_qso": "DESI DR1 zpix QSO (zwarn = 0, zcat_primary) x Legacy DR9 photometry by targetid",
    "dr16q_dr9": "SDSS DR16Q (all rows) x nearest Legacy DR9 row of its hemisphere within 1 arcsec",
    "pairs_desi_dr1": "DESI DR1 quasar-companion pairs (scripts/build_pair_validation.py)",
    "pairs_desi_dr1_dr9": "pair companions x nearest Legacy DR9 row of their hemisphere",
    "qso_draw_dr9": "the flat-in-z DESI+SDSS draw of the multi-survey sample x Legacy DR9",
}


def root() -> Path:
    return Path(os.environ.get("QSO_PCOLOR_STORE", "~/data/qso_p_color")).expanduser()


def path(name: str) -> Path:
    return root() / "catalogues" / f"{name}.npz"


def provenance(name: str) -> dict:
    p = path(name).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


def load(name: str, columns: tuple[str, ...] | None = None) -> dict:
    """A master table as a dict of arrays; only ``columns`` if given."""
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} is not in the store: python scripts/build_store.py --only {name}")
    with np.load(p, allow_pickle=False) as d:
        have = set(d.files)
        want = have if columns is None else set(columns)
        missing = want - have
        if missing:
            raise KeyError(f"{name} lacks {sorted(missing)}; add them to scripts/build_store.py "
                           f"and rebuild with --only {name} --refresh")
        return {k: d[k] for k in want}


def save(name: str, arrays: dict, info: dict) -> Path:
    """Write a master atomically with its provenance sidecar."""
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    tmp.replace(p)
    p.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str))
    return p
