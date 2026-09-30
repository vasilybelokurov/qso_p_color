"""Population completion for a frozen full-data PSF density model.

Counts use only fit/select roles. Rates are per square degree per native
reference luptitude; the QSO abundance prior is retained, not renormalized by
spectroscopic training-set size. No model shapes are fitted here.
"""
from copy import deepcopy
import numpy as np
from .full_sample import ROLES
from .spatial import SpatialSurfaceDensity


def population_rows(data: dict, roles: tuple[str, ...]) -> np.ndarray:
    """Return all eligible rows in explicit roles, without a sample-size cap."""
    if not roles or len(set(roles)) != len(roles) or any(r not in ROLES for r in roles):
        raise ValueError('invalid population roles')
    return np.flatnonzero(data['eligible'] & np.isin(data['role'], [ROLES.index(r) for r in roles]))


def merge_empty_counts(counts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent magnitude intervals until every pooled count is positive.

    ``counts`` has shape (sky regions, magnitude intervals). Counts are conserved;
    no pseudocount or arbitrary positive floor is introduced.
    """
    counts, edges = np.array(counts, copy=True), np.array(edges, float, copy=True)
    if (counts.ndim != 2 or counts.shape[1] != len(edges)-1 or
            not np.isfinite(counts).all() or (counts < 0).any() or
            not np.isfinite(edges).all() or (np.diff(edges) <= 0).any() or not counts.sum()):
        raise ValueError('invalid or empty population counts')
    while (counts.sum(axis=0) == 0).any():
        j = int(np.flatnonzero(counts.sum(axis=0) == 0)[0])
        boundary = 1 if j == 0 else j
        counts[:, boundary-1] += counts[:, boundary]
        counts = np.delete(counts, boundary, axis=1)
        edges = np.delete(edges, boundary)
    return counts, edges


def count_prior(counts: np.ndarray, areas: np.ndarray, cells: np.ndarray,
                edges: np.ndarray, *, nside: int, nside_parent: int,
                n0: float, meta: dict) -> SpatialSurfaceDensity:
    """Pool region counts and measured/declared areas into Galactic sky cells."""
    counts, areas, cells = np.asarray(counts), np.asarray(areas), np.asarray(cells)
    if (areas.shape != (len(counts),) or cells.shape != areas.shape or
            not np.isfinite(areas).all() or (areas < 0).any() or
            np.any((areas == 0) & (counts.sum(axis=1) > 0)) or not (areas > 0).any()):
        raise ValueError('counts lack corresponding usable area')
    counts, edges = merge_empty_counts(counts, edges)
    cell_counts, cell_area = {}, {}
    for cell in np.unique(cells[areas > 0]):
        use = (cells == cell) & (areas > 0)
        cell_area[int(cell)] = float(areas[use].sum())
        for j, value in enumerate(counts[use].sum(axis=0)):
            cell_counts[int(cell), j] = float(value)
    density = counts.sum(axis=0)/(areas.sum()*np.diff(edges))
    return SpatialSurfaceDensity(nside, nside_parent, edges, cell_counts,
                                 cell_area, density, n0, meta)


def rebind_qso_prior(source: dict, *, run_id: str, origin: str, origin_sha256: str) -> dict:
    """Retain numerical abundance prior exactly; record reuse in new bundle metadata."""
    result = deepcopy(source)
    result['meta'].update(model_run_id=run_id, population='psf',
                          abundance_policy='numerically unchanged inherited abundance prior',
                          abundance_source=origin, abundance_source_sha256=origin_sha256,
                          abundance_source_model_run_id=source['meta']['model_run_id'])
    return result
