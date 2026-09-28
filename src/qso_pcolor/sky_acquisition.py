"""Resumable acquisition blocks; these never limit the scientific sample."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import healpy as hp
import numpy as np

from .data import _load_npz, _save_npz


def write_record(path: Path, record: dict) -> None:
    """Atomically replace a JSON checkpoint."""
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(path)


@contextmanager
def acquisition_lock(path: Path):
    """Prevent concurrent new workers from querying through the same cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'an acquisition worker already holds {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sky_blocks(ra: np.ndarray, dec: np.ndarray, *, max_targets: int,
               parent_nside: int, order_nside: int) -> list[np.ndarray]:
    """Partition ICRS positions (degrees), retaining each input row exactly once.

    Sort by nested HEALPix pixel, then split within each parent cell at the
    requested row limit. A block is a group of targets, not a survey cut:
    counterparts across its boundary remain searchable. Repeated coordinates
    still represent distinct input rows. The limit is operational, not a cap.
    """
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    if (ra.ndim != 1 or ra.shape != dec.shape or
            not np.all(np.isfinite(ra+dec)) or np.any(np.abs(dec) > 90)):
        raise ValueError('finite one-dimensional sky coordinates required')
    if (isinstance(max_targets, bool) or not isinstance(max_targets, (int, np.integer)) or max_targets < 1 or
            not hp.isnsideok(parent_nside, nest=True) or
            not hp.isnsideok(order_nside, nest=True) or order_nside < parent_nside):
        raise ValueError('positive integer block size and nested HEALPix resolutions required')
    if not len(ra):
        return []
    pixels = hp.ang2pix(order_nside, ra % 360, dec, nest=True, lonlat=True)
    order = np.argsort(pixels, kind='stable')
    parents = pixels[order] // (order_nside//parent_nside)**2
    edges = np.r_[0, np.flatnonzero(np.diff(parents))+1, len(order)]
    return [order[start:min(start+max_targets, hi)]
            for lo, hi in zip(edges[:-1], edges[1:])
            for start in range(lo, hi, max_targets)]


def fetch_sky_blocks(inputs: dict, directory: Path, *, identity: dict,
                      config: dict, fetch: Callable[[dict, Path], dict],
                      progress_path: Path | None = None) -> dict:
    """Fetch sequential blocks, saving durable rows and timing after each query.

    ``inputs`` contains degree-valued ``ra, dec`` and optional exact integer
    IDs. ``fetch`` returns one row per input, with a local integer ``idx``;
    unmatched rows must be retained. Return rows in original input order.
    SQL, input arrays and block geometry all enter the cache identity.
    A failed block leaves every earlier block usable on restart.
    """
    inputs = {k: np.asarray(v) for k, v in inputs.items()}
    n = len(inputs['ra'])
    if any(v.ndim != 1 or len(v) != n or v.dtype.hasobject for v in inputs.values()):
        raise ValueError('aligned non-object input arrays required')
    blocks = sky_blocks(inputs['ra'], inputs['dec'], **config)
    if not blocks:
        raise ValueError('cannot query an empty target list')
    manifest = dict(identity=identity, config=config, targets=n, block_sizes=[len(b) for b in blocks])
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    for name, values in sorted(inputs.items()):
        digest.update(name.encode()+values.dtype.str.encode()+values.tobytes())
    root = directory/digest.hexdigest()[:24]
    root.mkdir(parents=True, exist_ok=True)
    def save_status(value):
        write_record(root/'status.json', value)
        if progress_path is not None:
            write_record(progress_path, value)
    with acquisition_lock(directory/'worker.lock'):
        write_record(root/'manifest.json', manifest)
        result = {}
        done = 0
        for number, rows in enumerate(blocks):
            path = root/f'block_{number:05d}.npz'
            meta = path.with_suffix('.json')
            cached = path.exists()
            started = time.monotonic()
            status = dict(block=number, blocks=len(blocks), completed_targets=done,
                          total_targets=n, block_targets=len(rows),
                          updated_utc=datetime.now(timezone.utc).isoformat())
            if cached:
                raw = _load_npz(path)
                if not np.array_equal(raw.pop('block_input_row'), rows):
                    raise ValueError('cached block target identities differ')
            else:
                save_status(dict(status, stage='querying'))
                try:
                    raw = fetch({k: v[rows] for k, v in inputs.items()}, root/f'query_{number:05d}')
                    order = np.argsort(raw['idx'])
                    if not np.array_equal(raw['idx'][order], np.arange(len(rows))):
                        raise ValueError('block query lost or duplicated input identities')
                    raw = {k: np.asarray(v)[order] for k, v in raw.items()}
                    _save_npz(path, block_input_row=rows, **raw)
                    elapsed = time.monotonic()-started
                    write_record(meta, dict(status, elapsed_seconds=elapsed,
                        targets_per_second=len(rows)/elapsed,
                        matched=int(np.isfinite(raw['ra']+raw['dec']).sum())))
                except Exception as error:
                    save_status(dict(status, stage='failed',
                        error=f'{type(error).__name__}: {error}'))
                    raise
            if (not np.array_equal(raw['idx'], np.arange(len(rows))) or
                    any(v.ndim != 1 or len(v) != len(rows) for v in raw.values())):
                raise ValueError('invalid cached block rows')
            if result and set(result) != set(raw):
                raise ValueError('query schema changed between blocks')
            for key, values in raw.items():
                if key not in result:
                    result[key] = np.empty(n, dtype=values.dtype)
                result[key][rows] = values
            done += len(rows)
            save_status(dict(status, stage='complete' if done == n else 'saved',
                completed_targets=done, updated_utc=datetime.now(timezone.utc).isoformat()))
            timing = json.loads(meta.read_text()) if meta.exists() else {}
            print(f'Sky block {number+1}/{len(blocks)}: {len(rows):,} rows '
                  f'{"reused" if cached else "saved"}; {done:,}/{n:,}; '
                  f'{timing.get("elapsed_seconds", float("nan")):.1f} s', flush=True)
        result['idx'] = np.arange(n)
        return result
