#!/usr/bin/env python
"""Measure the Legacy morphology-selection area in the multi-survey field cones."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from qso_pcolor.data import galactic_from_equatorial
from qso_pcolor.legacy import (brick_image_fetcher, cone_usable_fraction, hemisphere_of,
                               load_bricks)
from qso_pcolor.spatial import cone_within_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multisurvey_psf_recovery.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    area_config = cfg["area"]
    fields = json.loads((Path(cfg["sample_dir"]) / "fields.json").read_text())
    bricks = load_bricks(area_config["bricks"])
    old = Path(area_config["existing_image_cache"])
    old_fetch = brick_image_fetcher(old)
    new_fetch = brick_image_fetcher(Path(cfg["cache_dir"]) / "brick_images")
    root = Path(cfg["cache_dir"]) / "areas"
    root.mkdir(parents=True, exist_ok=True)

    def fetch(brick, hemisphere, kind):
        path = old / hemisphere / f"{brick}-{kind}.fits.fz"
        return old_fetch(brick, hemisphere, kind) if path.exists() else new_fetch(brick, hemisphere, kind)

    def measure(f):
        path = root / f"field_{f['field']:02d}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous.get("area_config") == area_config and
                    previous.get("seed") == cfg["seed"] + f["field"] and
                    previous["ra"] == f["ra"] and previous["dec"] == f["dec"] and
                    previous.get("containment_nside") == cfg["spatial"]["nside"]):
                return previous
        l, b = galactic_from_equatorial([f["ra"]], [f["dec"]])
        hemisphere = hemisphere_of([f["ra"]], [f["dec"]])[0]
        area = cone_usable_fraction(f["ra"], f["dec"], area_config["radius_deg"],
            hemisphere, bricks, fetch, n_points=area_config["n_points"],
            seed=cfg["seed"] + f["field"])
        record = dict(f, l=float(l[0]), b=float(b[0]), area=area, hemisphere=str(hemisphere),
            area_config=area_config, seed=cfg["seed"] + f["field"],
            containment_nside=cfg["spatial"]["nside"],
            contained=cone_within_cell(float(l[0]), float(b[0]), area_config["radius_deg"],
                nside=cfg["spatial"]["nside"], boundary_factor=area_config["boundary_factor"]))
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(record)); tmp.replace(path)
        print(f"field {f['field']}: {area['area_deg2']:.6f} usable deg2", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
        result = list(executor.map(measure, fields))
    (Path(cfg["cache_dir"]) / "areas.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
