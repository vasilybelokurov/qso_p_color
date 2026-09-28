#!/usr/bin/env python
"""Save a versioned, uncapped SDSS+DESI positional master and all source IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.qso_catalogue import combine_qso_catalogues


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/qso_master.json"))
    parser.add_argument("--output-root",type=Path,
                        help="Override destination for staging; catalogue content is unchanged")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    sources={}
    data={}
    for name,keys in (("desi",("targetid","ra","dec","zspec")),
                      ("sdss",("sdss_name","ra","dec","zspec","zwarning","is_qso_final"))):
        path=Path(cfg["inputs"][name]).expanduser()
        with np.load(path,allow_pickle=False) as d:
            data[name]={k:d[k] for k in keys}
            embedded_query = str(d["_query"]) if "_query" in d else None
        sidecar=path.with_suffix(".json")
        sources[name]=dict(path=str(path),sha256=sha256(path),rows=len(data[name]["ra"]),
            source_provenance=json.loads(sidecar.read_text()) if sidecar.exists()
                              else dict(query=embedded_query))
    method={k:v for k,v in cfg.items() if k not in ("inputs","output_root")}
    identity=dict(inputs={k:v["sha256"] for k,v in sources.items()},method=method,
        implementation_sha256=sha256(Path(__file__).resolve().parents[1]/"src/qso_pcolor/qso_catalogue.py"))
    build_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    root=(args.output_root or Path(cfg["output_root"]).expanduser())
    dest=root/build_id
    if dest.exists():
        manifest=json.loads((dest/"manifest.json").read_text())
        for name,expected in manifest["files"].items():
            if sha256(dest/name)!=expected: raise ValueError(f"damaged existing output: {name}")
        print(f"Verified existing master: {dest}")
        return
    start=time.monotonic()
    objects,members,report=combine_qso_catalogues(data["desi"],data["sdss"],
        radius_arcsec=cfg["radius_arcsec"],redshift_conflict_tolerance=cfg["redshift_conflict_tolerance"])
    dest.mkdir(parents=True)
    files={}
    for name,arrays in (("objects.npz",objects),("members.npz",members)):
        tmp=dest/f".{name}"
        with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
        tmp.replace(dest/name);files[name]=sha256(dest/name)
    manifest=dict(kind="sdss_desi_positional_master",build_id=build_id,identity=identity,
        sources=sources,selection=method,report=report,files=files,
        elapsed_seconds=time.monotonic()-start,
        units=dict(ra="ICRS degrees",dec="ICRS degrees",zspec="dimensionless",max_member_separation_arcsec="arcsec"),
        schema=dict(objects="One row per connected positional group; object_id names its adopted source.",
                    members="All original rows; catalogue + source_id identifies the original entry; input_row indexes the hashed input file; object_index links to objects.npz."),
        limitations=["Input selections and any inherited redshift restrictions are recorded in sources.*.source_provenance; no additional redshift cut is made by this builder.",
            "SDSS local input retains all DR16Q rows and flags, including invalid redshifts; these remain in membership.",
            "No photometric measurements, morphology cuts, sky cuts, magnitude cuts, training holdouts or random caps applied.",
            "Redshift-discordant or extended positional groups require a declared training policy; no source membership was deleted."])
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    pointer=root/"current.json";tmp=root/".current.json.tmp"
    tmp.write_text(json.dumps(dict(build_id=build_id,manifest=f"{build_id}/manifest.json"),indent=2)+"\n")
    tmp.replace(pointer)
    print(json.dumps(report,indent=2),flush=True)
    print(f"Saved {dest} in {manifest['elapsed_seconds']:.1f}s",flush=True)


if __name__=="__main__": main()
