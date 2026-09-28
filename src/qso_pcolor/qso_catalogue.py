"""Lossless source membership for a positional SDSS+DESI quasar master."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def combine_qso_catalogues(desi: dict, sdss: dict, *, radius_arcsec: float,
                          redshift_conflict_tolerance: float) -> tuple[dict, dict, dict]:
    """Merge all supplied catalogue rows, retaining every original identifier.

    Positions are ICRS degrees. Connected groups of pairs separated by at most
    ``radius_arcsec`` become one object. Redshift disagreement never creates a
    second object or deletes a membership; it is flagged using the dimensionless
    range ``(max(z)-min(z))/(1+min(z))`` over finite positive redshifts.

    Adopt a row with a usable redshift and clean spectroscopic flags first,
    then DESI before SDSS, then the original identifier in sorted order.
    The DESI input is required to have already passed zwarn=0 and QSO selection;
    this is the selection of the local DESI master. No morphology, sky, magnitude,
    redshift-range, band-coverage or object-count selection is applied here.

    Returns object arrays, source membership arrays, and an accounting report.
    The membership ``input_row`` refers to the original input file row, before
    sorting. ``object_index`` indexes the returned object arrays. Object IDs
    identify the adopted catalogue row, not a new survey-issued identifier.
    """
    if (not np.isfinite(radius_arcsec) or radius_arcsec <= 0 or
            not np.isfinite(redshift_conflict_tolerance) or redshift_conflict_tolerance <= 0):
        raise ValueError("positive finite positional and redshift tolerances are required")
    parts = []
    for catalogue, data, key in (("desi", desi, "targetid"), ("sdss", sdss, "sdss_name")):
        ids = np.asarray(data[key]).astype(str)
        ra, dec, z = (np.asarray(data[k], float) for k in ("ra", "dec", "zspec"))
        if (ids.ndim != 1 or ra.shape != ids.shape or dec.shape != ids.shape or z.shape != ids.shape or
                not np.isfinite(ra + dec).all() or (np.abs(dec) > 90).any() or
                (ra < 0).any() or (ra >= 360).any() or len(np.unique(ids)) != len(ids)):
            raise ValueError(f"{catalogue}: invalid coordinates, row shapes or duplicate source IDs")
        order = np.argsort(ids, kind="stable")
        warning = np.asarray(data.get("zwarning", np.zeros(len(ids), np.int64)), np.int64)
        final = np.asarray(data.get("is_qso_final", np.ones(len(ids), np.int64)), np.int64)
        if warning.shape != ids.shape or final.shape != ids.shape:
            raise ValueError(f"{catalogue}: spectroscopic flags do not match rows")
        parts.append(dict(catalogue=np.full(len(ids), catalogue), source_id=ids,
            input_row=np.arange(len(ids), dtype=np.int64), ra=ra, dec=dec, zspec=z,
            zwarning=warning, is_qso_final=final, sort=order))
    members = {key: np.concatenate([p[key][p["sort"]] for p in parts])
               for key in parts[0] if key != "sort"}
    n = len(members["ra"])
    if not n:
        raise ValueError("empty parent catalogues")
    ra, dec = np.deg2rad(members["ra"]), np.deg2rad(members["dec"])
    xyz = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    chord = 2*np.sin(np.deg2rad(radius_arcsec/3600)/2)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n,n))
    count, group = connected_components(graph.tocsr(), directed=False)
    valid_z = np.isfinite(members["zspec"]) & (members["zspec"] > 0)
    clean = valid_z & (members["zwarning"] == 0) & (members["is_qso_final"] == 1)
    # Lexicographic priority: valid z, clean spectrum, catalogue, identifier.
    order = np.lexsort((members["source_id"], members["catalogue"] == "sdss", ~clean, ~valid_z))
    preferred = np.full(count, n, dtype=np.int64)
    rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)
    np.minimum.at(preferred, group, rank)
    preferred = order[preferred]
    object_id = np.char.add(np.char.add(members["catalogue"][preferred], ":"), members["source_id"][preferred])
    # Sorting by adopted ID makes object order reproducible under input permutations.
    object_order = np.argsort(object_id, kind="stable")
    remap = np.empty(count, dtype=np.int64); remap[object_order] = np.arange(count)
    group = remap[group]; preferred = preferred[object_order]
    sizes = np.bincount(group, minlength=count)
    nz_min, nz_max = np.full(count,np.inf), np.full(count,-np.inf)
    np.minimum.at(nz_min,group[valid_z],members["zspec"][valid_z])
    np.maximum.at(nz_max,group[valid_z],members["zspec"][valid_z])
    dz = np.full(count,np.nan)
    have_z = np.isfinite(nz_min)
    dz[have_z] = (nz_max[have_z]-nz_min[have_z])/(1+nz_min[have_z])
    reference = xyz[preferred[group]]
    separation = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(xyz,reference),axis=1),
                                      np.einsum("ij,ij->i",xyz,reference)))*3600
    max_separation = np.zeros(count); np.maximum.at(max_separation,group,separation)
    n_desi = np.bincount(group[members["catalogue"]=="desi"],minlength=count)
    n_sdss = np.bincount(group[members["catalogue"]=="sdss"],minlength=count)
    objects = dict(object_id=object_id[object_order], ra=members["ra"][preferred],
        dec=members["dec"][preferred], zspec=np.where(valid_z[preferred],members["zspec"][preferred],np.nan),
        preferred_catalogue=members["catalogue"][preferred], preferred_source_id=members["source_id"][preferred],
        preferred_input_row=members["input_row"][preferred], n_members=sizes,
        n_desi=n_desi, n_sdss=n_sdss, spectroscopic_quality_clean=clean[preferred],
        redshift_conflict=dz>redshift_conflict_tolerance, redshift_span_scaled=dz,
        max_member_separation_arcsec=max_separation, extended_duplicate_group=max_separation>radius_arcsec)
    members["object_index"] = group
    order = np.argsort(group, kind="stable")
    members = {k:v[order] for k,v in members.items()}
    report = dict(input_desi=len(desi["ra"]),input_sdss=len(sdss["ra"]),input_rows=n,
        unique_objects=count,duplicate_memberships=n-count,position_pairs=len(pairs),
        desi_only=int(((n_desi>0)&(n_sdss==0)).sum()),sdss_only=int(((n_sdss>0)&(n_desi==0)).sum()),
        both_catalogues=int(((n_desi>0)&(n_sdss>0)).sum()),
        redshift_conflicts=int(objects["redshift_conflict"].sum()),
        extended_duplicate_groups=int(objects["extended_duplicate_group"].sum()),
        invalid_adopted_redshifts=int((~np.isfinite(objects["zspec"])).sum()),
        nonclean_adopted_spectra=int((~objects["spectroscopic_quality_clean"]).sum()),
        maximum_group_members=int(sizes.max()),memberships_preserved=len(members["source_id"])==n)
    return objects, members, report
