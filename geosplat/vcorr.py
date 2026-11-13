"""
Vertical datum correction of a GeoSplat project from independent GNSS/RTK checkpoints.

Why this is needed
------------------
GeoSplat georeferences the SfM model with a similarity transform onto the drone GNSS track. Its residuals
measure how consistent the two trajectories are; an error shared by every GNSS position is absorbed by the
transform and is invisible in them. On DJI aircraft the per-frame ``abs_alt`` is the home-point altitude set at
take-off plus ``rel_alt``; a wrong home-point altitude therefore becomes one constant height offset for the whole
model (25.4 m in the ITS Research Center data set). Only independent ground control can reveal and remove it.

What this module does (see docs/METHODS.md section 9)
-----------------------------------------------------
1. ``read_checkpoints``      - read a checkpoint CSV (UTM E/N/h or lat/lon/h), optional class and VRMS columns.
2. ``ntrip_base_correction`` - optional: re-reference RTK coordinates from the base coordinate broadcast over NTRIP
                               to the official coordinate of that base (ECEF translation).
3. ``epoch_shift``           - optional: move coordinates between epochs with a station velocity (ECEF, m/yr).
4. ``surface_height``        - model surface height at a checkpoint: opacity-weighted kernel-density mode of the
                               Gaussian centres (or point heights) within a horizontal radius.
5. ``assess``                - offset estimate b = mean dZ of the non-vegetated (NVA) class, leave-one-out RMSE,
                               vegetated (VVA) class corrected with the NVA offset, robust measures (median, NMAD),
                               bootstrap intervals, datum-free pairwise height-difference test. Follows the
                               reporting of the ASPRS Positional Accuracy Standards, Edition 2 (2024).
6. ``altitude_reference_check`` - flags when abs_alt - rel_alt is constant (home-point referenced altitude).
7. ``apply_to_project``      - writes height-corrected copies of the georeferenced products to
                               <project>/10_Vertical_Correction. Original files are never modified.

Only the vertical datum is corrected. Horizontal coordinates are left unchanged.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os

import numpy as np

from . import geodesy as g

NVA, VVA, EXCLUDED = "NVA", "VVA", "EXCLUDED"
OUT_DIR = "10_Vertical_Correction"


# --------------------------------------------------------------------------- checkpoints
_ALIASES = {
    "id": ("id", "name", "point", "pt", "nama", "titik"),
    "e": ("e", "easting", "utm_e", "east", "x"),
    "n": ("n", "northing", "utm_n", "north", "y"),
    "h": ("h", "height", "h_ell", "h_m", "ellipsoidal_height", "elevation", "z"),
    "lat": ("lat", "latitude", "lat_deg"),
    "lon": ("lon", "long", "longitude", "lon_deg"),
    "cls": ("class", "cls", "category", "kelas"),
    "desc": ("desc", "description", "code", "keterangan"),
    "vrms": ("vrms", "sigma_h", "sigma_v"),
}


def _col(header, key):
    low = {h.strip().lower(): h for h in header}
    for a in _ALIASES[key]:
        if a in low:
            return low[a]
    return None


def read_checkpoints(path, zone, south):
    """Read checkpoints from CSV.

    Required: an ID column and either (E, N, h) in the project's UTM zone or (lat, lon, h) in decimal degrees.
    Heights must be ellipsoidal (WGS84/ITRF-aligned). Optional: class (NVA / VVA / EXCLUDE), description, VRMS.
    Returns a dict of NumPy arrays: id, E, N, h, lat, lon, cls, desc, vrms (NaN when absent).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    hdr = list(rows[0].keys())
    c = {k: _col(hdr, k) for k in _ALIASES}
    if c["h"] is None:
        raise ValueError(f"{path}: no height column (expected one of {_ALIASES['h']})")
    ids = [r[c["id"]].strip() if c["id"] else str(i + 1) for i, r in enumerate(rows)]
    h = np.array([float(r[c["h"]]) for r in rows])
    if c["e"] and c["n"]:
        E = np.array([float(r[c["e"]]) for r in rows])
        N = np.array([float(r[c["n"]]) for r in rows])
        lat, lon = g.utm_to_latlon(E, N, zone, south)
    elif c["lat"] and c["lon"]:
        lat = np.array([float(r[c["lat"]]) for r in rows])
        lon = np.array([float(r[c["lon"]]) for r in rows])
        E, N = g.latlon_to_utm(lat, lon, zone, south)[:2]
    else:
        raise ValueError(f"{path}: need E/N or lat/lon columns")
    cls = [(r[c["cls"]].strip().upper() if c["cls"] else "") for r in rows]
    cls = [EXCLUDED if x.startswith("EXCL") else x for x in cls]
    desc = [(r[c["desc"]].strip() if c["desc"] else "") for r in rows]
    vrms = np.array([float(r[c["vrms"]]) if c["vrms"] and r[c["vrms"]].strip() else np.nan for r in rows])
    return {"id": np.array(ids), "E": np.asarray(E, float), "N": np.asarray(N, float), "h": h,
            "lat": np.asarray(lat, float), "lon": np.asarray(lon, float), "cls": np.array(cls, dtype=object),
            "desc": np.array(desc, dtype=object), "vrms": vrms}


def ntrip_base_correction(lat, lon, h, base_used_xyz, base_official_xyz):
    """X_rover(official) = X_rover(RTK) - X_base(used) + X_base(official), in ECEF.

    RTK measures the baseline from the reference coordinate the base broadcasts. If that coordinate differs from
    the official datum coordinate of the station, every rover position inherits the difference.
    """
    x, y, z = g.geodetic_to_ecef(lat, lon, h)
    d = np.asarray(base_official_xyz, float) - np.asarray(base_used_xyz, float)
    return g.ecef_to_geodetic(x + d[0], y + d[1], z + d[2])


def epoch_shift(lat, lon, h, vel_xyz, t_from, t_to):
    """Move coordinates from epoch t_from to t_to (decimal years) with a constant ECEF velocity (m/yr)."""
    x, y, z = g.geodetic_to_ecef(lat, lon, h)
    v = np.asarray(vel_xyz, float) * (float(t_to) - float(t_from))
    return g.ecef_to_geodetic(x + v[0], y + v[1], z + v[2])


def decimal_year(d):
    """datetime/date or 'YYYY-MM-DD[...]' string -> decimal year (mid-day of that date)."""
    if isinstance(d, str):
        d = _dt.date.fromisoformat(d[:10])
    if isinstance(d, _dt.datetime):
        d = d.date()
    y0 = _dt.date(d.year, 1, 1)
    n = (_dt.date(d.year + 1, 1, 1) - y0).days
    return d.year + ((d - y0).days + 0.5) / n


# --------------------------------------------------------------------------- model surface height
def kde_mode(z, w=None, bw=0.15, step=0.02):
    """Mode of a weighted Gaussian kernel density of heights z (bandwidth bw, grid step `step`)."""
    z = np.asarray(z, float)
    w = np.ones_like(z) if w is None else np.asarray(w, float)
    grid = np.arange(z.min() - 3 * bw, z.max() + 3 * bw + step, step)
    dens = np.zeros(len(grid))
    for i in range(0, len(z), 4096):                       # chunked: bounded memory for dense neighbourhoods
        zz, ww = z[i:i + 4096], w[i:i + 4096]
        dens += (np.exp(-0.5 * ((grid[:, None] - zz[None, :]) / bw) ** 2) * ww[None, :]).sum(1)
    return float(grid[int(np.argmax(dens))])


def surface_height(px, py, pz, weight, E, N, radius=2.5, fallback_radius=4.0, min_weight=0.3, bw=0.15, min_n=5):
    """Model surface height at (E, N): KDE mode of points within `radius` (or `fallback_radius` if fewer than
    `min_n`) whose weight (Gaussian opacity, or 1 for plain points) exceeds `min_weight`.
    Returns (height or NaN, number of supporting points)."""
    dx, dy = px - E, py - N
    near = (np.abs(dx) < fallback_radius) & (np.abs(dy) < fallback_radius)
    d = np.hypot(dx[near], dy[near])
    zz, ww = pz[near], weight[near]
    for r in (radius, fallback_radius):
        m = (d < r) & (ww > min_weight)
        if m.sum() >= min_n:
            return kde_mode(zz[m], ww[m], bw), int(m.sum())
    return float("nan"), int(((d < fallback_radius) & (ww > min_weight)).sum())


# --------------------------------------------------------------------------- statistics
def nmad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def loo_residuals(dz):
    """e_i = dz_i - mean(dz_j, j != i): residual of each point after an offset estimated without it."""
    dz = np.asarray(dz, float)
    n = len(dz)
    return (dz - dz.mean()) * n / (n - 1)


def _boot(x, fn, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.choice(x, len(x))) for _ in range(n)]
    vals = [v for v in vals if np.isfinite(v)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def pairwise_errors(model_h, ref_h):
    """Datum-free check: e_ij = (Z_i - Z_j) - (h_i - h_j) for all pairs i < j. A constant offset cancels."""
    i, j = np.triu_indices(len(model_h), 1)
    return (model_h[i] - model_h[j]) - (ref_h[i] - ref_h[j])


def assess(dz, cls, vrms=None, canopy_threshold=3.0):
    """Classify checkpoints and compute the vertical-accuracy statistics.

    dz   : Z_model - h_checkpoint (NaN = no model coverage)
    cls  : per-point label 'NVA', 'VVA', 'EXCLUDED' or '' (unlabelled -> treated as NVA candidate)
    vrms : checkpoint survey VRMS (optional; included in RMSE_V as sqrt(RMSE_data^2 + RMSE_survey^2))
    Points whose dz exceeds the provisional offset by more than `canopy_threshold` m are excluded as
    'canopy top' (the model surface there is not the ground the checkpoint measured), unless labelled NVA,
    in which case a warning is issued.
    Returns (final_class array, status array, stats dict).
    """
    dz = np.asarray(dz, float)
    n = len(dz)
    vrms = np.full(n, np.nan) if vrms is None else np.asarray(vrms, float)
    lab = np.array([c if c in (NVA, VVA, EXCLUDED) else "" for c in cls], dtype=object)
    final = np.where(lab == "", NVA, lab).astype(object)
    status = np.array(["ok"] * n, dtype=object)
    warnings = []
    final[~np.isfinite(dz)] = EXCLUDED
    status[~np.isfinite(dz)] = "no model coverage"
    status[(lab == EXCLUDED) & np.isfinite(dz)] = "excluded by user"
    cand = (final == NVA) & np.isfinite(dz)
    if cand.sum() == 0:
        raise ValueError("no usable NVA checkpoint (label points NVA or leave the class empty)")
    b0 = float(np.median(dz[cand]))
    for k in range(n):
        if np.isfinite(dz[k]) and final[k] != EXCLUDED and dz[k] - b0 > canopy_threshold:
            if lab[k] == NVA:
                warnings.append(f"checkpoint {k}: labelled NVA but {dz[k] - b0:+.2f} m above the provisional offset")
            else:
                final[k], status[k] = EXCLUDED, "canopy top"
    nva = (final == NVA) & np.isfinite(dz)
    d = dz[nva]
    b = float(d.mean())
    st = {"offset_m": b, "n_total": int(n), "warnings": warnings, "canopy_threshold_m": canopy_threshold}

    def surv(mask):
        v = vrms[mask]
        return _rms(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0

    s_n = {"n": int(nva.sum()), "ME_m": b, "SD_m": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
           "SE_offset_m": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
           "RMSE_uncorrected_m": _rms(d), "median_m": float(np.median(d)), "NMAD_m": nmad(d),
           "RMSE_survey_m": surv(nva)}
    if len(d) >= 3:
        e = loo_residuals(d)
        s_n["RMSE_LOO_m"] = _rms(e)
        s_n["RMSE_V_m"] = float(np.hypot(s_n["RMSE_LOO_m"], s_n["RMSE_survey_m"]))
        s_n["max_abs_LOO_m"] = float(np.abs(e).max())
        s_n["CI95_RMSE_LOO_m"] = _boot(d, lambda s: _rms(loo_residuals(s)) if np.ptp(s) > 0 else np.nan)
    else:
        warnings.append("fewer than 3 NVA checkpoints: no leave-one-out statistics")
    if len(d) < 30:
        warnings.append(f"{len(d)} NVA checkpoints: below the 30 required by ASPRS Ed. 2 (limited assessment)")
    st["NVA"] = s_n
    vva = (final == VVA) & np.isfinite(dz)
    if vva.any():
        v = dz[vva] - b
        st["VVA"] = {"n": int(vva.sum()), "ME_after_correction_m": float(v.mean()), "RMSE_after_correction_m": _rms(v),
                     "median_m": float(np.median(v)), "NMAD_m": nmad(v), "RMSE_survey_m": surv(vva),
                     "RMSE_V_m": float(np.hypot(_rms(v), surv(vva))),
                     "CI95_RMSE_m": _boot(v, _rms) if len(v) > 1 else None}
    st["warnings"] = warnings
    return final, status, st


def altitude_reference_check(telemetry_csv):
    """Is abs_alt = constant + rel_alt (home-point referenced)? Returns a dict, or None if columns are missing."""
    import pandas as pd
    t = pd.read_csv(telemetry_csv)
    a = next((c for c in ("abs_alt", "djmd_abs_alt_m") if c in t), None)
    r = next((c for c in ("rel_alt", "djmd_rel_alt_m") if c in t), None)
    if not a or not r:
        return None
    d = (t[a] - t[r]).dropna().to_numpy(float)
    if len(d) < 2:
        return None
    sd = float(d.std())
    return {"abs_minus_rel_mean_m": float(d.mean()), "abs_minus_rel_sd_m": sd, "n": int(len(d)),
            "home_point_referenced": bool(sd < 0.05),
            "note": ("abs_alt equals a fixed home-point altitude plus rel_alt; any error of the home-point altitude "
                     "is a constant height offset of the whole model - check with ground control.") if sd < 0.05 else
                    "abs_alt varies independently of rel_alt."}


# --------------------------------------------------------------------------- products
def _read_header(path):
    f = open(path, "rb")
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        line = f.readline()
        if not line:
            f.close()
            raise ValueError(f"{path}: not a PLY file")
        hdr += line
    return hdr.decode("latin1"), f


def _header_with_comment(hdr, comment, drop_prefixes=()):
    out = []
    for l in hdr.split("\n"):
        if l.startswith("comment") and any(p in l for p in drop_prefixes):
            continue
        out.append(l)
        if l.startswith("format"):
            out.append("comment " + comment)
    return "\n".join(out)


def _ply_shift_z(src, dst, dh, comment):
    """Copy a binary PLY, adding dh to property z (float or double). Header gets an explanatory comment."""
    hdr, f = _read_header(src)
    props = [l.split() for l in hdr.split("\n") if l.startswith("property")]
    tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
    dt = np.dtype([(p[2], tmap[p[1]]) for p in props])
    n = int(next(l for l in hdr.split("\n") if l.startswith("element vertex")).split()[-1])
    data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n).copy()
    f.close()
    data["z"] = data["z"] + np.asarray(dh, data.dtype["z"])
    with open(dst, "wb") as o:
        o.write(_header_with_comment(hdr, comment, ("abs_alt-referenced",)).encode("latin1"))
        o.write(data.tobytes())
    return n


def _ply_relabel_origin(src, dst, h_old, h_new):
    """Copy a local-ENU PLY unchanged except for the origin height written in its header comments."""
    hdr, f = _read_header(src)
    tag = f"h {h_old:.3f}"
    if tag not in hdr:
        f.close()
        return False
    hdr = hdr.replace(tag, f"h {h_new:.3f} (ellipsoidal; vertical datum corrected, ENU values unchanged)")
    with open(dst, "wb") as o:
        o.write(hdr.encode("latin1"))
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            o.write(b)
    f.close()
    return True


def apply_to_project(project, dh, info, log=print):
    """Write height-corrected copies of the georeferenced products to <project>/10_Vertical_Correction.

    dh   : correction added to heights (= -offset)
    info : dict stored in the corrected CRS sidecar (method, statistics)
    Returns the list of written files. Source files are only read.
    """
    O = os.path.join(project, OUT_DIR)
    os.makedirs(O, exist_ok=True)
    G, X = os.path.join(project, "06_Georeferencing"), os.path.join(project, "08_Exports")
    written = []
    note = f"vertical datum corrected by {dh:+.3f} m from GNSS checkpoints; ellipsoidal heights"
    tf = json.load(open(os.path.join(G, "transform.json")))
    h0 = float(tf["enu_origin"]["h_m"])
    utm_label = tf["conventions"]["utm"]

    def out(name):
        written.append(os.path.join(O, name))
        return written[-1]

    # absolute-height products: shift z
    for src in ("sparse_points_utm.ply", "camera_centres_utm.ply"):
        p = os.path.join(G, src)
        if os.path.exists(p):
            _ply_shift_z(p, out(src.replace(".ply", "_hcorr.ply")), dh, f"{utm_label}; {note}")
    for name in sorted(os.listdir(X)) if os.path.isdir(X) else []:
        if name.startswith("gaussians_utm_") and name.endswith(".ply"):
            _ply_shift_z(os.path.join(X, name), out(name.replace(".ply", "_hcorr.ply")), dh,
                         f"{utm_label} with false origin (see gaussians_crs_hcorr.json); {note}")
    # local ENU products: values unchanged, origin height relabelled
    for d, src in ((G, "sparse_points_enu.ply"), (G, "sparse_points_enu_clean.ply"), (X, "gaussians_enu.ply")):
        p = os.path.join(d, src)
        if os.path.exists(p) and not _ply_relabel_origin(p, out(src.replace(".ply", "_hcorr.ply")), h0, h0 + dh):
            written.pop()
            log(f"  {src}: origin height tag not found in header - skipped")
    # camera trajectory
    import pandas as pd
    p = os.path.join(X, "camera_trajectory.csv")
    if os.path.exists(p):
        cam = pd.read_csv(p)
        cam["h_m"] = cam["h_m"] + dh
        if {"x_ecef", "y_ecef", "z_ecef"} <= set(cam.columns):
            cam["x_ecef"], cam["y_ecef"], cam["z_ecef"] = g.geodetic_to_ecef(cam["lat_deg"], cam["lon_deg"], cam["h_m"])
        cam.to_csv(out("camera_trajectory_hcorr.csv"), index=False)
    p = os.path.join(X, "camera_trajectory.geojson")
    if os.path.exists(p):
        gj = json.load(open(p))

        def shift(c):
            if isinstance(c, list) and c and isinstance(c[0], (int, float)):
                return c[:2] + [c[2] + dh] + c[3:] if len(c) >= 3 else c
            return [shift(x) for x in c] if isinstance(c, list) else c
        for ft in gj.get("features", []):
            ft["geometry"]["coordinates"] = shift(ft["geometry"]["coordinates"])
            ft.setdefault("properties", {})["height_reference"] = note
        json.dump(gj, open(out("camera_trajectory_hcorr.geojson"), "w"))
    # CRS sidecar
    p = os.path.join(X, "gaussians_crs.json")
    if os.path.exists(p):
        crs = json.load(open(p))
        files = crs.get("files", {})
        if "gaussians_enu.ply" in files:
            o = files["gaussians_enu.ply"]["origin"]
            o["h_m"] = round(o["h_m"] + dh, 3)
            o["ecef_m"] = [float(v) for v in g.geodetic_to_ecef(o["lat_deg"], o["lon_deg"], o["h_m"])]
            files["gaussians_enu_hcorr.ply"] = files.pop("gaussians_enu.ply")
        for k in [k for k in files if k.startswith("gaussians_utm_") and k.endswith(".ply")]:
            files[k]["from_enu"]["t"][2] += dh
            files[k.replace(".ply", "_hcorr.ply")] = files.pop(k)
        crs["height_reference"] = "Ellipsoidal height, vertical datum fixed to the GNSS checkpoints (see vertical_correction)."
        crs["vertical_correction"] = info
        json.dump(crs, open(out("gaussians_crs_hcorr.json"), "w"), indent=2)
    p = os.path.join(X, "web", "meta.json")
    if os.path.exists(p):
        m = json.load(open(p))
        m["origin"]["h"] = round(m["origin"]["h"] + dh, 3)
        if "ground_h" in m:
            m["ground_h"] = round(m["ground_h"] + dh, 3)
        m["vertical_correction_m"] = dh
        json.dump(m, open(out("web_meta_hcorr.json"), "w"), indent=2)
    return written


def self_test():
    rng = np.random.default_rng(0)
    # synthetic flat ground at h=31.7 seen by a model with a +25 m datum offset and 0.1 m noise
    px, py = rng.uniform(0, 100, 20000), rng.uniform(0, 100, 20000)
    pz = 31.7 + 25.0 + rng.normal(0, 0.1, px.size)
    w = np.full(px.size, 0.9)
    E, N = rng.uniform(10, 90, 10), rng.uniform(10, 90, 10)
    zm = np.array([surface_height(px, py, pz, w, e, n)[0] for e, n in zip(E, N)])
    dz = zm - 31.7
    final, status, st = assess(dz, [""] * len(dz))
    assert abs(st["offset_m"] - 25.0) < 0.05, st["offset_m"]
    assert st["NVA"]["RMSE_LOO_m"] < 0.1
    assert np.allclose(pairwise_errors(zm, np.full(len(zm), 31.7)).mean(), 0, atol=0.05)
    return True
