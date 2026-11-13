"""
GeoSplat stage 6 (optional) - vertical datum check and correction from independent GNSS/RTK checkpoints.

Always: reports whether the drone altitude is home-point referenced (abs_alt - rel_alt constant).
With --checkpoints CSV: extracts the model surface height at each checkpoint, estimates the vertical offset from
the non-vegetated (NVA) points, reports ASPRS Ed. 2 style statistics and writes height-corrected copies of the
georeferenced products to <project>/10_Vertical_Correction. Original files are never modified.

Checkpoint CSV columns (case-insensitive; aliases accepted, see geosplat/vcorr.py):
  id, E, N, h            (UTM of the project zone, ellipsoidal h)   - or -   id, lat, lon, h
  optional: class (NVA / VVA / EXCLUDE), desc, vrms
Heights must be ellipsoidal and in the same frame/epoch as the model (WGS84/ITRF at the flight date). Use
--ntrip-base-used/--ntrip-base-official and --velocity/--epoch-from/--epoch-to to harmonise RTK coordinates,
or put the same settings in a JSON file next to the CSV (same name, .json) so the app applies them in one run:
  {"ntrip_base_used": [X, Y, Z], "ntrip_base_official": [X, Y, Z],
   "velocity": [vx, vy, vz], "epoch_from": 2012.0, "epoch_to": "video"}
Command-line options override the JSON file.

Example:
  python scripts/stage6_vcorr.py --project F:\\ITS_GeoSplat --checkpoints cp.csv ^
      --ntrip-base-used -2443857.719,5835257.946,-808826.478 --ntrip-base-official -2443857.505,5835258.077,-808826.379 ^
      --velocity -0.024,-0.014,-0.011 --epoch-from 2012.0 --epoch-to video
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from geosplat import geodesy as g          # noqa: E402
from geosplat import gaussians as gs       # noqa: E402
from geosplat import vcorr as vc           # noqa: E402


def _xyz(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z")
    return v


def video_epoch(project):
    p = os.path.join(project, "02_Telemetry", "telemetry.csv")
    if os.path.exists(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("srt_datetime"):
                    return vc.decimal_year(r["srt_datetime"])
    return None


def _join_negative_triplets(argv):
    """Allow '--velocity -0.024,-0.014,-0.011' (argparse would read the value as an option)."""
    argv = list(argv)
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t in ("--velocity", "--ntrip-base-used", "--ntrip-base-official") and i + 1 < len(argv) \
                and argv[i + 1].startswith("-") and "," in argv[i + 1]:
            out.append(f"{t}={argv[i + 1]}")
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)      # live log lines in the GUI / CLI
    argv = _join_negative_triplets(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--checkpoints", default=None, help="checkpoint CSV (optional)")
    ap.add_argument("--radius", type=float, default=2.5, help="search radius for the model surface (m)")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--bandwidth", type=float, default=0.15, help="KDE bandwidth (m)")
    ap.add_argument("--canopy-threshold", type=float, default=3.0, help="exclude points this far above the offset (m)")
    ap.add_argument("--ntrip-base-used", type=_xyz, default=None, help="base ECEF X,Y,Z broadcast during RTK")
    ap.add_argument("--ntrip-base-official", type=_xyz, default=None, help="official ECEF X,Y,Z of that base")
    ap.add_argument("--velocity", type=_xyz, default=None, help="station ECEF velocity vx,vy,vz (m/yr)")
    ap.add_argument("--epoch-from", default=None, help="epoch of the checkpoint coordinates (decimal year)")
    ap.add_argument("--epoch-to", default="video", help="target epoch (decimal year or 'video')")
    ap.add_argument("--no-write", action="store_true", help="report only, do not write corrected products")
    a = ap.parse_args(argv)
    t0 = time.time()
    sidecar = None
    if a.checkpoints:
        sc = os.path.splitext(a.checkpoints)[0] + ".json"
        if os.path.exists(sc):
            cfg = json.load(open(sc, encoding="utf-8"))
            unknown = set(cfg) - {"ntrip_base_used", "ntrip_base_official", "velocity", "epoch_from", "epoch_to", "note"}
            if unknown:
                raise SystemExit(f"{sc}: unknown keys {sorted(unknown)}")
            for k in ("ntrip_base_used", "ntrip_base_official", "velocity"):
                if getattr(a, k) is None and cfg.get(k) is not None:
                    v = [float(x) for x in cfg[k]]
                    if len(v) != 3:
                        raise SystemExit(f"{sc}: {k} needs 3 values")
                    setattr(a, k, v)
            if a.epoch_from is None and cfg.get("epoch_from") is not None:
                a.epoch_from = str(cfg["epoch_from"])
            if "epoch_to" in cfg and a.epoch_to == "video":
                a.epoch_to = str(cfg["epoch_to"])
            sidecar = sc
            print(f"harmonisation settings from {os.path.basename(sc)}")
    P = os.path.abspath(a.project)
    O = os.path.join(P, vc.OUT_DIR)
    os.makedirs(O, exist_ok=True)
    report = {"software": "GeoSplat stage 6", "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    tel = os.path.join(P, "02_Telemetry", "telemetry.csv")
    if os.path.exists(tel):
        chk = vc.altitude_reference_check(tel)
        report["altitude_reference"] = chk
        if chk:
            print(f"abs_alt - rel_alt = {chk['abs_minus_rel_mean_m']:.3f} m (SD {chk['abs_minus_rel_sd_m']:.4f} m) -> "
                  + ("home-point referenced: absolute heights need ground control" if chk["home_point_referenced"]
                     else "independent altitude"))
    if not a.checkpoints:
        print("No checkpoint file given: height datum NOT corrected (diagnosis only).")
        json.dump(report, open(os.path.join(O, "vcorr_report.json"), "w"), indent=2)
        return 0

    tf = json.load(open(os.path.join(P, "06_Georeferencing", "transform.json")))
    st = json.load(open(os.path.join(P, "logs", "_state.json")))
    zone, south = st["zone"], st["south"]
    cp = vc.read_checkpoints(a.checkpoints, zone, south)
    lat, lon, h = cp["lat"], cp["lon"], cp["h"]
    harm = []
    if bool(a.ntrip_base_used) != bool(a.ntrip_base_official):
        raise SystemExit("give both the NTRIP base coordinate used and the official one")
    if a.ntrip_base_used and a.ntrip_base_official:
        lat, lon, h = vc.ntrip_base_correction(lat, lon, h, a.ntrip_base_used, a.ntrip_base_official)
        d = np.subtract(a.ntrip_base_official, a.ntrip_base_used)
        harm.append({"step": "NTRIP base re-reference", "delta_ecef_m": d.tolist()})
    if a.velocity:
        if a.epoch_from is None:
            raise SystemExit("--velocity needs --epoch-from")
        t_to = video_epoch(P) if a.epoch_to == "video" else float(a.epoch_to)
        if t_to is None:
            raise SystemExit("video epoch unknown; pass --epoch-to as a decimal year")
        lat, lon, h = vc.epoch_shift(lat, lon, h, a.velocity, float(a.epoch_from), t_to)
        harm.append({"step": "epoch shift", "velocity_ecef_m_per_yr": a.velocity,
                     "epoch_from": float(a.epoch_from), "epoch_to": round(t_to, 4)})
    E, N = g.latlon_to_utm(lat, lon, zone, south)[:2]
    report["checkpoint_harmonisation"] = harm
    report["harmonisation_settings_file"] = os.path.basename(sidecar) if sidecar else None

    # model surface: Gaussians (opacity-weighted) if present, else sparse SfM points
    X = os.path.join(P, "08_Exports")
    gfile = next((f for f in sorted(os.listdir(X)) if f.startswith("gaussians_utm_") and f.endswith("_offset.ply")),
                 None) if os.path.isdir(X) else None
    if gfile:
        crs = json.load(open(os.path.join(X, "gaussians_crs.json")))
        off = crs["files"][gfile]["add_offset_to_get_utm"]
        data, _ = gs.read_gs_ply(os.path.join(X, gfile))
        px = data["x"].astype(float) + off[0]
        py = data["y"].astype(float) + off[1]
        pz = data["z"].astype(float) + off[2]
        w = 1.0 / (1.0 + np.exp(-data["opacity"].astype(float)))
        src, min_w = gfile, a.min_opacity
        del data
    else:
        data, _ = gs.read_gs_ply(os.path.join(P, "06_Georeferencing", "sparse_points_utm.ply"))
        px, py, pz = (data[k].astype(float) for k in ("x", "y", "z"))
        w = np.ones_like(px)
        src, min_w = "sparse_points_utm.ply", 0.5
    zm, nsup = [], []
    for e, n in zip(E, N):
        z, k = vc.surface_height(px, py, pz, w, e, n, radius=a.radius, min_weight=min_w, bw=a.bandwidth)
        zm.append(z)
        nsup.append(k)
    zm = np.array(zm)
    dz = zm - h
    final, status, stats = vc.assess(dz, cp["cls"], cp["vrms"], a.canopy_threshold)
    nva = (final == vc.NVA) & np.isfinite(dz)
    if nva.sum() >= 2:
        e = vc.pairwise_errors(zm[nva], h[nva])
        stats["NVA"]["pairwise"] = {"pairs": int(len(e)), "RMS_m": float(np.sqrt(np.mean(e ** 2))),
                                    "mean_m": float(e.mean()), "max_abs_m": float(np.abs(e).max()),
                                    "note": "pairs share points: confirmatory, not independent"}
    b = stats["offset_m"]
    report.update({"model_surface_source": src, "radius_m": a.radius, "min_weight": min_w, "kde_bandwidth_m": a.bandwidth,
                   "statistics": stats, "correction_added_to_heights_m": -b})

    with open(os.path.join(O, "checkpoints_dz.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "desc", "E", "N", "h_checkpoint", "Z_model", "n_support", "dZ", "dZ_minus_offset", "class",
                     "status"])
        for i in range(len(h)):
            wr.writerow([cp["id"][i], cp["desc"][i], f"{E[i]:.3f}", f"{N[i]:.3f}", f"{h[i]:.3f}",
                         "" if np.isnan(zm[i]) else f"{zm[i]:.3f}", nsup[i],
                         "" if np.isnan(dz[i]) else f"{dz[i]:.3f}", "" if np.isnan(dz[i]) else f"{dz[i] - b:+.3f}",
                         final[i], status[i]])

    s = stats["NVA"]
    print(f"NVA n={s['n']}  offset b = {b:+.3f} m (SE {s['SE_offset_m']:.3f})  "
          f"RMSE_V after correction (LOO) = {s.get('RMSE_V_m', float('nan')):.3f} m")
    if "VVA" in stats:
        print(f"VVA n={stats['VVA']['n']}  RMSE_V after correction = {stats['VVA']['RMSE_V_m']:.3f} m")
    for w_ in stats["warnings"]:
        print("WARNING:", w_)

    if not a.no_write:
        info = {"shift_m": -b, "offset_m": b, "method": "mean dZ = Z_model - h_checkpoint over NVA checkpoints",
                "checkpoints": os.path.basename(a.checkpoints), "harmonisation": harm,
                "NVA": {k: v for k, v in s.items() if not isinstance(v, dict)},
                "horizontal": "unchanged, not validated by this step"}
        files = vc.apply_to_project(P, -b, info, log=print)
        report["written"] = [os.path.basename(f) for f in files]
        lines = ["GeoSplat - vertical datum correction", "=" * 36, f"created: {report['created']}",
                 f"correction: h_corrected = h_model {(-b):+.3f} m  (offset b = {b:+.3f} +/- {s['SE_offset_m']:.3f} m, "
                 f"n = {s['n']} NVA checkpoints)",
                 f"after correction: NVA RMSE_V (leave-one-out) = {s.get('RMSE_V_m', float('nan')):.3f} m",
                 "height reference: ellipsoidal (frame/epoch of the checkpoints). Not orthometric/MSL.",
                 "horizontal coordinates: unchanged and not validated by this step.",
                 "originals in folders 01-09 are not modified.", "", "files:"] + [f"  {x}" for x in report["written"]]
        open(os.path.join(O, "README.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(report, open(os.path.join(O, "vcorr_report.json"), "w"), indent=2, default=float)
    print(f"done in {time.time() - t0:.1f} s -> {O}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
