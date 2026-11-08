#!/usr/bin/env python3
"""
GeoSplat - Phase 3: GNSS georeferencing of the COLMAP sparse model.

  1. GNSS<->image time-offset scan (position RMSE)      + compass/gimbal heading-bias check
  2. Robust Sim3  COLMAP-local -> local ENU  (RANSAC + least-squares refit on inliers)
  3. 5-fold check-point cross-validation (independent accuracy of the fit)
  4. CRS chain: ENU -> ECEF -> WGS84 -> UTM; composite local->ECEF Sim3; ENU->UTM similarity
  5. Exports: georeferenced COLMAP model (ENU), PLYs (local / ENU / UTM), camera table,
     transform.json sidecar, residual report.
The local COLMAP model is never modified (kept as backup in 05_Sparse_Reconstruction).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, HERE)
from geosplat import colmap_io as ci  # noqa: E402
from geosplat import geodesy as g  # noqa: E402
from geosplat import georef as gr  # noqa: E402
from geosplat import reports, sim3  # noqa: E402
from geosplat.sync import SyncConfig, TelemetrySync  # noqa: E402


def heading_of(Rwc):
    """Camera optical axis (+Z_cam) in world: yaw (deg from North, clockwise) and pitch (deg)."""
    z = Rwc.T @ np.array([0.0, 0.0, 1.0])
    return np.degrees(np.arctan2(z[0], z[1])), np.degrees(np.arcsin(np.clip(z[2], -1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--model", default="0")
    ap.add_argument("--inlier-m", type=float, default=3.0)
    ap.add_argument("--offset-range", type=float, default=0.5)
    ap.add_argument("--offset-step", type=float, default=0.01)
    ap.add_argument("--apply-offset", default="auto", help="'auto' (scan result if significant), or seconds")
    ap.add_argument("--display-source", default=None)
    args = ap.parse_args()

    src = os.path.abspath(args.project)
    O = src
    D = os.path.join(O, "06_Georeferencing")
    os.makedirs(D, exist_ok=True)
    log = logging.getLogger("geosplat3")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(O, "logs", "phase3.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    log.addHandler(sh)
    t_all = time.time()

    # ------------------------------------------------------------------ inputs
    st = json.load(open(os.path.join(O, "logs", "_state.json")))
    tel = pd.read_pickle(os.path.join(O, "02_Telemetry", "_telemetry.pkl"))
    txt = os.path.join(O, "05_Sparse_Reconstruction", "sparse_txt", args.model)
    cams = ci.read_cameras_txt(os.path.join(txt, "cameras.txt"))
    imgs = ci.read_images_txt(os.path.join(txt, "images.txt"))
    xyz, rgb, perr, tlen = ci.read_points3d_txt(os.path.join(txt, "points3D.txt"))
    poses = pd.read_csv(os.path.join(O, "05_Sparse_Reconstruction", "camera_poses.csv"))
    name2id = {v["name"]: k for k, v in imgs.items()}
    poses = poses[poses["image"].isin(name2id)].sort_values("pts_s").reset_index(drop=True)
    Cl = np.array([imgs[name2id[n]]["center"] for n in poses["image"]])
    Rl = [imgs[name2id[n]]["R"] for n in poses["image"]]
    tq = poses["pts_s"].to_numpy()
    log.info("=== GeoSplat Phase 3 | model %s: %d cameras, %d points", args.model, len(Cl), len(xyz))

    sync0 = TelemetrySync(tel, SyncConfig(offset_s=0.0))
    lat0, lon0, h0 = sync0.ref
    zone, south, epsg = st["zone"], st["south"], st["epsg"]

    def gnss_at(offset):
        s_ = TelemetrySync(tel, SyncConfig(offset_s=offset))
        q = s_.query(tq)
        return q[["enu_e_m", "enu_n_m", "enu_u_m"]].to_numpy(), q

    # ------------------------------------------------------------------ 1. offset scan
    G0, q0 = gnss_at(0.0)
    ok0 = np.isfinite(G0).all(1)
    fit0 = sim3.ransac_sim3(Cl[ok0], G0[ok0], thresh_m=args.inlier_m)
    inl_fixed = np.zeros(len(Cl), bool)
    inl_fixed[np.where(ok0)[0][fit0["inliers"]]] = True
    offs = np.round(np.arange(-args.offset_range, args.offset_range + 1e-9, args.offset_step), 6)
    rows = []
    for o in offs:
        G, _ = gnss_at(o)
        m = inl_fixed & np.isfinite(G).all(1)
        s, R, t = sim3.umeyama(Cl[m], G[m])
        r = sim3.residuals(s, R, t, Cl[m], G[m])
        # heading check: aligned COLMAP optical-axis yaw vs gimbal yaw at the same instant
        gy = TelemetrySync(tel, SyncConfig(offset_s=o)).query(tq)["gimbal_yaw_deg"].to_numpy()
        cy = np.array([heading_of(Rw @ R.T)[0] for Rw in Rl])
        dy = g.wrap180(cy - gy)
        rows.append({"offset_s": o, "rmse_m": float(np.sqrt(np.mean(r ** 2))), "n": int(m.sum()),
                     "sim3_yaw_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                     "heading_bias_deg": float(np.mean(dy[m])), "heading_std_deg": float(np.std(dy[m]))})
    scan = pd.DataFrame(rows)
    scan.to_csv(os.path.join(D, "time_offset_scan.csv"), index=False)
    k = int(scan["rmse_m"].idxmin())
    o_best = float(scan["offset_s"][k])
    if 0 < k < len(scan) - 1:                      # parabolic refinement
        y0, y1, y2 = scan["rmse_m"][k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            o_best += 0.5 * (y0 - y2) / den * args.offset_step
    r_at0 = float(scan.loc[np.isclose(scan["offset_s"], 0.0), "rmse_m"].iloc[0])
    r_best = float(scan["rmse_m"].min())
    gain = r_at0 - r_best
    # significance: improvement must exceed 5 % of RMSE and 2 cm, and not sit at the scan edge
    significant = gain > max(0.05 * r_at0, 0.02) and 0 < k < len(scan) - 1
    if args.apply_offset == "auto":
        offset = round(o_best, 3) if significant else 0.0
    else:
        offset = float(args.apply_offset)
    log.info("Offset scan: RMSE(0)=%.3f m, min %.3f m at %+.3f s (gain %.3f m) -> applied offset %+.3f s",
             r_at0, r_best, o_best, gain, offset)

    # ------------------------------------------------------------------ 2. final Sim3
    G, q = gnss_at(offset)
    ok = np.isfinite(G).all(1)
    fit = sim3.ransac_sim3(Cl[ok], G[ok], thresh_m=args.inlier_m)
    s, R, t = fit["scale"], fit["R"], fit["t"]
    inl = np.zeros(len(Cl), bool)
    inl[np.where(ok)[0][fit["inliers"]]] = True
    Ce = sim3.apply_sim3(s, R, t, Cl)
    res = Ce - G
    log.info("Sim3: scale %.6f m/unit, RMSE inliers %.3f m (%d/%d), rejected %d",
             s, fit["rmse_inliers"], fit["n_inliers"], ok.sum(), ok.sum() - fit["n_inliers"])

    # ------------------------------------------------------------------ 3. check points (5-fold CV)
    idx = np.where(inl)[0]
    cv = np.full(len(Cl), np.nan)
    for f in range(5):
        test = idx[f::5]
        train = np.setdiff1d(idx, test)
        s_, R_, t_ = sim3.umeyama(Cl[train], G[train])
        cv[test] = np.linalg.norm(sim3.apply_sim3(s_, R_, t_, Cl[test]) - G[test], axis=1)
    cv_rmse = float(np.sqrt(np.nanmean(cv ** 2)))
    log.info("5-fold check-point RMSE: %.3f m", cv_rmse)

    # heading consistency at the applied offset
    cam_yaw, cam_pitch = np.array([heading_of(Rw @ R.T) for Rw in Rl]).T
    dyaw = g.wrap180(cam_yaw - q["gimbal_yaw_deg"].to_numpy())
    dpitch = cam_pitch - q["gimbal_pitch_deg"].to_numpy()

    # ------------------------------------------------------------------ 4. CRS chain
    sE, RE, tE = gr.compose_local_to_ecef(s, R, t, lat0, lon0, h0)
    Cecef = sim3.apply_sim3(sE, RE, tE, Cl)
    lat, lon, h = g.ecef_to_geodetic(*Cecef.T)
    E, N, _, _ = g.latlon_to_utm(lat, lon, zone, south)
    # consistency of the two routes (local->ENU->geodetic  vs  local->ECEF->geodetic)
    lat_b, lon_b, h_b = g.enu_to_geodetic(Ce[:, 0], Ce[:, 1], Ce[:, 2], lat0, lon0, h0)
    route_err = float(np.max(np.abs(np.r_[(lat - lat_b) * 111320, (lon - lon_b) * 111320 * np.cos(np.radians(lat0)),
                                          h - h_b])))
    pts_enu = sim3.apply_sim3(s, R, t, xyz)
    good = (perr < 2.0) & (tlen >= 3)
    # ENU->UTM similarity is fitted on the CORE scene (cameras + points within 1.5x the camera
    # footprint radius). Far background points (km away) would be dominated by Earth curvature
    # (d^2/2R) and are not part of the reconstruction target.
    cen = Ce[:, :2].mean(0)
    r_cam = float(np.max(np.hypot(Ce[:, 0] - cen[0], Ce[:, 1] - cen[1])))
    r_core = 1.5 * r_cam
    d_pts = np.hypot(pts_enu[:, 0] - cen[0], pts_enu[:, 1] - cen[1])
    core = good & (d_pts <= r_core)
    sample = np.vstack([Ce, pts_enu[core][:: max(1, core.sum() // 20000)]])
    utm_sim = gr.fit_enu_to_utm_similarity(sample, lat0, lon0, h0, zone, south)
    far = good & (d_pts > r_core)
    if far.any():
        fu, _, _ = gr.enu_to_utm_exact(pts_enu[far], lat0, lon0, h0, zone, south)
        far_err = float(np.max(sim3.residuals(utm_sim["scale"], utm_sim["R"], utm_sim["t"], pts_enu[far],
                                              fu - utm_sim["offset"])))
    else:
        far_err = 0.0
    log.info("ENU->UTM similarity (core r<=%.0f m): k=%.8f, convergence %.4f deg, max misfit %.4f m "
             "(outside core: %.3f m)", r_core, utm_sim["scale"], utm_sim["grid_convergence_deg"],
             utm_sim["max_err_m"], far_err)

    # ------------------------------------------------------------------ 5. exports
    enu_model = os.path.join(D, "sparse_enu_txt")
    gr.transform_colmap_txt(txt, enu_model, s, R, t)
    # verify reprojection invariance on the written model
    imgs2 = ci.read_images_txt(os.path.join(enu_model, "images.txt"))
    cam = next(iter(cams.values()))
    fx, fy, cx, cy = cam["params"][:4]
    rng = np.random.default_rng(0)
    worst = 0.0
    for iid in rng.choice(list(imgs.keys()), 20, replace=False):
        a, b = imgs[iid], imgs2[iid]
        P = xyz[rng.choice(len(xyz), 200, replace=False)]
        Pa = (a["R"] @ P.T).T + a["tvec"]
        Pb = (b["R"] @ sim3.apply_sim3(s, R, t, P).T).T + b["tvec"]
        m = Pa[:, 2] > 0
        ua = np.c_[fx * Pa[m, 0] / Pa[m, 2] + cx, fy * Pa[m, 1] / Pa[m, 2] + cy]
        ub = np.c_[fx * Pb[m, 0] / Pb[m, 2] + cx, fy * Pb[m, 1] / Pb[m, 2] + cy]
        worst = max(worst, float(np.abs(ua - ub).max()) if m.any() else 0.0)
    cen_err = max(float(np.linalg.norm(imgs2[name2id[n]]["center"] - Ce[i])) for i, n in enumerate(poses["image"]))
    log.info("Georeferenced COLMAP model written; reprojection invariance max %.2e px, centre check %.2e m",
             worst, cen_err)

    utm_pts, _, _ = gr.enu_to_utm_exact(pts_enu, lat0, lon0, h0, zone, south)
    cm = [f"GeoSplat Phase 3", f"WGS84 local ENU origin lat {lat0:.9f} lon {lon0:.9f} h {h0:.3f}"]
    gr.write_ply(os.path.join(D, "sparse_points_local.ply"), xyz, rgb, comments=["COLMAP local frame (backup)"])
    gr.write_ply(os.path.join(D, "sparse_points_enu.ply"), pts_enu, rgb, comments=cm)
    gr.write_ply(os.path.join(D, "sparse_points_utm.ply"), utm_pts, rgb,
                 comments=[f"EPSG:{epsg} E N + DJI abs_alt-referenced height h (datum unconfirmed)"])
    gr.write_ply(os.path.join(D, "sparse_points_enu_clean.ply"), pts_enu[good], rgb[good],
                 comments=cm + ["filtered: reproj err < 2 px, track >= 3"])
    gr.write_ply(os.path.join(D, "camera_centres_utm.ply"), np.column_stack([E, N, h]),
                 np.tile(np.array([[235, 104, 52]], np.uint8), (len(E), 1)), comments=[f"EPSG:{epsg}"])

    out = poses[["image", "frame_index", "pts_s"]].copy()
    out["cx_local"], out["cy_local"], out["cz_local"] = Cl.T
    out["e_enu"], out["n_enu"], out["u_enu"] = Ce.T
    out["x_ecef"], out["y_ecef"], out["z_ecef"] = Cecef.T
    out["lat_deg"], out["lon_deg"], out["h_m"] = lat, lon, h
    out["utm_e"], out["utm_n"] = E, N
    out["gnss_e"], out["gnss_n"], out["gnss_u"] = G.T
    out["res_e_m"], out["res_n_m"], out["res_u_m"] = res.T
    out["res_h_m"] = np.hypot(res[:, 0], res[:, 1])
    out["res_3d_m"] = np.linalg.norm(res, axis=1)
    out["inlier"] = inl
    out["checkpoint_res_m"] = cv
    out["cam_yaw_deg"], out["cam_pitch_deg"] = cam_yaw, cam_pitch
    out["gimbal_yaw_deg"], out["gimbal_pitch_deg"] = q["gimbal_yaw_deg"], q["gimbal_pitch_deg"]
    out["d_yaw_deg"], out["d_pitch_deg"] = dyaw, dpitch
    out["sync_quality"] = q["sync_quality"]
    out.to_csv(os.path.join(D, "georeferenced_cameras.csv"), index=False, float_format="%.9f")

    ri = res[inl]
    stats = {
        "n_cameras": int(len(Cl)), "n_with_gnss": int(ok.sum()), "n_inliers": int(inl.sum()),
        "rejected": out.loc[ok & ~inl, "image"].tolist(), "no_gnss": out.loc[~ok, "image"].tolist(),
        "mean_m": dict(zip("ENU", np.round(ri.mean(0), 4).tolist())),
        "std_m": dict(zip("ENU", np.round(ri.std(0), 4).tolist())),
        "rmse_m": dict(zip("ENU", np.round(np.sqrt((ri ** 2).mean(0)), 4).tolist())),
        "rmse_h_m": float(np.sqrt(np.mean(ri[:, 0] ** 2 + ri[:, 1] ** 2))),
        "rmse_3d_m": float(np.sqrt(np.mean((ri ** 2).sum(1)))),
        "p95_3d_m": float(np.percentile(np.linalg.norm(ri, axis=1), 95)),
        "max_3d_m": float(np.linalg.norm(ri, axis=1).max()),
        "checkpoint_rmse_3d_m": cv_rmse,
    }
    tf = {
        "description": "GeoSplat georeferencing sidecar. Apply X_target = scale * R @ X_local + t.",
        "conventions": {
            "datum": "WGS84 (EPSG:4326 / ECEF EPSG:4978)",
            "local_enu": "East-North-Up tangent plane at origin; metres",
            "utm": f"WGS 84 / UTM zone {zone}{'S' if south else 'N'} (EPSG:{epsg})",
            "colmap_camera": "world-to-camera R,t; camera +Z forward, +X right, +Y down",
            "height_reference": "DJI 'abs_alt' from the aircraft (datum NOT stated by DJI: ellipsoidal vs "
                                "orthometric unconfirmed). Treated numerically as ellipsoidal height h for "
                                "ENU/ECEF. Verify with a ground control point before metric vertical use.",
            "accuracy_note": "Residuals measure consistency between SfM and the DJI GNSS track. A common GNSS "
                             "bias (typ. metre level without RTK) is NOT observable here; absolute accuracy "
                             "needs GCPs or RTK.",
        },
        "time_offset_s": offset,
        "enu_origin": {"lat_deg": lat0, "lon_deg": lon0, "h_m": h0,
                       "ecef_m": np.array(g.geodetic_to_ecef(lat0, lon0, h0), float).ravel().tolist()},
        "local_to_enu": {"scale": s, "R": R.tolist(), "t": t.tolist()},
        "local_to_ecef": {"scale": sE, "R": RE.tolist(), "t": tE.tolist()},
        "enu_to_utm_similarity": {"scale": utm_sim["scale"], "R": utm_sim["R"].tolist(),
                                  "t": utm_sim["t"].tolist(), "offset_EN_h": utm_sim["offset"].tolist(),
                                  "grid_convergence_deg": utm_sim["grid_convergence_deg"],
                                  "max_misfit_m": utm_sim["max_err_m"], "rms_misfit_m": utm_sim["rms_err_m"],
                                  "valid_region": {"centre_enu_en": cen.tolist(), "radius_m": r_core},
                                  "max_misfit_outside_region_m": far_err,
                                  "usage": "UTM = scale*R@enu + t + offset_EN_h (for Gaussian splats; points/cameras "
                                           "use the exact projection)"},
        "residuals": stats,
        "source_model": txt,
    }
    json.dump(tf, open(os.path.join(D, "transform.json"), "w"), indent=2)

    # ------------------------------------------------------------------ plots + report
    plot3(scan, offset, out, inl, D)
    for f in ("time_offset_scan.png", "residual_map.png", "residual_series.png"):
        shutil.copy2(os.path.join(D, f), os.path.join(O, "09_Reports", f))

    def disp(pth):
        if args.display_source and src in pth:
            return pth.replace(src, args.display_source.rstrip("\\/")).replace("/", "\\")
        return pth
    blocks = [
        ("INPUT", [("COLMAP model (local, kept as backup)", disp(txt)), ("cameras / points", f"{len(Cl)} / {len(xyz)}"),
                   ("GNSS", "DJI djmd track, 10 Hz, interpolated in ENU (Phase 1 sync engine)")]),
        ("TIME OFFSET (GNSS -> image)", [
            ("scan", f"{-args.offset_range:+.2f} .. {args.offset_range:+.2f} s, step {args.offset_step} s"),
            ("RMSE at 0 s (m)", round(r_at0, 4)), ("minimum RMSE (m)", round(r_best, 4)),
            ("offset at minimum (s, parabolic)", round(o_best, 4)), ("RMSE gain (m)", round(gain, 4)),
            ("significant (>5 % and >2 cm, not at edge)", bool(significant)),
            ("applied offset (s)", offset),
            ("heading bias at applied offset (deg)", round(float(np.mean(dyaw[inl])), 3)),
            ("RMSE at -0.5 / +0.5 s (m)", f"{scan['rmse_m'].iloc[0]:.3f} / {scan['rmse_m'].iloc[-1]:.3f}"),
            ("observability", "A constant shift on a circular orbit partly maps into a Sim3 rotation, but the "
                              "flight speed varies (0 -> 12.6 m/s), so the RMSE curve still has a clear minimum. "
                              "Resolution is ~1 video frame (33 ms)."),
            ("heading check", "camera yaw - gimbal yaw is ~flat across offsets on a constant-rate orbit (both "
                              "shift together), so it does NOT constrain the offset; it measures the "
                              "compass/gimbal yaw bias instead (see residuals).")]),
        ("SIM3  COLMAP-local -> local ENU", [
            ("scale (m per COLMAP unit)", round(s, 6)),
            ("rotation (deg about axis)", round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 4)),
            ("translation (m)", np.round(t, 3).tolist()),
            ("robust estimator", f"RANSAC 3-point, inlier threshold {args.inlier_m} m, LSQ refit"),
            ("inliers / with GNSS / total", f"{stats['n_inliers']} / {stats['n_with_gnss']} / {stats['n_cameras']}"),
            ("rejected as outliers", stats["rejected"] or "none"), ("no GNSS", stats["no_gnss"] or "none")]),
        ("RESIDUALS (aligned COLMAP - GNSS), inliers", [
            ("mean E / N / U (m)", stats["mean_m"]), ("std E / N / U (m)", stats["std_m"]),
            ("RMSE E / N / U (m)", stats["rmse_m"]), ("RMSE horizontal (m)", round(stats["rmse_h_m"], 4)),
            ("RMSE 3D (m)", round(stats["rmse_3d_m"], 4)), ("95th pct 3D (m)", round(stats["p95_3d_m"], 4)),
            ("max 3D (m)", round(stats["max_3d_m"], 4)),
            ("5-fold check-point RMSE 3D (m)", round(cv_rmse, 4)),
            ("heading: camera yaw - gimbal yaw (deg) mean / std",
             f"{np.mean(dyaw[inl]):.3f} / {np.std(dyaw[inl]):.3f}  (mean = compass/gimbal yaw bias vs SfM)"),
            ("pitch: camera - gimbal (deg) mean / std", f"{np.mean(dpitch[inl]):.3f} / {np.std(dpitch[inl]):.3f}")]),
        ("CRS CHAIN", [
            ("ENU origin (lat, lon, h)", f"{lat0:.9f}, {lon0:.9f}, {h0:.3f}"),
            ("local -> ECEF composite Sim3 scale", round(sE, 6)),
            ("route check ENU-vs-ECEF (max, m)", f"{route_err:.2e}"),
            ("UTM", f"zone {zone}{'S' if south else 'N'}, EPSG:{epsg}"),
            ("ENU -> UTM similarity scale k", round(utm_sim["scale"], 8)),
            ("grid convergence (deg)", round(utm_sim["grid_convergence_deg"], 5)),
            ("ENU -> UTM similarity fit region", f"core: cameras + points within {r_core:.0f} m of orbit centre"),
            ("ENU -> UTM similarity max / rms misfit in core (m)",
             f"{utm_sim['max_err_m']:.4f} / {utm_sim['rms_err_m']:.4f}"),
            ("ENU -> UTM similarity max misfit outside core (m)", f"{far_err:.3f} (background; use exact projection)"),
            ("height reference", tf["conventions"]["height_reference"])]),
        ("VERIFICATION", [
            ("reprojection invariance of written ENU model (max px)", f"{worst:.2e}"),
            ("camera centre check, written model vs computed (max m)", f"{cen_err:.2e}"),
            ("absolute accuracy", tf["conventions"]["accuracy_note"])]),
        ("OUTPUTS (06_Georeferencing)", [
            ("transform.json", "all transforms + conventions + residual stats"),
            ("georeferenced_cameras.csv", "local / ENU / ECEF / WGS84 / UTM + residuals per camera"),
            ("sparse_enu_txt/", "COLMAP model in metric local ENU (input for Phase 4)"),
            ("PLY", "sparse_points_local / _enu / _enu_clean / _utm, camera_centres_utm"),
            ("time_offset_scan.csv", "RMSE + heading bias per tested offset")]),
    ]
    reports.write_text_report(os.path.join(O, "09_Reports", "phase3_report.txt"), blocks)
    reports.write_html(os.path.join(O, "09_Reports", "phase3_report.html"), "GeoSplat · Phase 3 georeferencing", blocks,
                       [("Time-offset scan", os.path.join(D, "time_offset_scan.png")),
                        ("Residual vectors (exaggerated)", os.path.join(D, "residual_map.png")),
                        ("Residuals per axis", os.path.join(D, "residual_series.png"))])
    log.info("=== Phase 3 done in %.1f s", time.time() - t_all)


def plot3(scan, offset, out, inl, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

    def sty(a):
        a.grid(True, color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    ax[0].plot(scan["offset_s"], scan["rmse_m"], color=C1, lw=1.5)
    ax[0].axvline(offset, color=C2, lw=1, ls="--")
    ax[0].set_xlabel("time offset (s)")
    ax[0].set_ylabel("Sim3 position RMSE (m)")
    ax[0].set_title("Position RMSE vs offset", loc="left", fontsize=10)
    ax[1].plot(scan["offset_s"], scan["heading_bias_deg"], color=C1, lw=1.5, label="mean")
    ax[1].fill_between(scan["offset_s"], scan["heading_bias_deg"] - scan["heading_std_deg"],
                       scan["heading_bias_deg"] + scan["heading_std_deg"], color=C1, alpha=0.15, lw=0, label="±1 std")
    ax[1].axvline(offset, color=C2, lw=1, ls="--")
    ax[1].set_xlabel("time offset (s)")
    ax[1].set_ylabel("camera yaw − gimbal yaw (deg)")
    ax[1].set_title("Heading bias (compass/gimbal vs SfM)", loc="left", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        sty(a)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "time_offset_scan.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    k = 50.0
    ax.plot(out["gnss_e"], out["gnss_n"], color="#c3c2b7", lw=1)
    ax.quiver(out["gnss_e"][inl], out["gnss_n"][inl], out["res_e_m"][inl] * k, out["res_n_m"][inl] * k,
              angles="xy", scale_units="xy", scale=1, color=C1, width=0.002)
    ax.scatter(out["gnss_e"][inl], out["gnss_n"][inl], c=out["res_u_m"][inl], cmap="RdBu_r", vmin=-1, vmax=1, s=12,
               zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m, ENU)")
    ax.set_ylabel("North (m, ENU)")
    ax.set_title(f"Horizontal residual vectors ×{k:.0f} · dot colour = vertical residual (±1 m, red = up)",
                 loc="left", fontsize=9)
    sty(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "residual_map.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    for c, col, lab in (("res_e_m", C1, "East"), ("res_n_m", C2, "North"), ("res_u_m", C3, "Up")):
        ax.plot(out["pts_s"], out[c], color=col, lw=1.2, label=lab)
    ax.scatter(out["pts_s"], out["checkpoint_res_m"], s=6, color="#52514e", label="check-point |res| (CV)", zorder=3)
    ax.axhline(0, color="#9a9994", lw=0.6)
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("residual (m)")
    ax.set_title("Residuals per axis (aligned COLMAP − GNSS)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    sty(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "residual_series.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
