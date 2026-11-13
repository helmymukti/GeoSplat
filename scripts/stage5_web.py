#!/usr/bin/env python3
"""
GeoSplat - build web/GIS assets from the trained, georeferenced Gaussians (08_Exports/gaussians_enu.ply).

Outputs (08_Exports/web/):
  scene.splat        compact splat file (32 B/Gaussian: pos f32x3, scale f32x3, RGBA u8, quat u8x4),
                     core region only, sorted by visual importance (progressive loading)
  ortho.webp         top-down orthographic render of the Gaussians (z-buffer, opaque splats), north-up,
                     for a Google Maps GroundOverlay on satellite imagery
  trajectory.json    camera centres (SfM, georeferenced) + GNSS positions, WGS84
  meta.json          origin, region, ground level, overlay bounds, accuracy, provenance
Outputs (08_Exports/):
  gs_orthophoto_wgs84.png + .pgw + .prj   same orthophoto for QGIS (EPSG:4326 world file)
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, HERE)
from geosplat import geodesy as g  # noqa: E402
from geosplat import gaussians as gs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--gsd", type=float, default=0.15, help="orthophoto ground sampling distance (m)")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--max-scale-m", type=float, default=15.0, help="drop huge background blobs")
    args = ap.parse_args()
    t0 = time.time()
    O = os.path.abspath(args.project)
    X = os.path.join(O, "08_Exports")
    W = os.path.join(X, "web")
    os.makedirs(W, exist_ok=True)
    crs = json.load(open(os.path.join(X, "gaussians_crs.json")))
    tf = json.load(open(os.path.join(O, "06_Georeferencing", "transform.json")))
    o = tf["enu_origin"]
    lat0, lon0, h0 = o["lat_deg"], o["lon_deg"], o["h_m"]
    reg = tf["enu_to_utm_similarity"]["valid_region"]
    cen = np.array(reg["centre_enu_en"])
    R = float(reg["radius_m"])

    data, _ = gs.read_gs_ply(os.path.join(X, "gaussians_enu.ply"))
    n_all = len(data)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
    sc = np.exp(np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32))
    q = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rgb = np.clip(0.5 + gs.C0 * np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]), 0, 1)
    del data

    d = np.hypot(xyz[:, 0] - cen[0], xyz[:, 1] - cen[1])
    keep = (d <= R) & (op >= args.min_opacity) & (sc.max(1) <= args.max_scale_m)
    print(f"Gaussians: {n_all:,} total -> {keep.sum():,} in core r<={R:.0f} m, opacity>={args.min_opacity}, "
          f"scale<={args.max_scale_m} m")

    # ground level (for the flat Google base map): dense lowest layer of opaque splats near the centre
    near = keep & (d <= 0.8 * R) & (op > 0.5)
    low = xyz[near, 2] <= np.percentile(xyz[near, 2], 40)
    hl, el = np.histogram(xyz[near, 2][low], bins=np.arange(np.floor(xyz[near, 2][low].min()), np.ceil(xyz[near, 2][low].max()) + 0.5, 0.5))
    ground_u = float(el[np.argmax(hl)] + 0.25)
    print(f"Ground level (mode of lowest 40% opaque splats): U = {ground_u:.2f} m  -> h = {h0 + ground_u:.2f} m")

    # ---------------- .splat (importance-sorted)
    k = np.where(keep)[0]
    imp = (sc[k].prod(1) ** (1 / 3)) * op[k]
    k = k[np.argsort(-imp)]
    rec = np.zeros(len(k), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = xyz[k]
    rec["s"] = sc[k]
    rec["c"][:, :3] = (rgb[k] * 255 + 0.5).astype(np.uint8)
    rec["c"][:, 3] = (op[k] * 255 + 0.5).astype(np.uint8)
    rec["r"] = np.clip(q[k] * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(os.path.join(W, "scene.splat"))
    print(f"scene.splat: {len(k):,} splats, {os.path.getsize(os.path.join(W, 'scene.splat')) / 2**20:.1f} MB")

    # ---------------- orthophoto (top-down, z-buffer of opaque splats)
    gsd = args.gsd
    e0, n1 = cen[0] - R, cen[1] + R              # upper-left corner (west, north)
    npx = int(np.ceil(2 * R / gsd))
    # Each Gaussian covers pixel offset d with effective alpha  a = opacity * exp(-0.5 d^T Sxy^-1 d),
    # Sxy = xy block of Sigma = R S^2 R^T (orthographic top-down projection of the 3D covariance, px^2).
    # A pixel is painted by the HIGHEST Gaussian whose effective alpha >= A_MIN (z-buffer, 2 passes).
    A_MIN, RMAX = 0.25, 30
    m = keep & (op >= A_MIN)
    P = xyz[m]
    Rm = gs.quat_to_rotmat(q[m].astype(np.float64))
    S2 = (sc[m].astype(np.float64) ** 2)
    cov = np.einsum("nij,nj,nkj->nik", Rm, S2, Rm)[:, :2, :2] / gsd ** 2
    cov[:, 0, 0] += 0.09                          # +0.3 px low-pass, avoids sub-pixel aliasing
    cov[:, 1, 1] += 0.09
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det        # inverse covariance (px^-2), metric axes (x E, y N)
    lam = 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)
    opm = op[m].astype(np.float64)
    rpx = np.clip(np.sqrt(np.maximum(2 * lam * np.log(np.maximum(opm / A_MIN, 1.0)), 0)), 0.5, RMAX)
    del Rm, S2, cov
    col = (rgb[m] * 255 + 0.5).astype(np.uint8)
    px = np.floor((P[:, 0] - e0) / gsd).astype(np.int64)
    py = np.floor((n1 - P[:, 1]) / gsd).astype(np.int64)
    fx = (P[:, 0] - e0) / gsd - px - 0.5            # sub-pixel centre offset (image axes)
    fy = (n1 - P[:, 1]) / gsd - py - 0.5
    z = P[:, 2].astype(np.float32)
    zbuf = np.full(npx * npx, -1e9, np.float32)
    img = np.zeros((npx * npx, 3), np.uint8)
    rint = int(np.ceil(RMAX))
    offs = [(dx, dy, np.hypot(dx, dy)) for dy in range(-rint, rint + 1) for dx in range(-rint, rint + 1)
            if np.hypot(dx, dy) <= RMAX + 0.75]
    offs.sort(key=lambda t: t[2])
    order = np.argsort(-rpx)                         # splats sorted by radius: subset = prefix
    rs = rpx[order]
    for pass_ in (0, 1):
        for dx, dy, r in offs:
            cnt = int(np.searchsorted(-rs, -(r - 0.75), side="right"))
            if cnt == 0:
                continue
            s = order[:cnt]
            ex, ey = dx - fx[s], -(dy - fy[s])      # image offset -> metric offset (y North positive)
            qd = ia[s] * ex * ex + 2 * ib[s] * ex * ey + ic[s] * ey * ey
            s = s[opm[s] * np.exp(-0.5 * qd) >= A_MIN]
            xx, yy = px[s] + dx, py[s] + dy
            ok = (xx >= 0) & (xx < npx) & (yy >= 0) & (yy < npx)
            s, idx = s[ok], (yy[ok] * npx + xx[ok])
            if pass_ == 0:
                np.maximum.at(zbuf, idx, z[s])
            else:
                win = z[s] >= zbuf[idx]
                img[idx[win]] = col[s[win]]
    alpha = (zbuf > -1e8).reshape(npx, npx)
    img = img.reshape(npx, npx, 3)
    import cv2
    holes = (~alpha).astype(np.uint8)
    small = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # keep only big holes
    fill = (holes == 1) & (small == 0)
    if fill.any():
        img = cv2.inpaint(img, fill.astype(np.uint8), 2, cv2.INPAINT_TELEA)
        alpha = alpha | fill
    # circular core mask
    yy, xx = np.mgrid[0:npx, 0:npx]
    circ = np.hypot((xx + 0.5) * gsd - R, (yy + 0.5) * gsd - R) <= R
    alpha &= circ
    rgba = np.dstack([img, (alpha * 255).astype(np.uint8)])
    print(f"orthophoto {npx}x{npx} px @ {gsd} m, coverage {alpha.sum() / circ.sum() * 100:.1f} % of core disc")

    # bounds: ENU square -> WGS84. GroundOverlay maps the image linearly between lat/lon bounds; check the
    # error of that linear assumption at the four corners.
    lat_n = g.enu_to_geodetic(cen[0], cen[1] + R, ground_u, lat0, lon0, h0)[0][0]
    lat_s = g.enu_to_geodetic(cen[0], cen[1] - R, ground_u, lat0, lon0, h0)[0][0]
    lon_e = g.enu_to_geodetic(cen[0] + R, cen[1], ground_u, lat0, lon0, h0)[1][0]
    lon_w = g.enu_to_geodetic(cen[0] - R, cen[1], ground_u, lat0, lon0, h0)[1][0]
    ce, cn = np.array([-R, R, R, -R]) + cen[0], np.array([R, R, -R, -R]) + cen[1]
    la, lo, _ = g.enu_to_geodetic(ce, cn, np.full(4, ground_u), lat0, lon0, h0)
    lin_la = np.where(cn > cen[1], lat_n, lat_s)
    lin_lo = np.where(ce > cen[0], lon_e, lon_w)
    corner_err = float(np.max(np.hypot((la - lin_la) * 110574.0, (lo - lin_lo) * 111320.0 * np.cos(np.radians(lat0)))))
    print(f"overlay bounds N {lat_n:.8f} S {lat_s:.8f} E {lon_e:.8f} W {lon_w:.8f}; linear-bounds corner error "
          f"{corner_err * 100:.2f} cm")

    from PIL import Image
    im = Image.fromarray(rgba, "RGBA")
    im.save(os.path.join(W, "ortho.webp"), "WEBP", quality=88, method=4)
    im.save(os.path.join(X, "gs_orthophoto_wgs84.png"))
    dlat = (lat_n - lat_s) / npx
    dlon = (lon_e - lon_w) / npx
    with open(os.path.join(X, "gs_orthophoto_wgs84.pgw"), "w") as f:
        f.write(f"{dlon:.12e}\n0\n0\n{-dlat:.12e}\n{lon_w + dlon / 2:.12f}\n{lat_n - dlat / 2:.12f}\n")
    with open(os.path.join(X, "gs_orthophoto_wgs84.prj"), "w") as f:
        f.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

    # ---------------- trajectory
    import pandas as pd
    cams = pd.read_csv(os.path.join(O, "06_Georeferencing", "georeferenced_cameras.csv")).sort_values("pts_s")
    gl = cams.dropna(subset=["gnss_e"])
    glat, glon, gh = g.enu_to_geodetic(gl["gnss_e"].to_numpy(), gl["gnss_n"].to_numpy(), gl["gnss_u"].to_numpy(),
                                       lat0, lon0, h0)
    traj = {"sfm": [{"image": r.image, "t": round(r.pts_s, 3), "lat": r.lat_deg, "lon": r.lon_deg, "h": round(r.h_m, 3),
                     "e": round(r.e_enu, 3), "n": round(r.n_enu, 3), "u": round(r.u_enu, 3),
                     "yaw": round(r.cam_yaw_deg, 2), "pitch": round(r.cam_pitch_deg, 2),
                     "res": None if np.isnan(r.res_3d_m) else round(r.res_3d_m, 3)} for r in cams.itertuples()],
            "gnss": [{"lat": a, "lon": b, "h": round(c, 3)} for a, b, c in zip(glat, glon, gh)]}
    json.dump(traj, open(os.path.join(W, "trajectory.json"), "w"))

    meta = {
        "title": "GeoSplat - " + json.load(open(os.path.join(O, "logs", "_state.json"))).get("title", "project"),
        "origin": {"lat": lat0, "lon": lon0, "h": h0},
        "core": {"centre_e": float(cen[0]), "centre_n": float(cen[1]), "radius_m": R,
                 "centre_lat": float(g.enu_to_geodetic(cen[0], cen[1], 0, lat0, lon0, h0)[0][0]),
                 "centre_lon": float(g.enu_to_geodetic(cen[0], cen[1], 0, lat0, lon0, h0)[1][0])},
        "ground_u": ground_u, "ground_h": h0 + ground_u,
        "overlay": {"north": lat_n, "south": lat_s, "east": lon_e, "west": lon_w, "gsd_m": gsd, "px": npx,
                    "linear_bounds_corner_error_m": corner_err},
        "splats": {"file": "scene.splat", "count": int(len(k)), "of_total": n_all,
                   "filters": {"radius_m": R, "min_opacity": args.min_opacity, "max_scale_m": args.max_scale_m},
                   "format": "32 B: position f32x3 (ENU m), scale f32x3 (m), RGBA u8, quaternion u8x4 (w,x,y,z)*128+128"},
        "frame": "local ENU (x East, y North, z Up) metres; Google WebGLOverlayView fromLatLngAltitude uses the same "
                 "axes (x east, y north, z up, metres)",
        "accuracy": {"sfm_vs_gnss_rmse_3d_m": tf["residuals"]["rmse_3d_m"],
                     "checkpoint_rmse_3d_m": tf["residuals"]["checkpoint_rmse_3d_m"],
                     "note": tf["conventions"]["accuracy_note"]},
        "height_note": "Google base map is flat (ground = altitude 0). The 3D view places the estimated ground "
                       "level of the model at altitude 0; horizontal georeferencing is unchanged.",
        "metrics": crs.get("image_metrics", {}),
        "epsg_utm": int(tf["conventions"]["utm"].split("EPSG:")[1].rstrip(")")),
    }
    json.dump(meta, open(os.path.join(W, "meta.json"), "w"), indent=2)
    # viewer page + local key file (key comes from the app settings via environment, never from the repo)
    shutil.copy2(os.path.join(HERE, "web", "viewer.html"), os.path.join(W, "index.html"))
    key = os.environ.get("GEOSPLAT_GMAPS_KEY", "").strip()
    with open(os.path.join(W, "config.local.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by GeoSplat on this computer - do not share or commit.\n")
        f.write("window.GEOSPLAT_CONFIG = " + json.dumps({"googleMapsKey": key}) + ";\n")
    print(f"done in {time.time() - t0:.1f} s -> {W}")


if __name__ == "__main__":
    main()
