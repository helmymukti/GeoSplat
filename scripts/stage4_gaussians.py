#!/usr/bin/env python3
"""
GeoSplat - Phase 4: Graphdeco 3D Gaussian Splatting on the georeferenced (metric ENU) COLMAP model,
then consistent export to local / ENU / UTM frames.

Stages (resumable, marker files in 07_Gaussian_Splatting):
  prepare : colmap image_undistorter (OPENCV -> PINHOLE, max 1920 px) on 06_Georeferencing/sparse_enu_txt
  train   : graphdeco train.py  (images kept on CPU: --data_device cpu; auto fallback on CUDA OOM)
  render  : graphdeco render.py + metrics.py on the held-out views (--eval: every 8th image)
  export  : transform Gaussians (positions, covariances, SH) ENU -> COLMAP-local and ENU -> UTM,
            CRS sidecar JSON, camera trajectory (CSV/GeoJSON/PLY), processing report

Run with the Gaussian-Splatting environment:  run_phase4.bat   (created by setup_gaussian_splatting.bat)
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, HERE)
from geosplat import gaussians as gs  # noqa: E402
from geosplat import geodesy as g  # noqa: E402

PROFILES = {
    # name: (train resolution arg, densify_grad_threshold, densify_until_iter)
    "balanced": ("1", 0.0003, 15000),     # 1920x1080 training images
    "low": ("2", 0.0004, 12000),          # 960x540, fewer Gaussians (fallback for 8 GB VRAM)
    "quality": ("1", 0.0002, 15000),      # Graphdeco defaults at 1920 px (needs more VRAM)
}


def setup_log(O):
    log = logging.getLogger("geosplat4")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(os.path.join(O, "logs", "phase4.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def stream(cmd, log_path, log, echo_pat=None, env=None, cwd=None):
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                             env=env, cwd=cwd, bufsize=1)
        last = 0.0
        tail = []
        for line in p.stdout:
            fh.write(line)
            tail = (tail + [line])[-60:]
            s = line.strip()
            if echo_pat and re.search(echo_pat, s) and time.time() - last > 15:
                log.info("   %s", s[-160:])
                last = time.time()
        p.wait()
    return p.returncode, time.time() - t0, "".join(tail)


def mark(D, stage, info):
    json.dump(info, open(os.path.join(D, f"_done_{stage}.json"), "w"), indent=2, default=str)


def done(D, stage):
    return os.path.exists(os.path.join(D, f"_done_{stage}.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project folder")
    ap.add_argument("--gs-env", default=os.path.join(HERE, "runtime", "gs_env.json"))
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--max-image-size", type=int, default=1920)
    ap.add_argument("--no-eval", action="store_true", help="train on all images (no held-out metrics)")
    ap.add_argument("--stages", default="prepare,train,render,export")
    ap.add_argument("--colmap", default=None)
    args = ap.parse_args()

    src = os.path.abspath(args.project)
    O = src
    D = os.path.join(O, "07_Gaussian_Splatting")
    X = os.path.join(O, "08_Exports")
    os.makedirs(D, exist_ok=True)
    os.makedirs(X, exist_ok=True)
    log = setup_log(O)
    if not os.path.exists(args.gs_env):
        sys.exit("Gaussian Splatting environment not installed (runtime/gs_env.json missing). Run the GPU setup first.")
    setup = json.load(open(args.gs_env))
    st = json.load(open(os.path.join(O, "logs", "_state.json")))
    frames = os.path.join(O, "03_Extracted_Frames", f"fps_{st.get('extract_fps', 2.0):g}")
    REPO, PY = setup["repo"], setup["python"]
    tf = json.load(open(os.path.join(O, "06_Georeferencing", "transform.json")))
    data_dir = os.path.join(D, "dataset")
    model_dir = os.path.join(D, "model_enu")
    stages = args.stages.split(",")
    log.info("=== GeoSplat Phase 4 | profile %s | %d iterations | Graphdeco %s", args.profile, args.iterations,
             setup.get("commit", "?")[:10])

    # ------------------------------------------------------------------ preflight (GPU stages only)
    if {"prepare", "train", "render"} & set(stages):
        import torch  # this script runs inside the GS environment
        free, total = torch.cuda.mem_get_info()
        du = shutil.disk_usage(D)
        n_img = len(glob.glob(os.path.join(frames, "*.jpg")))
        img_ram_gb = n_img * args.max_image_size ** 2 * (st["height"] / st["width"]) * 3 * 4 / 2**30
        log.info("GPU %s: VRAM free %.1f / %.1f GB | disk free %.0f GB | training images in RAM ~%.1f GB",
                 torch.cuda.get_device_name(0), free / 2**30, total / 2**30, du.free / 2**30, img_ram_gb)
        if total / 2**30 < 7.5:
            log.warning("Less than 8 GB VRAM: forcing profile 'low'")
            args.profile = "low"
        if du.free / 2**30 < 10:
            sys.exit("Less than 10 GB free disk - aborting (nothing deleted).")
        if free / 2**30 < 0.6 * total / 2**30:
            log.warning("Only %.1f GB VRAM free - close other GPU applications (browser, games) for best results.",
                        free / 2**30)

    # ------------------------------------------------------------------ prepare
    if "prepare" in stages and not done(D, "prepare"):
        from geosplat.colmap_io import Colmap
        cm = Colmap(args.colmap, log)
        if os.path.exists(data_dir):
            os.replace(data_dir, data_dir + time.strftime("_old_%Y%m%d_%H%M%S"))
        log.info("[prepare] undistorting %s -> PINHOLE (max %d px)", "sparse_enu_txt", args.max_image_size)
        cm.run("image_undistorter", ["--image_path", frames,
                                     "--input_path", os.path.join(O, "06_Georeferencing", "sparse_enu_txt"),
                                     "--output_path", data_dir, "--output_type", "COLMAP",
                                     "--max_image_size", str(args.max_image_size)],
               os.path.join(D, "colmap_undistort.log"))
        s0 = os.path.join(data_dir, "sparse", "0")
        os.makedirs(s0, exist_ok=True)
        for f in os.listdir(os.path.join(data_dir, "sparse")):
            fp = os.path.join(data_dir, "sparse", f)
            if os.path.isfile(fp):
                os.replace(fp, os.path.join(s0, f))
        n_img = len(glob.glob(os.path.join(data_dir, "images", "*")))
        log.info("[prepare] %d undistorted images, model in dataset/sparse/0 (metric ENU frame)", n_img)
        mark(D, "prepare", {"images": n_img})

    # ------------------------------------------------------------------ train
    if "train" in stages and not done(D, "train"):
        h = subprocess.run([PY, os.path.join(REPO, "train.py"), "-h"], capture_output=True, text=True,
                           cwd=REPO).stdout
        profile = args.profile
        for attempt in range(2):
            res, thr, until = PROFILES[profile]
            cmd = [PY, os.path.join(REPO, "train.py"), "-s", data_dir, "-m", model_dir, "-r", res,
                   "--data_device", "cpu", "--iterations", str(args.iterations),
                   "--densify_grad_threshold", str(thr), "--densify_until_iter", str(until),
                   "--test_iterations", "7000", str(args.iterations),
                   "--save_iterations", "7000", str(args.iterations),
                   "--checkpoint_iterations", "7000", "15000", "--quiet"]
            if not args.no_eval:
                cmd.append("--eval")
            if "--disable_viewer" in h:
                cmd.append("--disable_viewer")
            ck = sorted(glob.glob(os.path.join(model_dir, "chkpnt*.pth")),
                        key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
            if ck and attempt == 0:
                cmd += ["--start_checkpoint", ck[-1]]
                log.info("[train] resuming from %s", os.path.basename(ck[-1]))
            log.info("[train] profile %s: -r %s, densify thr %s until %d -> %s", profile, res, thr, until, model_dir)
            rc, dt, tail = stream(cmd, os.path.join(D, "train.log"), log, echo_pat=r"Training progress|ITER|Evaluat",
                                  cwd=REPO)
            if rc == 0:
                log.info("[train] finished in %.1f min", dt / 60)
                mark(D, "train", {"minutes": dt / 60, "profile": profile, "cmd": cmd})
                break
            if "out of memory" in tail.lower() and profile != "low":
                log.warning("[train] CUDA out of memory -> retrying with profile 'low' (new model folder)")
                if os.path.exists(model_dir):
                    os.replace(model_dir, model_dir + time.strftime("_oom_%Y%m%d_%H%M%S"))
                profile = "low"
                continue
            sys.exit(f"[train] failed (exit {rc}); see {os.path.join(D, 'train.log')}")

    # ------------------------------------------------------------------ render + metrics
    if "render" in stages and not done(D, "render") and not args.no_eval:
        rc1, dt1, _ = stream([PY, os.path.join(REPO, "render.py"), "-m", model_dir, "--skip_train", "--quiet"],
                             os.path.join(D, "render.log"), log, cwd=REPO)
        rc2, dt2, _ = stream([PY, os.path.join(REPO, "metrics.py"), "-m", model_dir],
                             os.path.join(D, "metrics.log"), log, echo_pat=r"SSIM|PSNR|LPIPS", cwd=REPO)
        log.info("[render] render rc=%d (%.0fs), metrics rc=%d (%.0fs)", rc1, dt1, rc2, dt2)
        mark(D, "render", {"render_rc": rc1, "metrics_rc": rc2})

    # ------------------------------------------------------------------ export
    if "export" in stages:
        export(O, D, X, model_dir, tf, setup, log)


def export(O, D, X, model_dir, tf, setup, log):
    import pandas as pd
    its = sorted(glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    if not its:
        sys.exit("No trained point_cloud found")
    ply = os.path.join(its[-1], "point_cloud.ply")
    it = int(its[-1].rsplit("_", 1)[1])
    data, _ = gs.read_gs_ply(ply)
    deg = gs.sh_degree_of(data)
    log.info("[export] %s: %d Gaussians, SH degree %d", ply, len(data), deg)
    o = tf["enu_origin"]
    L = tf["local_to_enu"]
    s, R, t = L["scale"], np.array(L["R"]), np.array(L["t"])
    U = tf["enu_to_utm_similarity"]
    su, Ru, tu, off = U["scale"], np.array(U["R"]), np.array(U["t"]), np.array(U["offset_EN_h"])
    epsg = tf["conventions"]["utm"].split("EPSG:")[1].rstrip(")")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(float)
    common = [f"GeoSplat Phase 4 | Graphdeco 3DGS iteration {it}", "WGS84"]

    # 1) ENU (native training frame = georeferenced, metric)
    p_enu = os.path.join(X, "gaussians_enu.ply")
    gs.write_gs_ply(p_enu, data, common + [f"frame local ENU origin lat {o['lat_deg']:.9f} lon {o['lon_deg']:.9f} "
                                           f"h {o['h_m']:.3f} (see gaussians_crs.json)"])
    # 2) COLMAP local (inverse Sim3) - original reconstruction frame, backup
    inv_s, inv_R = 1.0 / s, R.T
    inv_t = -(R.T @ t) / s
    loc = gs.transform_gaussians(data, inv_s, inv_R, inv_t)
    p_loc = os.path.join(X, "gaussians_local_colmap.ply")
    gs.write_gs_ply(p_loc, loc, common + ["frame COLMAP local (arbitrary scale), inverse of local_to_enu"])
    # 3) UTM (similarity fitted in Phase 3, false origin subtracted for float32 precision)
    utm = gs.transform_gaussians(data, su, Ru, tu)
    p_utm = os.path.join(X, f"gaussians_utm_epsg{epsg}_offset.ply")
    gs.write_gs_ply(p_utm, utm, common + [f"frame EPSG:{epsg} minus offset E {off[0]:.1f} N {off[1]:.1f} h {off[2]:.1f}"])

    # verification of the written files
    chk = {}
    d2, _ = gs.read_gs_ply(p_loc)
    back = gs.transform_gaussians(d2, s, R, t)
    chk["local_roundtrip_max_pos_err_m"] = float(np.max(np.abs(np.column_stack([back["x"], back["y"], back["z"]]) - xyz)))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(data), min(2000, len(data)), replace=False)
    lat, lon, h = g.enu_to_geodetic(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], o["lat_deg"], o["lon_deg"], o["h_m"])
    zone = int(re.search(r"zone (\d+)", tf["conventions"]["utm"]).group(1))
    E, N, _, _ = g.latlon_to_utm(lat, lon, zone, "S" in tf["conventions"]["utm"].split("zone")[1][:5])
    exact = np.column_stack([E - off[0], N - off[1], h - off[2]])
    got = np.column_stack([utm["x"][sel], utm["y"][sel], utm["z"][sel]]).astype(float)
    cen = np.array(U["valid_region"]["centre_enu_en"])
    rr = np.hypot(xyz[sel, 0] - cen[0], xyz[sel, 1] - cen[1])
    inside = rr <= U["valid_region"]["radius_m"]
    err = np.linalg.norm(got - exact, axis=1)
    chk["utm_vs_exact_projection_max_err_m_in_region"] = float(err[inside].max()) if inside.any() else None
    chk["utm_vs_exact_projection_max_err_m_all"] = float(err.max())
    chk["fraction_gaussians_in_valid_region"] = float(inside.mean())
    log.info("[export] checks: %s", chk)

    # camera trajectory
    cams = pd.read_csv(os.path.join(O, "06_Georeferencing", "georeferenced_cameras.csv"))
    cams.to_csv(os.path.join(X, "camera_trajectory.csv"), index=False)
    feats = [{"type": "Feature", "properties": {"name": "camera trajectory", "crs": "EPSG:4326 lon/lat + h (m)"},
              "geometry": {"type": "LineString",
                           "coordinates": [[r.lon_deg, r.lat_deg, r.h_m] for r in cams.itertuples()]}}]
    for r in cams.itertuples():
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon_deg, r.lat_deg, r.h_m]},
                      "properties": {"image": r.image, "t_s": r.pts_s, "yaw_deg": r.cam_yaw_deg,
                                     "pitch_deg": r.cam_pitch_deg, "res_3d_m": r.res_3d_m}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(X, "camera_trajectory.geojson"), "w"))

    # metrics
    metrics = {}
    rj = os.path.join(model_dir, "results.json")
    if os.path.exists(rj):
        metrics = json.load(open(rj))
    sidecar = {
        "software": {"geosplat": "Phase 4", "gaussian_splatting": "graphdeco-inria/gaussian-splatting",
                     "commit": setup.get("commit"), "torch": setup.get("torch"), "iteration": it},
        "n_gaussians": int(len(data)), "sh_degree": deg,
        "files": {
            os.path.basename(p_enu): {"frame": "local ENU (East, North, Up), metres", "origin": o,
                                      "to_ecef": "X_ecef = M_enu2ecef @ X + X0 (see transform.json local_to_ecef)"},
            os.path.basename(p_loc): {"frame": "COLMAP local (arbitrary scale)",
                                      "to_enu": {"scale": s, "R": R.tolist(), "t": t.tolist()}},
            os.path.basename(p_utm): {"frame": f"EPSG:{epsg} with false origin removed",
                                      "add_offset_to_get_utm": off.tolist(),
                                      "from_enu": {"scale": su, "R": Ru.tolist(), "t": tu.tolist()},
                                      "valid_region": U["valid_region"],
                                      "max_similarity_misfit_in_region_m": U["max_misfit_m"]},
        },
        "gaussian_transform_rule": "mu' = sR mu + t; rotation q' = q_R * q; log-scale' = log-scale + ln(s); "
                                   "SH bands rotated so that c'(d') = c(R^T d'); opacity unchanged",
        "height_reference": tf["conventions"]["height_reference"],
        "accuracy": {"gnss_vs_sfm_rmse_3d_m": tf["residuals"]["rmse_3d_m"],
                     "checkpoint_rmse_3d_m": tf["residuals"]["checkpoint_rmse_3d_m"],
                     "note": tf["conventions"]["accuracy_note"]},
        "time_offset_s": tf["time_offset_s"],
        "verification": chk, "image_metrics": metrics,
    }
    json.dump(sidecar, open(os.path.join(X, "gaussians_crs.json"), "w"), indent=2)
    tr = json.load(open(os.path.join(D, "_done_train.json"))) if done(D, "train") else {}
    lines = [f"GeoSplat Phase 4 report  ({time.strftime('%Y-%m-%d %H:%M')})", "=" * 70,
             f"Gaussians                : {len(data):,}  (SH degree {deg}, iteration {it})",
             f"Training                 : {tr.get('minutes', float('nan')):.1f} min, profile {tr.get('profile')}",
             f"Held-out image metrics   : {json.dumps(metrics)}",
             f"Exports (08_Exports)     : {os.path.basename(p_enu)}, {os.path.basename(p_loc)}, {os.path.basename(p_utm)},",
             "                           gaussians_crs.json, camera_trajectory.csv/.geojson",
             f"Verification             : {json.dumps(chk)}",
             f"Georeferencing accuracy  : SfM-vs-GNSS RMSE {tf['residuals']['rmse_3d_m']:.3f} m (relative; "
             "absolute needs GCP/RTK)"]
    open(os.path.join(O, "09_Reports", "phase4_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log.info("[export] done -> %s", X)


if __name__ == "__main__":
    main()
