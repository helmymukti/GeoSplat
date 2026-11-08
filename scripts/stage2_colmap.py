#!/usr/bin/env python3
"""
GeoSplat - Phase 2: COLMAP sparse reconstruction from the Phase-1 frames.

Pipeline
  1. GPS-guided pair selection   -> 04_COLMAP/pairs.txt
  2. feature_extractor (GPU SIFT) -> 04_COLMAP/database.db
  3. matches_importer  (GPU)      -> geometric verification of the selected pairs
  4. mapper (incremental SfM + bundle adjustment) -> 05_Sparse_Reconstruction/sparse/<k>
  5. model_converter (TXT) + model_analyzer statistics
  6. Export camera poses + preliminary Sim3 check against GNSS (full georeferencing = Phase 3)

    python scripts/stage2_colmap.py --project <project folder>   [--dry-run]
Each COLMAP stage writes a marker file; re-running resumes. --redo <stage> restarts from a stage.
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
from geosplat import reports  # noqa: E402
from geosplat import sim3  # noqa: E402

STAGES = ["features", "matches", "mapper", "export"]


# ============================================================================ pairs
def view_geometry(ft):
    """Unit viewing direction (ENU) and approximate ground look-at point per frame."""
    yaw = np.radians(ft["gimbal_yaw_deg"].to_numpy(float))
    pit = np.radians(ft["gimbal_pitch_deg"].to_numpy(float))
    d = np.column_stack([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), np.sin(pit)])
    C = ft[["enu_e_m", "enu_n_m", "enu_u_m"]].to_numpy(float)
    h = ft["rel_alt_m"].to_numpy(float)                     # height above take-off level
    rng = np.where(pit < np.radians(-5), h / np.maximum(-np.sin(pit), 1e-3), 300.0)
    rng = np.minimum(rng, 600.0)
    look = C + d * rng[:, None]
    return d, C, look


def gps_guided_pairs(ft, seq_window=8, max_yaw_diff=60.0, max_lookat_dist=80.0):
    """Pair (i, j) if temporally close, OR if both cameras look at nearby ground points
    with a similar azimuth (captures loop closure on orbits / repeated passes)."""
    n = len(ft)
    d, C, look = view_geometry(ft)
    ok = np.all(np.isfinite(look), axis=1)
    yaw = ft["gimbal_yaw_deg"].to_numpy(float)
    pairs, reason = set(), {}
    for i in range(n):
        for j in range(i + 1, n):
            if j - i <= seq_window:
                pairs.add((i, j))
                reason[(i, j)] = "sequential"
                continue
            if not (ok[i] and ok[j]):
                continue
            dy = abs((yaw[i] - yaw[j] + 180) % 360 - 180)
            if dy <= max_yaw_diff and np.linalg.norm(look[i] - look[j]) <= max_lookat_dist:
                pairs.add((i, j))
                reason[(i, j)] = "spatial"
    pairs = sorted(pairs)
    return pairs, reason, look


# ============================================================================ main
def setup(args):
    out = os.path.abspath(args.project)
    if not os.path.exists(os.path.join(out, "logs", "_state.json")):
        sys.exit("Run stage 1 first: " + out + " has no logs/_state.json")
    if args.frames_dir is None:
        st = json.load(open(os.path.join(out, "logs", "_state.json")))
        args.frames_dir = f"fps_{st.get('extract_fps', 2.0):g}"
    P = {"out": out,
         "frames": os.path.join(out, "03_Extracted_Frames", args.frames_dir),
         "colmap": os.path.join(out, "04_COLMAP"),
         "sparse": os.path.join(out, "05_Sparse_Reconstruction"),
         "reports": os.path.join(out, "09_Reports"), "logs": os.path.join(out, "logs")}
    for k in ("colmap", "sparse", "reports", "logs"):
        os.makedirs(P[k], exist_ok=True)
    log = logging.getLogger("geosplat2")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(os.path.join(P["logs"], "phase2.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(sh)
    return P, log


def marker(P, stage):
    return os.path.join(P["colmap"], f"_done_{stage}.json")


def done(P, stage):
    return os.path.exists(marker(P, stage))


def mark(P, stage, info):
    json.dump(info, open(marker(P, stage), "w"), indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project folder created by stage 1")
    ap.add_argument("--frames-dir", default=None, help="default: fps_<extract fps> from stage 1")
    ap.add_argument("--camera-model", default="OPENCV")
    ap.add_argument("--focal-prior-px", type=float, default=None,
                    help="initial focal (px). Default: from 24 mm 35mm-equivalent over the image diagonal")
    ap.add_argument("--max-image-size", type=int, default=3840)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--seq-window", type=int, default=8)
    ap.add_argument("--max-yaw-diff", type=float, default=60.0)
    ap.add_argument("--max-lookat-dist", type=float, default=80.0)
    ap.add_argument("--gpu-index", default="0")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--gnss-inlier-m", type=float, default=3.0)
    ap.add_argument("--redo", choices=STAGES, default=None, help="restart from this stage")
    ap.add_argument("--dry-run", action="store_true", help="prepare pairs/commands only, do not call COLMAP")
    ap.add_argument("--colmap", default=None, help="path to colmap.bat / colmap.exe")
    args = ap.parse_args()
    P, log = setup(args)
    t_all = time.time()
    log.info("=== GeoSplat Phase 2 | frames: %s", P["frames"])

    man = pd.read_csv(os.path.join(P["frames"], "frames_manifest.csv"))
    ft = pd.read_csv(os.path.join(P["frames"], "frame_telemetry.csv"))
    ft = ft.set_index("image").loc[man["image"]].reset_index()
    log.info("Images: %d", len(ft))

    # ---------------- redo handling (never deletes: old outputs are renamed)
    if args.redo:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for s in STAGES[STAGES.index(args.redo):]:
            if done(P, s):
                os.replace(marker(P, s), marker(P, s) + f".old_{stamp}")
        if args.redo == "features":
            db = os.path.join(P["colmap"], "database.db")
            if os.path.exists(db):
                os.replace(db, db + f".old_{stamp}")
        if args.redo in ("features", "matches", "mapper"):
            sp = os.path.join(P["sparse"], "sparse")
            if os.path.isdir(sp):
                os.replace(sp, sp + f"_old_{stamp}")
        log.info("Redo from stage '%s' (previous outputs renamed with suffix %s)", args.redo, stamp)

    # ---------------- 1. pairs + image list
    img_list = os.path.join(P["colmap"], "image_list.txt")
    with open(img_list, "w") as f:
        f.write("\n".join(man["image"]) + "\n")
    pairs, reason, look = gps_guided_pairs(ft, args.seq_window, args.max_yaw_diff, args.max_lookat_dist)
    pairs_path = os.path.join(P["colmap"], "pairs.txt")
    with open(pairs_path, "w") as f:
        for i, j in pairs:
            f.write(f"{ft['image'][i]} {ft['image'][j]}\n")
    n_seq = sum(1 for p in pairs if reason[p] == "sequential")
    exhaustive = len(ft) * (len(ft) - 1) // 2
    log.info("Pairs: %d (%d sequential + %d GPS/view-guided) = %.1f %% of exhaustive %d",
             len(pairs), n_seq, len(pairs) - n_seq, 100 * len(pairs) / exhaustive, exhaustive)
    deg = np.bincount(np.array(pairs).ravel(), minlength=len(ft))
    log.info("Pairs per image: min %d / median %d / max %d", deg.min(), np.median(deg), deg.max())
    pd.DataFrame({"image": ft["image"], "lookat_e": look[:, 0], "lookat_n": look[:, 1],
                  "n_pairs": deg}).to_csv(os.path.join(P["colmap"], "pair_graph_nodes.csv"), index=False)

    st = json.load(open(os.path.join(P["out"], "logs", "_state.json")))
    W, H = int(st["width"]), int(st["height"])
    if args.focal_prior_px is not None:
        f0 = args.focal_prior_px
    elif st.get("focal35_mm"):
        f0 = st["focal35_mm"] / np.hypot(36.0, 24.0) * np.hypot(W, H)   # 35mm-equivalent over the diagonal
    else:
        f0 = 1.2 * max(W, H)                                            # COLMAP's default prior
    cam_params = {"OPENCV": f"{f0:.1f},{f0:.1f},{W / 2},{H / 2},0,0,0,0",
                  "SIMPLE_RADIAL": f"{f0:.1f},{W / 2},{H / 2},0",
                  "RADIAL": f"{f0:.1f},{W / 2},{H / 2},0,0",
                  "PINHOLE": f"{f0:.1f},{f0:.1f},{W / 2},{H / 2}"}[args.camera_model]
    log.info("Camera: %s, single shared camera, initial params %s (refined by bundle adjustment)",
             args.camera_model, cam_params)

    if args.dry_run:
        log.info("Dry run finished - COLMAP not called.")
        return

    # ---------------- COLMAP
    from geosplat.colmap_io import Colmap, read_cameras_txt, read_images_txt, read_points3d_txt
    cm = Colmap(args.colmap, log)
    log.info("COLMAP: %s | %s | CUDA: %s", " ".join(cm.cmd), cm.version["version"], cm.version["cuda"])
    gpu = "0" if args.no_gpu else "1"
    db = os.path.join(P["colmap"], "database.db")
    L = lambda s: os.path.join(P["colmap"], f"colmap_{s}.log")  # noqa: E731

    if not done(P, "features"):
        if os.path.exists(db):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            os.replace(db, db + f".incomplete_{stamp}")
        log.info("[2/6] Feature extraction (GPU SIFT, max %d px, %d features/img)...",
                 args.max_image_size, args.max_features)
        a = ["--database_path", db, "--image_path", P["frames"], "--image_list_path", img_list]
        a += cm.args("feature_extractor", [
            ("--ImageReader.single_camera", 1), ("--ImageReader.camera_model", args.camera_model),
            ("--ImageReader.camera_params", cam_params),
            (("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"), gpu),
            (("--FeatureExtraction.gpu_index", "--SiftExtraction.gpu_index"), args.gpu_index),
            (("--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"), args.max_image_size),
            (("--SiftExtraction.max_num_features", "--FeatureExtraction.max_num_features"), args.max_features),
            (("--FeatureExtraction.num_threads", "--SiftExtraction.num_threads"), -1)])
        dt = cm.run("feature_extractor", a, L("features"))
        log.info("      done in %.0f s", dt)
        mark(P, "features", {"seconds": dt, "args": a})

    if not done(P, "matches"):
        log.info("[3/6] Matching %d selected pairs (GPU) + geometric verification...", len(pairs))
        a = ["--database_path", db, "--match_list_path", pairs_path, "--match_type", "pairs"]
        a += cm.args("matches_importer", [
            (("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"), gpu),
            (("--FeatureMatching.gpu_index", "--SiftMatching.gpu_index"), args.gpu_index),
            (("--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"), 1),
            (("--FeatureMatching.max_num_matches", "--SiftMatching.max_num_matches"), 32768)])
        dt = cm.run("matches_importer", a, L("matches"))
        log.info("      done in %.0f s", dt)
        mark(P, "matches", {"seconds": dt, "pairs": len(pairs)})

    sparse_root = os.path.join(P["sparse"], "sparse")
    if not done(P, "mapper"):
        os.makedirs(sparse_root, exist_ok=True)
        log.info("[4/6] Incremental SfM + bundle adjustment (mapper)...")
        a = ["--database_path", db, "--image_path", P["frames"], "--output_path", sparse_root]
        a += cm.args("mapper", [(("--Mapper.num_threads",), -1)])
        dt = cm.run("mapper", a, L("mapper"))
        log.info("      done in %.0f s", dt)
        mark(P, "mapper", {"seconds": dt})

    # ---------------- 5. convert / analyse every model, keep the largest
    models = sorted(d for d in os.listdir(sparse_root) if os.path.isdir(os.path.join(sparse_root, d))
                    and os.path.exists(os.path.join(sparse_root, d, "images.bin")))
    if not models:
        sys.exit("Mapper produced no model - see " + L("mapper"))
    log.info("[5/6] Models found: %s", models)
    txt_root = os.path.join(P["sparse"], "sparse_txt")
    os.makedirs(txt_root, exist_ok=True)
    summ = []
    for m in models:
        td = os.path.join(txt_root, m)
        os.makedirs(td, exist_ok=True)
        cm.run("model_converter", ["--input_path", os.path.join(sparse_root, m), "--output_path", td,
                                   "--output_type", "TXT"], L(f"convert_{m}"))
        imgs = read_images_txt(os.path.join(td, "images.txt"))
        summ.append((m, len(imgs)))
    summ.sort(key=lambda x: -x[1])
    best = summ[0][0]
    log.info("      registered images per model: %s -> using model %s", dict(summ), best)
    cm.run("model_analyzer", ["--path", os.path.join(sparse_root, best)], L("analyzer"))
    stats = parse_analyzer(open(L("analyzer"), encoding="utf-8", errors="replace").read())

    td = os.path.join(txt_root, best)
    cams = read_cameras_txt(os.path.join(td, "cameras.txt"))
    imgs = read_images_txt(os.path.join(td, "images.txt"))
    xyz, rgb, err, tl = read_points3d_txt(os.path.join(td, "points3D.txt"))

    # ---------------- 6. poses + preliminary GNSS check
    rows = []
    for iid, im in imgs.items():
        rows.append({"image": im["name"], "image_id": iid, "camera_id": im["camera_id"],
                     "qw": im["qvec"][0], "qx": im["qvec"][1], "qy": im["qvec"][2], "qz": im["qvec"][3],
                     "tx": im["tvec"][0], "ty": im["tvec"][1], "tz": im["tvec"][2],
                     "cx_local": im["center"][0], "cy_local": im["center"][1], "cz_local": im["center"][2],
                     "n_obs": im["n_obs"], "n_triangulated": im["n_tri"]})
    poses = pd.DataFrame(rows).merge(ft[["image", "frame_index", "pts_s", "enu_e_m", "enu_n_m", "enu_u_m",
                                         "lat_deg", "lon_deg", "abs_alt_m", "sync_quality"]],
                                     on="image", how="left").sort_values("frame_index")
    unreg = sorted(set(ft["image"]) - set(poses["image"]))
    ok = poses[["enu_e_m", "enu_n_m", "enu_u_m"]].notna().all(axis=1).to_numpy()
    src = poses.loc[ok, ["cx_local", "cy_local", "cz_local"]].to_numpy()
    dst = poses.loc[ok, ["enu_e_m", "enu_n_m", "enu_u_m"]].to_numpy()
    fit = sim3.ransac_sim3(src, dst, thresh_m=args.gnss_inlier_m)
    al = sim3.apply_sim3(fit["scale"], fit["R"], fit["t"], poses[["cx_local", "cy_local", "cz_local"]].to_numpy())
    poses["e_aligned"], poses["n_aligned"], poses["u_aligned"] = al.T
    poses["gnss_residual_m"] = np.linalg.norm(al - poses[["enu_e_m", "enu_n_m", "enu_u_m"]].to_numpy(), axis=1)
    poses["gnss_inlier"] = False
    poses.loc[poses.index[ok], "gnss_inlier"] = fit["inliers"]
    res_h = np.hypot(al[:, 0] - poses["enu_e_m"], al[:, 1] - poses["enu_n_m"])
    res_v = al[:, 2] - poses["enu_u_m"]
    poses.to_csv(os.path.join(P["sparse"], "camera_poses.csv"), index=False, float_format="%.6f")
    for u in unreg:
        log.warning("Not registered: %s", u)

    # plots
    pts_al = sim3.apply_sim3(fit["scale"], fit["R"], fit["t"], xyz)
    plot_phase2(poses, pts_al, rgb, err, os.path.join(P["sparse"], "sparse_vs_gnss.png"),
                os.path.join(P["sparse"], "gnss_residuals.png"))
    for f in ("sparse_vs_gnss.png", "gnss_residuals.png"):
        shutil.copy2(os.path.join(P["sparse"], f), os.path.join(P["reports"], f))

    cam = next(iter(cams.values()))
    times = {s: json.load(open(marker(P, s))).get("seconds") for s in ("features", "matches", "mapper") if done(P, s)}
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(fit["R"]) - 1) / 2, -1, 1))))
    blocks = [
        ("COLMAP", [("binary", " ".join(cm.cmd)), ("version", cm.version["version"]),
                    ("CUDA build", cm.version["cuda"]), ("GPU used", not args.no_gpu),
                    ("stage times (s)", {k: round(v, 1) for k, v in times.items()})]),
        ("INPUT", [("frames", len(ft)), ("folder", P["frames"]),
                   ("pairs (selected / exhaustive)", f"{len(pairs)} / {exhaustive}"),
                   ("pair rule", f"|i-j|<={args.seq_window} OR (yaw diff <= {args.max_yaw_diff} deg AND "
                                 f"ground look-at distance <= {args.max_lookat_dist} m)"),
                   ("SIFT", f"max {args.max_image_size} px, {args.max_features} features/img")]),
        ("RECONSTRUCTION", [("models", dict(summ)), ("model used", best),
                            ("registered images", f"{len(imgs)} / {len(ft)}"),
                            ("unregistered", ", ".join(unreg) if unreg else "none"),
                            ("3D points", len(xyz)),
                            ("mean reprojection error (px)", round(float(err.mean()), 3)),
                            ("median reprojection error (px)", round(float(np.median(err)), 3)),
                            ("mean track length", round(float(tl.mean()), 2)),
                            ("model_analyzer", stats)]),
        ("SELF-CALIBRATED CAMERA", [("model", cam["model"]), ("size", f"{cam['width']} x {cam['height']}"),
                                    ("params", [round(v, 6) for v in cam["params"]]),
                                    ("initial focal (px)", round(f0, 1)),
                                    ("refined focal fx (px)", round(cam["params"][0], 1)),
                                    ("horizontal FOV (deg)", round(float(np.degrees(
                                        2 * np.arctan(cam["width"] / 2 / cam["params"][0]))), 2))]),
        ("PRELIMINARY GNSS CHECK (Sim3, RANSAC)", [
            ("scale (m per COLMAP unit)", round(fit["scale"], 6)),
            ("rotation angle (deg)", round(rot_deg, 3)),
            ("inliers", f"{fit['n_inliers']} / {fit['n']} (threshold {args.gnss_inlier_m} m)"),
            ("3D residual RMSE inliers (m)", round(fit["rmse_inliers"], 3)),
            ("3D residual median / max (m)", f"{fit['median']:.3f} / {fit['max']:.3f}"),
            ("horizontal RMSE (m)", round(float(np.sqrt(np.nanmean(res_h ** 2))), 3)),
            ("vertical RMSE (m)", round(float(np.sqrt(np.nanmean(res_v ** 2))), 3)),
            ("note", "Phase 2 sanity check only. Phase 3 adds time-offset estimation, weighting, "
                     "ECEF/UTM outputs and the formal residual report.")]),
    ]
    reports.write_text_report(os.path.join(P["reports"], "phase2_report.txt"), blocks)
    reports.write_html(os.path.join(P["reports"], "phase2_report.html"), "GeoSplat · Phase 2 sparse reconstruction",
                       blocks, [("Sparse model (Sim3-aligned) vs GNSS", os.path.join(P["sparse"], "sparse_vs_gnss.png")),
                                ("GNSS residuals per image", os.path.join(P["sparse"], "gnss_residuals.png"))])
    json.dump({"best_model": best, "registered": len(imgs), "images": len(ft), "points": int(len(xyz)),
               "mean_reproj_px": float(err.mean()), "sim3": {"scale": fit["scale"], "R": fit["R"].tolist(),
                                                             "t": fit["t"].tolist(), "rmse_inliers": fit["rmse_inliers"],
                                                             "n_inliers": fit["n_inliers"]}},
              open(os.path.join(P["sparse"], "phase2_summary.json"), "w"), indent=2)
    mark(P, "export", {"model": best})
    log.info("[6/6] Registered %d/%d images, %d points, mean reprojection %.3f px, GNSS Sim3 RMSE %.2f m",
             len(imgs), len(ft), len(xyz), err.mean(), fit["rmse_inliers"])
    log.info("=== Phase 2 finished in %.1f min. Report: %s", (time.time() - t_all) / 60,
             os.path.join(P["reports"], "phase2_report.html"))


def parse_analyzer(txt):
    import re
    keys = ["Cameras", "Images", "Registered images", "Points", "Observations", "Mean track length",
            "Mean observations per image", "Mean reprojection error"]
    out = {}
    for k in keys:
        m = re.search(re.escape(k) + r"\s*:\s*([0-9.]+)", txt)
        if m:
            out[k] = float(m.group(1))
    return out


def plot_phase2(poses, pts, rgb, err, out1, out2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2 = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    keep = np.isfinite(pts).all(1) & (err < 2.0)
    sub = np.where(keep)[0]
    if len(sub) > 60000:
        sub = np.random.default_rng(0).choice(sub, 60000, replace=False)
    ax.scatter(pts[sub, 0], pts[sub, 1], c=rgb[sub] / 255.0, s=0.3, lw=0)
    ax.plot(poses["enu_e_m"], poses["enu_n_m"], color=C1, lw=1.2, label="GNSS (interpolated)")
    ax.scatter(poses["e_aligned"], poses["n_aligned"], s=10, facecolor="none", edgecolor=C2, lw=0.8,
               label="COLMAP cameras (Sim3-aligned)")
    has_gnss = poses[["enu_e_m", "enu_n_m"]].notna().all(axis=1)
    out = ~poses["gnss_inlier"] & has_gnss
    if out.any():
        ax.scatter(poses["e_aligned"][out], poses["n_aligned"][out], s=40, marker="x", color="black",
                   label="GNSS outliers (rejected)")
    if (~has_gnss).any():
        ax.scatter(poses["e_aligned"][~has_gnss], poses["n_aligned"][~has_gnss], s=40, marker="D",
                   facecolor="none", edgecolor="black", label="no GNSS (outside telemetry span)")
    # frame the plot on the bulk of the scene, not on a few far-away background points
    ce, cn = poses["e_aligned"], poses["n_aligned"]
    pe, pn = pts[sub, 0], pts[sub, 1]
    lo_e, hi_e = min(np.percentile(pe, 1), ce.min()), max(np.percentile(pe, 99), ce.max())
    lo_n, hi_n = min(np.percentile(pn, 1), cn.min()), max(np.percentile(pn, 99), cn.max())
    pad = 0.05 * max(hi_e - lo_e, hi_n - lo_n)
    ax.set_xlim(lo_e - pad, hi_e + pad)
    ax.set_ylim(lo_n - pad, hi_n + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m, local ENU)")
    ax.set_ylabel("North (m, local ENU)")
    ax.set_title("Sparse point cloud + camera centres aligned to GNSS", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(out1)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), dpi=150, sharex=True)
    h = np.hypot(poses["e_aligned"] - poses["enu_e_m"], poses["n_aligned"] - poses["enu_n_m"])
    v = poses["u_aligned"] - poses["enu_u_m"]
    ax[0].plot(poses["pts_s"], h, color=C1, lw=1.2)
    ax[0].set_ylabel("horizontal (m)")
    ax[0].set_title("COLMAP − GNSS residual per image after Sim3", loc="left", fontsize=10)
    ax[1].plot(poses["pts_s"], v, color=C2, lw=1.2)
    ax[1].axhline(0, color="#9a9994", lw=0.6)
    ax[1].set_ylabel("vertical (m)")
    ax[1].set_xlabel("video time (s)")
    for a in ax:
        a.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(out2)
    plt.close(fig)


if __name__ == "__main__":
    main()
