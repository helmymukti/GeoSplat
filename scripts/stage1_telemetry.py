#!/usr/bin/env python3
"""
GeoSplat - Stage 1: video inspection, telemetry decoding, frame extraction, synchronisation.

    python scripts/stage1_telemetry.py --video D:/flight/DJI_0001.MP4 [--srt D:/flight/DJI_0001.SRT]
                                       --project D:/flight/DJI_0001_GeoSplat --fps 2

Telemetry sources (used if present, both cross-checked):
  * DJI per-frame SRT subtitle file (auto-detected next to the video when --srt is omitted)
  * DJI 'djmd' protobuf data track embedded in the MP4 (decoded natively, see docs/METHODS.md)
At least one source with GNSS positions is required.
Steps: inspect | extract | sync | all.  The source video/SRT are opened read-only, never copied or modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, HERE)
from geosplat import geodesy as g  # noqa: E402
from geosplat import media, reports  # noqa: E402
from geosplat import telemetry_parsers as tp  # noqa: E402
from geosplat.sync import SyncConfig, TelemetrySync  # noqa: E402

SUBDIRS = ["01_Metadata", "02_Telemetry", "03_Extracted_Frames", "04_COLMAP",
           "05_Sparse_Reconstruction", "06_Georeferencing", "07_Gaussian_Splatting",
           "08_Exports", "09_Reports", "logs"]
SRT_COLUMNS = ["srt_index", "srt_start_s", "srt_end_s", "srt_datetime", "frame_cnt", "difftime_ms", "iso",
               "shutter_str", "shutter_s", "fnum", "ev", "focal_len", "dzoom_ratio", "latitude", "longitude",
               "rel_alt", "abs_alt", "gb_yaw", "gb_pitch", "gb_roll"]


def find_srt(video):
    stem = os.path.splitext(video)[0]
    for ext in (".SRT", ".srt", ".Srt"):
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


def setup(args):
    mp4 = os.path.abspath(args.video)
    if not os.path.isfile(mp4):
        sys.exit(f"Video not found: {mp4}")
    srt = os.path.abspath(args.srt) if args.srt else find_srt(mp4)
    if srt and not os.path.isfile(srt):
        sys.exit(f"SRT not found: {srt}")
    src = os.path.dirname(mp4)
    out = os.path.abspath(args.project) if args.project else os.path.splitext(mp4)[0] + "_GeoSplat"
    for d in SUBDIRS:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    log = logging.getLogger("geosplat")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(os.path.join(out, "logs", "phase1.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        log.addHandler(logging.StreamHandler(sys.stdout))
    P = {"src": src, "out": out, "mp4": mp4, "srt": srt}
    return P, log


def p(P, sub, name):
    return os.path.join(P["out"], sub, name)


# ============================================================================ INSPECT
def step_inspect(P, log, args):
    t0 = time.time()
    src_stat = {k: (os.path.getsize(P[k]), os.path.getmtime(P[k])) for k in ("mp4", "srt") if P[k]}
    log.info("Probing video with ffprobe (streaming, read-only)")
    probe = media.ffprobe_json(P["mp4"])
    vstreams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    v = max(vstreams, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
    data_streams = [s for s in probe["streams"] if s["codec_type"] == "data"]
    djmd_idx = next((s["index"] for s in data_streams if s.get("codec_tag_string") == "djmd"), None)
    fr = [int(x) for x in v["avg_frame_rate"].split("/")]
    fps = fr[0] / fr[1]
    vp = media.video_packet_table(P["mp4"], v["index"])
    kf = vp.index[vp["keyframe"]].to_numpy()
    pts = vp["pts_time"].to_numpy()
    dpts = np.diff(pts)

    # ExifTool (optional)
    exif = {"available": bool(shutil.which("exiftool"))}
    if exif["available"]:
        r = subprocess.run(["exiftool", "-j", "-G3", "-ee", "-n", P["mp4"]], capture_output=True, text=True)
        open(p(P, "01_Metadata", "exiftool_ee.json"), "w", encoding="utf-8").write(r.stdout)
        exif["output"] = "01_Metadata/exiftool_ee.json"
    else:
        exif["note"] = ("ExifTool not installed in this runtime; the embedded DJI 'djmd' protobuf track "
                        "was decoded directly by GeoSplat instead.")
    log.info("ExifTool available: %s", exif["available"])

    # Telemetry
    if P["srt"]:
        log.info("Parsing SRT %s", P["srt"])
        srt = tp.parse_srt(P["srt"])
        for c in SRT_COLUMNS:
            if c not in srt:
                srt[c] = np.nan
        log.info("SRT records: %d", len(srt))
    else:
        log.info("No SRT file - telemetry must come from the embedded track")
        srt = pd.DataFrame(columns=SRT_COLUMNS)
    djmd, hdr, fstats = (None, {}, {})
    if djmd_idx is not None:
        log.info("Decoding embedded djmd protobuf track (stream %d)", djmd_idx)
        djmd, hdr, fstats = tp.parse_djmd(P["mp4"], djmd_idx)
        proto = hdr.get("1-1-1", "?")
        ok_pos = djmd["djmd_lat_deg"].between(-90, 90) & djmd["djmd_lon_deg"].between(-180, 180) & \
            ~((djmd["djmd_lat_deg"] == 0) & (djmd["djmd_lon_deg"] == 0))
        if proto != "dvtm_wm265e.proto":
            log.warning("djmd proto %s: field map verified only for dvtm_wm265e (Mavic 3 Enterprise); "
                        "positions are used only if plausible and are cross-checked against the SRT", proto)
        if ok_pos.mean() < 0.5:
            log.warning("djmd positions not plausible for this proto -> ignoring embedded GNSS")
            for c in ("djmd_lat_deg", "djmd_lon_deg", "djmd_abs_alt_m", "djmd_rel_alt_m"):
                djmd[c] = np.nan
        log.info("djmd packets: %d (%s)", len(djmd), proto)

    # -------- master per-frame table
    n = len(vp)
    tel = pd.DataFrame({"frame_index": np.arange(n), "t": pts})
    if len(srt):
        s = srt.copy()
        if s["frame_cnt"].notna().all() and len(s) == n:
            s["frame_index"] = s["frame_cnt"].astype(int) - 1          # DJI per-frame SRT
        else:                                                           # assign each frame the cue covering its time
            k = np.clip(np.searchsorted(s["srt_start_s"].to_numpy(float), pts, side="right") - 1, 0, len(s) - 1)
            s = s.iloc[k].reset_index(drop=True)
            s["frame_index"] = np.arange(n)
            log.info("SRT is not per-frame (%d cues for %d frames) - matched by cue time", len(srt), n)
        tel = tel.merge(s, on="frame_index", how="left")
    else:
        for c in SRT_COLUMNS:
            tel[c] = np.nan
    if djmd is not None:
        k = np.clip(np.searchsorted(djmd["djmd_pts_s"].to_numpy(), pts), 0, len(djmd) - 1)
        km = np.clip(k - 1, 0, len(djmd) - 1)
        k = np.where(np.abs(djmd["djmd_pts_s"].to_numpy()[km] - pts) <
                     np.abs(djmd["djmd_pts_s"].to_numpy()[k] - pts), km, k)
        tel = pd.concat([tel, djmd.iloc[k].reset_index(drop=True)], axis=1)
        tel["djmd_minus_video_pts_s"] = tel["djmd_pts_s"] - tel["t"]
    tel["srt_minus_video_pts_s"] = tel["srt_start_s"] - tel["t"]

    # canonical channels: prefer djmd (full precision), fall back to SRT
    def pick(a, b):
        if a in tel and tel[a].notna().any():
            return tel[a]
        return tel[b] if b in tel else pd.Series(np.nan, index=tel.index)
    tel["lat"] = pick("djmd_lat_deg", "latitude")
    tel["lon"] = pick("djmd_lon_deg", "longitude")
    tel["abs_alt"] = pick("djmd_abs_alt_m", "abs_alt")
    tel["rel_alt"] = pick("djmd_rel_alt_m", "rel_alt")
    tel["gimbal_yaw"] = pick("djmd_gimbal_yaw_deg", "gb_yaw")
    tel["gimbal_pitch"] = pick("djmd_gimbal_pitch_deg", "gb_pitch")
    tel["gimbal_roll"] = pick("djmd_gimbal_roll_deg", "gb_roll")
    for c in ("drone_yaw", "drone_pitch", "drone_roll"):
        tel[c] = tel.get(c + "_deg", np.nan)
    tel["vel_n"] = tel.get("djmd_vel_north_ms", np.nan)
    tel["vel_e"] = tel.get("djmd_vel_east_ms", np.nan)
    tel["vel_vertical_raw"] = tel.get("djmd_vel_vertical_raw_ms", np.nan)
    tel["speed_h_ms"] = np.hypot(tel["vel_n"], tel["vel_e"])

    if tel["lat"].notna().sum() < 10:
        sys.exit("NO GNSS TELEMETRY: neither the SRT nor the embedded track contains positions for this video. "
                 "Provide the DJI .SRT file recorded with the video (enable 'Video Captions' on the drone).")
    if tel["rel_alt"].isna().all():
        tel["rel_alt"] = tel["abs_alt"] - tel["abs_alt"].dropna().iloc[0]
        log.warning("No relative altitude in telemetry - using abs_alt minus first value")
    if tel["abs_alt"].isna().all():
        tel["abs_alt"] = tel["rel_alt"]
        log.warning("No absolute altitude - using relative altitude as height (vertical datum unknown)")
    missing = tel[["lat", "lon", "abs_alt"]].isna().any(axis=1) | ((tel["lat"] == 0) & (tel["lon"] == 0))
    tel["telemetry_missing"] = missing
    E, N, zone, south = g.latlon_to_utm(tel["lat"], tel["lon"])
    tel["utm_e"], tel["utm_n"] = E, N
    epsg = g.utm_epsg(zone, south)

    # synchronisation on ALL frames (quality statistics + staircase error)
    sync = TelemetrySync(tel, SyncConfig(offset_s=args.offset))
    q = sync.query(pts)
    ref = sync.ref
    e_raw, n_raw, _ = g.geodetic_to_enu(tel["lat"], tel["lon"], tel["abs_alt"], *ref)
    tel["pos_is_update_epoch"] = sync.pos_epoch_mask
    tel["interp_lat"], tel["interp_lon"], tel["interp_abs_alt"] = q["lat_deg"], q["lon_deg"], q["abs_alt_m"]
    tel["gnss_speed_h_ms"] = q.get("gnss_speed_h_ms")
    tel["gnss_course_deg"] = q.get("gnss_course_deg")
    tel["staircase_err_m"] = np.hypot(q["enu_e_m"] - e_raw, q["enu_n_m"] - n_raw)
    tel["sync_quality"] = q["sync_quality"]
    if "latitude" in tel:
        es, ns, _ = g.geodetic_to_enu(tel["latitude"], tel["longitude"], tel["abs_alt"], *ref)
        tel["srt_vs_djmd_h_m"] = np.hypot(es - e_raw, ns - n_raw)

    # vertical velocity sign check
    vz_corr = float(np.corrcoef(np.nan_to_num(tel["vel_vertical_raw"]),
                                np.nan_to_num(q.get("gnss_speed_v_ms", np.zeros(n))))[0, 1])

    cols_first = ["frame_index", "t", "srt_datetime", "lat", "lon", "abs_alt", "rel_alt", "utm_e", "utm_n",
                  "interp_lat", "interp_lon", "interp_abs_alt", "pos_is_update_epoch", "sync_quality",
                  "drone_yaw", "drone_pitch", "drone_roll", "gimbal_yaw", "gimbal_pitch", "gimbal_roll",
                  "vel_n", "vel_e", "vel_vertical_raw", "speed_h_ms", "gnss_speed_h_ms", "gnss_course_deg",
                  "iso", "shutter_str", "fnum", "ev", "focal_len", "dzoom_ratio", "djmd_color_temp_k",
                  "telemetry_missing", "staircase_err_m", "srt_vs_djmd_h_m",
                  "srt_minus_video_pts_s", "djmd_minus_video_pts_s", "frame_cnt", "difftime_ms",
                  "djmd_clock_us", "djmd_sensor_frame_seq", "latitude", "longitude"]
    cols = [c for c in cols_first if c in tel] + [c for c in tel.columns if c not in cols_first]
    tel = tel[cols]
    tel.to_csv(p(P, "02_Telemetry", "telemetry.csv"), index=False, float_format="%.9f")
    tel.to_pickle(p(P, "02_Telemetry", "_telemetry.pkl"))
    log.info("telemetry.csv written (%d rows, %d cols)", len(tel), len(tel.columns))

    # -------- video_info.json
    tags = probe["format"].get("tags", {})
    gps_container = any(k.lower() in ("location", "com.apple.quicktime.location.iso6709", "©xyz")
                        for k in tags) or any(k.lower() in ("location",) for k in v.get("tags", {}))
    dur = float(probe["format"]["duration"])
    seq = tel.get("djmd_sensor_frame_seq")
    seq_skips = int((np.diff(seq.dropna()) > 1).sum()) if seq is not None else None
    info = {
        "source": {"video": P["mp4"], "srt": P["srt"], "video_bytes": src_stat["mp4"][0],
                   "srt_bytes": src_stat["srt"][0] if "srt" in src_stat else None},
        "container": {"format": probe["format"]["format_name"], "duration_s": dur,
                      "bit_rate_bps": int(probe["format"]["bit_rate"]), "creation_time_utc": tags.get("creation_time"),
                      "encoder": tags.get("encoder"), "streams": len(probe["streams"])},
        "video": {"stream_index": v["index"], "codec": v["codec_name"], "profile": v.get("profile"),
                  "width": int(v["width"]), "height": int(v["height"]), "pix_fmt": v.get("pix_fmt"),
                  "fps": fps, "fps_rational": v["avg_frame_rate"], "frames": n,
                  "constant_frame_rate": bool(np.allclose(dpts, dpts.mean(), atol=2e-4)),
                  "frame_interval_s": {"min": float(dpts.min()), "max": float(dpts.max())},
                  "keyframes": int(len(kf)), "gop_frames_median": float(np.median(np.diff(kf))) if len(kf) > 1 else None,
                  "color_transfer": v.get("color_transfer"), "color_primaries": v.get("color_primaries"),
                  "bit_rate_bps": int(v.get("bit_rate", 0))},
        "other_streams": [{"index": s_["index"], "type": s_["codec_type"], "codec": s_.get("codec_name"),
                           "tag": s_.get("codec_tag_string"), "handler": s_.get("tags", {}).get("handler_name"),
                           "width": s_.get("width"), "height": s_.get("height")}
                          for s_ in probe["streams"] if s_["index"] != v["index"]],
        "camera": {"model_container": tags.get("encoder"), "model_djmd": hdr.get("1-1-10"),
                   "camera_short_name": "DJI M3E (from djmd)", "serial_djmd": hdr.get("1-1-5"),
                   "proto": hdr.get("1-1-1"), "proto_versions": [hdr.get("1-1-2"), hdr.get("1-1-3")],
                   "focal_len_35mm_equiv_mm_srt": float(tel["focal_len"].median()) if "focal_len" in tel else None,
                   "f_number": float(tel["fnum"].median()) if "fnum" in tel else None,
                   "iso_range": [float(tel["iso"].min()), float(tel["iso"].max())] if "iso" in tel else None,
                   "shutter_range": [tel["shutter_str"].iloc[tel["shutter_s"].argmax()],
                                     tel["shutter_str"].iloc[tel["shutter_s"].argmin()]]
                   if ("shutter_s" in tel and tel["shutter_s"].notna().any()) else None,
                   "digital_zoom_range": [float(tel["dzoom_ratio"].min()), float(tel["dzoom_ratio"].max())]},
        "gps_availability": {
            "container_level_location_tag": bool(gps_container),
            "embedded_djmd_per_frame": djmd is not None and tel["djmd_lat_deg"].notna().any(),
            "srt_per_frame": bool(tel["latitude"].notna().any()) if "latitude" in tel else False,
            "verdict": ("GPS in BOTH the embedded DJI 'djmd' track (float64) and the SRT (rounded)"
                        if (djmd is not None and tel["djmd_lat_deg"].notna().any() and tel["latitude"].notna().any())
                        else "GPS from the embedded DJI 'djmd' track only" if (djmd is not None and tel["djmd_lat_deg"].notna().any())
                        else "GPS from the SRT only"),
            "gnss_update_rate_hz": round(1.0 / sync.nominal_dt, 2),
            "rtk": "No RTK fields found in SRT or djmd track; RTK use cannot be confirmed (treat as standard GNSS).",
        },
        "telemetry_time": {"srt_records": len(srt), "djmd_packets": 0 if djmd is None else len(djmd),
                           "srt_timestamp_format": "SRT cue HH:MM:SS,mmm --> HH:MM:SS,mmm + FrameCnt + DiffTime(ms) + "
                                                   "local datetime 'YYYY-MM-DD HH:MM:SS.mmm'",
                           "srt_datetime_first": str(srt["srt_datetime"].iloc[0]) if len(srt) else None,
                           "srt_datetime_last": str(srt["srt_datetime"].iloc[-1]) if len(srt) else None,
                           "utc_offset_inferred_h": round((pd.Timestamp(str(srt["srt_datetime"].iloc[0])) -
                                                           pd.Timestamp(tags.get("creation_time", "")[:19])
                                                           ).total_seconds() / 3600, 2)
                           if (tags.get("creation_time") and len(srt) and pd.notna(srt["srt_datetime"].iloc[0])) else None,
                           "djmd_sensor_frame_seq_skips": seq_skips},
    }
    json.dump(info, open(p(P, "01_Metadata", "video_info.json"), "w"), indent=2, default=str)
    json.dump(fstats, open(p(P, "01_Metadata", "djmd_field_inventory.json"), "w"), indent=2, default=str)
    hw = media.detect_hardware(P["out"])
    json.dump(hw, open(p(P, "01_Metadata", "hardware_runtime.json"), "w"), indent=2)

    # -------- plots
    utm_label = f"WGS84 / UTM zone {zone}{'S' if south else 'N'}, EPSG:{epsg}"
    reports.plot_trajectory(tel, None, utm_label, p(P, "02_Telemetry", "gps_trajectory.png"))
    reports.plot_altitude(tel, p(P, "02_Telemetry", "altitude_profile.png"))
    reports.plot_sync(tel, p(P, "02_Telemetry", "sync_quality.png"))

    f35 = float(tel["focal_len"].median()) if tel["focal_len"].notna().any() else None
    created = tags.get("creation_time", "")[:10]
    state = {"zone": zone, "south": south, "epsg": epsg, "fps": fps, "n_frames": n, "ref": ref,
             "nominal_dt": sync.nominal_dt, "vz_corr": vz_corr, "inspect_s": round(time.time() - t0, 1),
             "width": int(v["width"]), "height": int(v["height"]), "src_stat": src_stat,
             "video": P["mp4"], "srt": P["srt"], "focal35_mm": f35,
             "camera": hdr.get("1-1-10") or tags.get("encoder") or "unknown camera", "date": created,
             "title": f"{os.path.splitext(os.path.basename(P['mp4']))[0]} ({hdr.get('1-1-10') or tags.get('encoder') or 'video'}, {created})"}
    json.dump(state, open(p(P, "logs", "_state.json"), "w"), indent=2)
    log.info("Inspect step done in %.1f s", time.time() - t0)


# ============================================================================ EXTRACT
def step_extract(P, log, args):
    st = json.load(open(p(P, "logs", "_state.json")))
    tel = pd.read_pickle(p(P, "02_Telemetry", "_telemetry.pkl"))
    pts = tel["t"].to_numpy()
    idx = media.target_frames(st["n_frames"], st["fps"], args.fps, args.t_start, args.t_end)
    out_dir = os.path.join(P["out"], "03_Extracted_Frames", f"fps_{args.fps:g}")
    os.makedirs(out_dir, exist_ok=True)
    st["extract_fps"] = args.fps
    json.dump(st, open(p(P, "logs", "_state.json"), "w"), indent=2)
    fmt = args.format
    log.info("Extraction: %d target frames at %.3g fps -> %s", len(idx), args.fps, out_dir)
    if media.existing_complete(out_dir, idx, pts, fmt):
        log.info("All target frames already exist - skipping decode (no duplication).")
    else:
        # Staging folder keeps partial results of an interrupted run -> resume, never delete.
        stage = os.path.join(out_dir, "_staging")
        os.makedirs(stage, exist_ok=True)
        for f in os.listdir(out_dir):          # legacy layout: temp files directly in out_dir
            if f.startswith("_tmp_") and f.endswith(fmt):
                os.replace(os.path.join(out_dir, f), os.path.join(stage, f))
        done = media.staged_count(stage, fmt)
        k = max(0, done - 1)                    # last staged file may be truncated -> redo it
        if done:
            log.info("Resuming: %d staged frames found; re-decoding from target #%d", done, k + 1)
        hw = None
        if args.hwaccel != "none":
            cands = ["cuda", "d3d11va", "dxva2", "vaapi"] if args.hwaccel == "auto" else [args.hwaccel]
            hw = media.probe_hwaccel(P["mp4"], cands)
        log.info("Decoder: %s", f"hardware ({hw})" if hw else "software (libavcodec)")
        hf = 0.5 / st["fps"]
        t0 = time.time()
        r = subprocess.run(media.build_extract_cmd(P["mp4"], pts[idx[k:]], stage, hw, fmt, args.jpg_q,
                                                   k + 1, hf), capture_output=True, text=True)
        if r.returncode != 0 and hw:
            log.warning("Hardware decode failed (%s); retrying in software.", r.stderr.strip()[-300:])
            hw = None
            r = subprocess.run(media.build_extract_cmd(P["mp4"], pts[idx[k:]], stage, None, fmt, args.jpg_q,
                                                       k + 1, hf), capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("ffmpeg failed: " + r.stderr[-1000:])
        log.info("Decode finished in %.1f s (decoder: %s)", time.time() - t0, hw or "software")
        man = media.finalize_extraction(stage, out_dir, idx, pts, fmt)
        man["decoder"] = hw or "software"
        man.to_csv(os.path.join(out_dir, "frames_manifest.csv"), index=False)
    log.info("Frames present: %d", len(idx))


# ============================================================================ SYNC
def step_sync(P, log, args):
    st = json.load(open(p(P, "logs", "_state.json")))
    tel = pd.read_pickle(p(P, "02_Telemetry", "_telemetry.pkl"))
    out_dir = os.path.join(P["out"], "03_Extracted_Frames", f"fps_{args.fps:g}")
    man = pd.read_csv(os.path.join(out_dir, "frames_manifest.csv"))
    cfg = SyncConfig(offset_s=args.offset, allow_nearest=args.allow_nearest)
    sync = TelemetrySync(tel, cfg)
    q = sync.query(man["pts_s"].to_numpy())
    nn = sync.nearest_neighbour(man["pts_s"].to_numpy())
    ft = pd.concat([man[["image", "frame_index", "pts_s"]].reset_index(drop=True), q], axis=1)
    E, N, _, _ = g.latlon_to_utm(ft["lat_deg"], ft["lon_deg"], st["zone"], st["south"])
    ft["utm_e"], ft["utm_n"] = E, N
    xyz = g.geodetic_to_ecef(ft["lat_deg"], ft["lon_deg"], ft["abs_alt_m"])
    ft["ecef_x"], ft["ecef_y"], ft["ecef_z"] = xyz
    raw = tel.iloc[nn].reset_index(drop=True)
    ft["nn_frame_index"] = raw["frame_index"]
    ft["nn_lat_raw"], ft["nn_lon_raw"] = raw["lat"], raw["lon"]
    er, nr, _ = g.geodetic_to_enu(raw["lat"], raw["lon"], raw["abs_alt"], *sync.ref)
    ft["interp_minus_raw_h_m"] = np.hypot(ft["enu_e_m"] - er, ft["enu_n_m"] - nr)
    ft["srt_datetime"] = raw["srt_datetime"].astype(str)
    ft["sync_offset_s"] = args.offset
    ft.to_csv(p(P, "02_Telemetry", "frame_telemetry.csv"), index=False, float_format="%.9f")
    ft.to_csv(os.path.join(out_dir, "frame_telemetry.csv"), index=False, float_format="%.9f")
    log.info("frame_telemetry.csv: %d frames, quality %s", len(ft), ft["sync_quality"].value_counts().to_dict())
    finalize_reports(P, log, args, st, tel, ft, sync)


def finalize_reports(P, log, args, st, tel, ft, sync):
    info = json.load(open(p(P, "01_Metadata", "video_info.json")))
    hw = json.load(open(p(P, "01_Metadata", "hardware_runtime.json")))
    inv = json.load(open(p(P, "01_Metadata", "djmd_field_inventory.json")))
    utm_label = f"WGS84 / UTM zone {st['zone']}{'S' if st['south'] else 'N'}, EPSG:{st['epsg']}"
    ft2 = ft.rename(columns={})
    reports.plot_trajectory(tel, ft2, utm_label, p(P, "02_Telemetry", "gps_trajectory.png"))
    for f in ("gps_trajectory.png", "altitude_profile.png", "sync_quality.png"):
        shutil.copy2(p(P, "02_Telemetry", f), p(P, "09_Reports", f))

    # integrity: sources untouched
    now = {k: (os.path.getsize(P[k]), os.path.getmtime(P[k])) for k in ("mp4", "srt") if P[k]}
    untouched = all(tuple(now[k]) == tuple(st["src_stat"][k]) for k in now)

    E, N = tel["utm_e"], tel["utm_n"]
    epochs = tel["pos_is_update_epoch"].sum()
    ex_dir = os.path.join(P["out"], "03_Extracted_Frames", f"fps_{args.fps:g}")
    man = pd.read_csv(os.path.join(ex_dir, "frames_manifest.csv"))
    seg = np.hypot(np.diff(E[tel["pos_is_update_epoch"]]), np.diff(N[tel["pos_is_update_epoch"]]))
    qc = ft["sync_quality"].value_counts().to_dict()
    qc_all = tel["sync_quality"].value_counts().to_dict()
    take_off_abs = float((tel["abs_alt"] - tel["rel_alt"]).median())
    gsd = None
    # nadir-equivalent GSD at 24 mm-equiv (M3E wide: 4/3" 17.3 mm sensor, f = 12.29 mm real) - only if known
    fov_note = ("Real focal length/sensor size are not stored in the telemetry; intrinsics will be "
                "self-calibrated by COLMAP in Phase 2 (35 mm-equivalent focal 24 mm from SRT).")

    blocks = [
        ("FILES", [("video", info["source"]["video"]), ("video size (bytes)", info["source"]["video_bytes"]),
                   ("srt", info["source"]["srt"]), ("srt size (bytes)", info["source"]["srt_bytes"]),
                   ("sources unmodified after processing", untouched),
                   ("output root", P["out"])]),
        ("VIDEO", [("codec / profile", f"{info['video']['codec']} / {info['video']['profile']}"),
                   ("resolution", f"{info['video']['width']} x {info['video']['height']}"),
                   ("frame rate", f"{info['video']['fps']:.5f} fps ({info['video']['fps_rational']}), "
                                  f"CFR={info['video']['constant_frame_rate']}"),
                   ("frames", info["video"]["frames"]), ("duration (s)", round(info["container"]["duration_s"], 3)),
                   ("bit rate (Mb/s)", round(info["video"]["bit_rate_bps"] / 1e6, 1)),
                   ("keyframes / median GOP", f"{info['video']['keyframes']} / {info['video']['gop_frames_median']}"),
                   ("color", f"{info['video']['color_primaries']} / {info['video']['color_transfer']} (SDR)"),
                   ("other streams", "; ".join(f"#{s['index']} {s['type']} {s['tag'] or s['codec']} "
                                               f"{s['handler'] or ''} {s['width'] or ''}x{s['height'] or ''}".strip()
                                               for s in info["other_streams"])),
                   ("creation time (UTC)", info["container"]["creation_time_utc"])]),
        ("CAMERA", [("model (container encoder)", info["camera"]["model_container"]),
                    ("model (djmd)", info["camera"]["model_djmd"]),
                    ("serial (djmd header)", info["camera"]["serial_djmd"]),
                    ("metadata proto", f"{info['camera']['proto']} {info['camera']['proto_versions']}"),
                    ("focal length (35mm-equiv, SRT)", info["camera"]["focal_len_35mm_equiv_mm_srt"]),
                    ("f-number", info["camera"]["f_number"]), ("ISO range", info["camera"]["iso_range"]),
                    ("shutter range", info["camera"]["shutter_range"]),
                    ("digital zoom", info["camera"]["digital_zoom_range"]),
                    ("intrinsics", fov_note)]),
        ("GPS / TELEMETRY AVAILABILITY", [
            ("embedded per-frame GPS (djmd)", info["gps_availability"]["embedded_djmd_per_frame"]),
            ("SRT per-frame GPS", info["gps_availability"]["srt_per_frame"]),
            ("container location tag", info["gps_availability"]["container_level_location_tag"]),
            ("verdict", info["gps_availability"]["verdict"]),
            ("GNSS update rate (Hz)", info["gps_availability"]["gnss_update_rate_hz"]),
            ("GNSS update epochs", int(epochs)),
            ("RTK", info["gps_availability"]["rtk"]),
            ("SRT timestamp format", info["telemetry_time"]["srt_timestamp_format"]),
            ("SRT local datetime span", f"{info['telemetry_time']['srt_datetime_first']} -> "
                                        f"{info['telemetry_time']['srt_datetime_last']}"),
            ("SRT local time vs container UTC (h)", info["telemetry_time"]["utc_offset_inferred_h"]),
            ("ExifTool", "available" if hw["tools"].get("exiftool") else
             "not available in runtime - djmd decoded natively")]),
        ("FIELDS DETECTED", [
            ("SRT fields", "FrameCnt, DiffTime, datetime, iso, shutter, fnum, ev, color_md, ae_meter_md, focal_len, "
                           "dzoom_ratio, latitude, longitude, rel_alt, abs_alt, gb_yaw, gb_pitch, gb_roll"),
            ("djmd fields (decoded)", "clock_us, sensor_frame_seq, lat/lon (rad, float64), abs_alt (mm), rel_alt (mm), "
                                      "aircraft roll/pitch/yaw (0.1 deg), velocity N/E (m/s, verified), vertical "
                                      "velocity (sign unverified), gimbal pitch/roll/yaw (0.1 deg), ISO, colour temp, zoom"),
            ("aircraft yaw/pitch/roll", "djmd only (NOT in SRT); roll/pitch labels follow ExifTool DJI convention"),
            ("gimbal yaw/pitch/roll", "SRT and djmd (identical)"),
            ("flight speed", "djmd velocity N/E (+ GNSS-derived check); no speed field in SRT"),
            ("vertical velocity vs GNSS dU/dt corr", round(st["vz_corr"], 3)),
            ("unmapped djmd paths", ", ".join(k for k, v in inv.items() if not v["mapped_to"]))]),
        ("FLIGHT SUMMARY", [
            ("start lat/lon (deg)", f"{tel['lat'].iloc[0]:.8f}, {tel['lon'].iloc[0]:.8f}"),
            ("end lat/lon (deg)", f"{tel['lat'].iloc[-1]:.8f}, {tel['lon'].iloc[-1]:.8f}"),
            ("projected CRS", utm_label),
            ("extent E x N (m)", f"{E.max() - E.min():.1f} x {N.max() - N.min():.1f}"),
            ("path length (m, epoch-to-epoch)", round(float(seg.sum()), 1)),
            ("relative alt (m) min/mean/max", f"{tel['rel_alt'].min():.2f} / {tel['rel_alt'].mean():.2f} / "
                                              f"{tel['rel_alt'].max():.2f}"),
            ("absolute alt (m) min/max", f"{tel['abs_alt'].min():.2f} / {tel['abs_alt'].max():.2f}"),
            ("implied take-off abs alt (abs-rel, m)", round(take_off_abs, 2)),
            ("altitude reference", "abs_alt datum NOT stated by DJI metadata (ellipsoidal vs MSL unconfirmed); "
                                   "rel_alt = barometric height above take-off. Must be resolved with GCP/known "
                                   "height before metric vertical georeferencing."),
            ("horizontal speed (m/s) mean/max (djmd)", f"{tel['speed_h_ms'].mean():.2f} / {tel['speed_h_ms'].max():.2f}"),
            ("gimbal pitch (deg) min/max", f"{tel['gimbal_pitch'].min():.1f} / {tel['gimbal_pitch'].max():.1f}"),
            ("gimbal yaw (deg) range", f"{tel['gimbal_yaw'].min():.1f} .. {tel['gimbal_yaw'].max():.1f}"),
            ("aircraft pitch/roll (deg) abs max", f"{tel['drone_pitch'].abs().max():.1f} / {tel['drone_roll'].abs().max():.1f}")]),
        ("FRAME EXTRACTION", [
            ("rate (fps)", args.fps), ("frames extracted", len(man)),
            ("first / last source frame", f"{man['frame_index'].min()} / {man['frame_index'].max()}"),
            ("format", f"{args.format} (q={args.jpg_q})" if args.format == "jpg" else args.format),
            ("resolution", "original (3840x2160)"),
            ("decoder", man["decoder"].iloc[0] if "decoder" in man else "n/a"),
            ("total size (MB)", round(man["bytes"].sum() / 2**20, 1)),
            ("folder", ex_dir),
            ("naming", "F<source frame index>_T<presentation time ms>.jpg")]),
        ("SYNCHRONISATION QUALITY", [
            ("time base", "video PTS; telemetry_time = frame_time + offset"),
            ("offset applied (s)", args.offset),
            ("SRT cue start vs video PTS |max| (ms)", round(float(tel["srt_minus_video_pts_s"].abs().max() * 1000), 2)),
            ("djmd PTS vs video PTS |max| (ms)", round(float(tel["djmd_minus_video_pts_s"].abs().max() * 1000), 3)),
            ("SRT FrameCnt contiguous", bool((np.diff(tel["frame_cnt"]) == 1).all()) if tel["frame_cnt"].notna().any() else "n/a (no SRT)"),
            ("SRT records == video frames", len(tel) == tel["frame_cnt"].notna().sum()),
            ("djmd sensor frame-seq skips", info["telemetry_time"]["djmd_sensor_frame_seq_skips"]),
            ("SRT wall-clock jitter", "datetime increments 12-62 ms (not used for timing; PTS used)"),
            ("SRT vs djmd horizontal diff (cm) median/max", f"{tel['srt_vs_djmd_h_m'].median() * 100:.1f} / "
                                                            f"{tel['srt_vs_djmd_h_m'].max() * 100:.1f}"),
            ("missing/zero GPS records", int(tel["telemetry_missing"].sum())),
            ("ALL frames quality", qc_all),
            ("EXTRACTED frames quality", qc),
            ("bracket gap (s) median/max", f"{ft['pos_bracket_gap_s'].median():.4f} / {ft['pos_bracket_gap_s'].max():.4f}"),
            ("interp vs raw per-frame value (m) mean/max",
             f"{ft['interp_minus_raw_h_m'].mean():.3f} / {ft['interp_minus_raw_h_m'].max():.3f}"),
            ("unresolved", "absolute GNSS->image latency is unknown; estimate offset in Phase 3 by minimising "
                           "COLMAP-vs-GNSS residuals over a range of offsets"),
            ("accuracy caveat", "standard DJI GNSS, no RTK: expect metre-level absolute error; positions are "
                                "georeferencing observations, not ground truth")]),
        ("RUNTIME HARDWARE (where Phase 1 executed)", [
            ("platform", hw["platform"]), ("CPU", hw["cpu_model"]), ("logical CPUs", hw["logical_cpus"]),
            ("RAM (GB)", hw.get("ram_total_gb")), ("NVIDIA GPUs", hw["nvidia_gpus"] or "none visible"),
            ("disk free on output volume (GB)", hw["disk_free_gb"]),
            ("ffmpeg", hw["tools"].get("ffmpeg_version")), ("exiftool", hw["tools"].get("exiftool")),
            ("colmap", hw["tools"].get("colmap")),
            ("note", "Phase 1 ran in Claude's local sandbox VM on this laptop. Run hwcheck.py natively on "
                     "Windows to detect the real GPU/VRAM before Phase 2/4.")]
         + [("pre-flight", w) for w in media.requirement_warnings(hw, len(man), st["width"], st["height"],
                                                                     man["bytes"].mean() / 2**20)]),
    ]
    if args.display_source:
        ds = args.display_source.rstrip("\\/")
        sep = "\\" if ":" in ds[:3] else "/"

        def fix(v):
            if isinstance(v, str) and P["src"] in v:
                return v.replace(P["src"], ds).replace("/", sep)
            return v
        blocks = [(name, [(k, fix(v)) for k, v in rows]) for name, rows in blocks]
    reports.write_text_report(p(P, "01_Metadata", "metadata_report.txt"), blocks)
    shutil.copy2(p(P, "01_Metadata", "metadata_report.txt"), p(P, "09_Reports", "metadata_report.txt"))
    shutil.copy2(p(P, "01_Metadata", "video_info.json"), p(P, "09_Reports", "video_info.json"))
    reports.write_html(p(P, "09_Reports", "telemetry_analysis.html"), "GeoSplat · Telemetry analysis — "
                       + os.path.basename(P["mp4"]), blocks,
                       [("GNSS trajectory", p(P, "02_Telemetry", "gps_trajectory.png")),
                        ("Altitude / speed / attitude", p(P, "02_Telemetry", "altitude_profile.png")),
                        ("Synchronisation diagnostics", p(P, "02_Telemetry", "sync_quality.png"))])
    shutil.copy2(p(P, "09_Reports", "telemetry_analysis.html"), p(P, "02_Telemetry", "telemetry_analysis.html"))
    log.info("Reports written. Sources untouched: %s", untouched)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="DJI video (.MP4)")
    ap.add_argument("--srt", default=None, help="DJI .SRT (default: auto-detect next to the video)")
    ap.add_argument("--project", default=None, help="output folder (default: <video>_GeoSplat)")
    ap.add_argument("--step", default="all", choices=["inspect", "extract", "sync", "all"])
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--t-start", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--jpg-q", type=int, default=2)
    ap.add_argument("--hwaccel", default="auto", help="auto | none | cuda | d3d11va | qsv | dxva2")
    ap.add_argument("--offset", type=float, default=0.0, help="telemetry time offset (s)")
    ap.add_argument("--allow-nearest", action="store_true")
    ap.add_argument("--display-source", default=None,
                    help="path to show in reports when running inside a mount (e.g. 'F:\\ITS RESEARCH CENTER')")
    args = ap.parse_args()
    P, log = setup(args)
    log.info("GeoSplat stage 1 | step=%s | video=%s | srt=%s | project=%s", args.step, P["mp4"], P["srt"], P["out"])
    if args.step in ("inspect", "all"):
        step_inspect(P, log, args)
    if args.step in ("extract", "all"):
        step_extract(P, log, args)
    if args.step in ("sync", "all"):
        step_sync(P, log, args)


if __name__ == "__main__":
    main()
