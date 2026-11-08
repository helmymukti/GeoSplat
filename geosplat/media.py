"""
Video probing, hardware detection and frame extraction (FFmpeg, streaming only).
The source MP4 is opened read-only by ffprobe/ffmpeg; nothing is loaded into RAM.
Uses FFmpeg/ffprobe (FFmpeg Developers). See docs/METHODS.md section 2.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from fractions import Fraction

import numpy as np
import pandas as pd


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------------------- probe
def ffprobe_json(mp4):
    r = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", mp4], check=True)
    return json.loads(r.stdout)


def video_packet_table(mp4, stream=0):
    r = run(["ffprobe", "-v", "error", "-select_streams", str(stream), "-show_entries",
             "packet=pts_time,dts_time,size,flags", "-of", "csv=p=0", mp4], check=True)
    rows = [l.split(",") for l in r.stdout.strip().splitlines()]
    df = pd.DataFrame(rows, columns=["pts_time", "dts_time", "size", "flags"][:len(rows[0])])
    df["pts_time"] = df["pts_time"].astype(float)
    df["size"] = df["size"].astype(int)
    df = df.sort_values("pts_time").reset_index(drop=True)
    df["frame_index"] = np.arange(len(df))
    df["keyframe"] = df["flags"].str.contains("K")
    return df


# ----------------------------------------------------------------------------- hardware
def detect_hardware(path_for_disk):
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        m = re.search(r"model name\s*:\s*(.+)", open("/proc/cpuinfo").read())
        cpu = m.group(1).strip() if m else cpu
    elif platform.system() == "Windows":
        r = run(["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"])
        cpu = r.stdout.strip() or cpu
    info["cpu_model"] = cpu
    # RAM
    try:
        import psutil  # optional
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 1)
    except Exception:
        if hasattr(os, "sysconf"):
            try:
                info["ram_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
            except Exception:
                pass
        if platform.system() == "Windows" and "ram_total_gb" not in info:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            info["ram_total_gb"] = round(ms.tot / 2**30, 1)
            info["ram_available_gb"] = round(ms.avail / 2**30, 1)
    # GPU
    gpus = []
    if shutil.which("nvidia-smi"):
        r = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"])
        for l in r.stdout.strip().splitlines():
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 4:
                gpus.append({"name": p[0], "vram_total_mb": int(p[1]), "vram_free_mb": int(p[2]),
                             "driver": p[3]})
    info["nvidia_gpus"] = gpus
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = torch.version.cuda
    except Exception:
        info["torch_cuda"] = None
    du = shutil.disk_usage(path_for_disk)
    info["disk_free_gb"] = round(du.free / 2**30, 1)
    info["disk_total_gb"] = round(du.total / 2**30, 1)
    tools = {}
    for t in ["ffmpeg", "ffprobe", "exiftool", "colmap", "nvidia-smi"]:
        tools[t] = shutil.which(t)
    if tools["ffmpeg"]:
        tools["ffmpeg_version"] = run(["ffmpeg", "-version"]).stdout.splitlines()[0]
        tools["ffmpeg_hwaccels"] = run(["ffmpeg", "-hide_banner", "-hwaccels"]).stdout.split()[3:]
    info["tools"] = tools
    return info


def requirement_warnings(hw, n_frames, width, height, est_jpg_mb):
    """Rough pre-flight estimates for later phases. Returns list of strings."""
    w = []
    frames_gb = n_frames * est_jpg_mb / 1024
    colmap_gb = n_frames * 0.006 + 0.3          # SIFT db (~8k feats/img) + matches + sparse model
    gs_gb = 2.0                                  # checkpoints + PLYs (30k iters)
    need = frames_gb + colmap_gb + gs_gb
    w.append(f"Estimated disk for frames+COLMAP+3DGS: ~{need:.1f} GB "
             f"(frames {frames_gb:.1f} GB, COLMAP ~{colmap_gb:.1f} GB, 3DGS ~{gs_gb:.1f} GB).")
    if hw.get("disk_free_gb", 1e9) < need * 1.5:
        w.append("WARNING: free disk space below 1.5x the estimate.")
    if not hw.get("nvidia_gpus"):
        w.append("WARNING: no NVIDIA GPU visible from this runtime -> Graphdeco 3DGS training "
                 "(CUDA-only) cannot run here; COLMAP would be CPU-only (slow).")
    else:
        v = max(g["vram_total_mb"] for g in hw["nvidia_gpus"]) / 1024
        # Graphdeco README: 24 GB for paper-quality training; fewer images / -r 2 reduces need.
        if v < 24:
            w.append(f"NOTE: {v:.0f} GB VRAM < 24 GB recommended by Graphdeco for full quality; "
                     f"plan to train at -r 2 (1920x1080) or -r 4, and/or limit densification.")
    if width * height > 1920 * 1080:
        w.append("NOTE: 4K frames: 3DGS will be trained at downscaled resolution (-r 2 or 4) "
                 "unless VRAM >= 24 GB.")
    return w


# ----------------------------------------------------------------------------- extraction
def target_frames(n_frames, fps_video, fps_out, t_start=0.0, t_end=None):
    dur = n_frames / fps_video
    t_end = dur if t_end is None else min(t_end, dur)
    ts = np.arange(t_start, t_end, 1.0 / fps_out)
    idx = np.unique(np.clip(np.round(ts * fps_video).astype(int), 0, n_frames - 1))
    return idx


_HW_FAIL = ("fail", "cannot", "could not", "no device", "unable", "not supported", "error")


def probe_hwaccel(mp4, candidates):
    """Return the first hwaccel that really initialises and decodes 1 s, else None.
    FFmpeg can silently fall back to software, so the verbose log is inspected for
    device/initialisation failures, not just the exit code. ('qsv' is excluded from
    auto mode: with plain -hwaccel qsv FFmpeg ignores it unless h264_qsv is selected.)"""
    for hw in candidates:
        try:
            r = run(["ffmpeg", "-hide_banner", "-v", "verbose", "-hwaccel", hw, "-t", "1", "-i", mp4,
                     "-map", "0:v:0", "-f", "null", "-"], timeout=120)
        except Exception:
            continue
        log = "\n".join(l for l in r.stderr.lower().splitlines() if "hw" in l or "device" in l
                        or "hwaccel" in l or hw in l)
        if r.returncode == 0 and not any(k in log for k in _HW_FAIL):
            return hw
    return None


def frame_passthrough_args():
    """'-fps_mode passthrough' (FFmpeg >= 5.1) or '-vsync 0' (older builds)."""
    r = run(["ffmpeg", "-hide_banner", "-h", "long"])
    return ["-fps_mode", "passthrough"] if "fps_mode" in r.stdout else ["-vsync", "0"]


def build_extract_cmd(mp4, target_pts, stage_dir, hwaccel=None, fmt="jpg", jpg_q=2,
                      start_number=1, half_frame_s=0.5 / 29.97):
    """Select frames by exact presentation time (robust to input seeking).
    With -ss before -i plus -copyts, 't' keeps the original timeline, so a resumed
    run can jump straight to the first missing frame instead of decoding from 0."""
    sel = "+".join(f"lt(abs(t-{t:.6f})\\,{half_frame_s:.6f})" for t in target_pts)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-progress",
           os.path.join(stage_dir, "_progress.txt")]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    seek = max(0.0, float(target_pts[0]) - 3.0)
    if seek > 0:
        cmd += ["-ss", f"{seek:.3f}"]
    # stop reading right after the last wanted frame (input duration limit, relative to the seek point)
    cmd += ["-t", f"{float(target_pts[-1]) - seek + 1.0:.3f}"]
    cmd += ["-copyts", "-i", mp4, "-map", "0:v:0", "-vf", f"select='{sel}'"] + frame_passthrough_args()
    cmd += ["-start_number", str(start_number)]
    if fmt == "jpg":
        cmd += ["-q:v", str(jpg_q), os.path.join(stage_dir, "_tmp_%05d.jpg")]
    else:
        cmd += ["-compression_level", "3", os.path.join(stage_dir, "_tmp_%05d.png")]
    return cmd


def staged_count(stage_dir, fmt):
    return len([f for f in os.listdir(stage_dir) if f.startswith("_tmp_") and f.endswith(fmt)])


def frame_name(idx, t, fmt="jpg"):
    return f"F{idx:05d}_T{int(round(t * 1000)):08d}ms.{fmt}"


def finalize_extraction(stage_dir, out_dir, frame_idx, pts, fmt="jpg"):
    """Move sequential ffmpeg outputs (staging) to frame-index/timestamp names; manifest.
    Uses rename only - never deletes anything."""
    tmp = sorted(f for f in os.listdir(stage_dir) if f.startswith("_tmp_") and f.endswith(fmt))
    if len(tmp) != len(frame_idx):
        raise RuntimeError(f"ffmpeg produced {len(tmp)} images, expected {len(frame_idx)}")
    rows = []
    for f, i in zip(tmp, frame_idx):
        name = frame_name(int(i), float(pts[i]), fmt)
        os.replace(os.path.join(stage_dir, f), os.path.join(out_dir, name))
        rows.append({"image": name, "frame_index": int(i), "pts_s": float(pts[i]),
                     "bytes": os.path.getsize(os.path.join(out_dir, name))})
    return pd.DataFrame(rows)


def existing_complete(out_dir, frame_idx, pts, fmt="jpg"):
    names = [frame_name(int(i), float(pts[i]), fmt) for i in frame_idx]
    return all(os.path.exists(os.path.join(out_dir, n)) for n in names)
