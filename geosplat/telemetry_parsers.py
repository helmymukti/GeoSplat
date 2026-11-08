"""
Telemetry parsers for DJI videos.

1. parse_srt(path)        -> DataFrame, one row per SRT subtitle block (DJI per-frame SRT)
2. parse_djmd(mp4, ...)   -> DataFrame, one row per packet of the embedded 'DJI meta'
                             (codec tag 'djmd') data track, decoded from protobuf.

The MP4 is NEVER loaded into RAM: packet byte offsets/sizes come from ffprobe and
each packet (~230 bytes) is read with a seek on the original file (read-only).
References: Google Protocol Buffers wire format; DJI 'dvtm' tag naming after P. Harvey, ExifTool DJI tags.
Field map verified empirically for dvtm_wm265e (docs/METHODS.md section 1.2).
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- SRT
_BLOCK = re.compile(
    r"(\d+)\s*\n(\d\d:\d\d:\d\d[,.]\d{3})\s*-->\s*(\d\d:\d\d:\d\d[,.]\d{3})\s*\n(.*?)(?=\n\s*\n\d+\s*\n\d\d:|\Z)",
    re.S,
)
_KV = re.compile(r"([A-Za-z_]+)\s*:\s*([^\s\[\],]+)")
_DT = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[.,]\d+)?)")
_GPS_OLD = re.compile(r"GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*(?:,\s*([-\d.]+)\s*)?\)")
_BARO = re.compile(r"(?:BAROMETER|\bH)\s*[:]?\s*([-\d.]+)\s*m?\b")


def _tc(s: str) -> float:
    h, m, rest = s.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _num(v: str):
    try:
        if "/" in v:
            a, b = v.split("/")
            return float(a) / float(b)
        return float(v)
    except ValueError:
        return v


def parse_srt(path: str) -> pd.DataFrame:
    txt = open(path, "r", encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    rows = []
    for m in _BLOCK.finditer(txt):
        idx, t0, t1, body = m.groups()
        body = re.sub(r"<[^>]+>", " ", body)
        r = {"srt_index": int(idx), "srt_start_s": _tc(t0), "srt_end_s": _tc(t1)}
        dm = _DT.search(body)
        r["srt_datetime"] = dm.group(1) if dm else None
        for k, v in _KV.findall(body):
            k = k.strip().lower()
            if k in ("framecnt", "difftime"):
                v = v.replace("ms", "")
            if k == "shutter":
                r["shutter_str"] = v
            r[k] = _num(v)
        if "latitude" not in r:                      # older DJI format: GPS(lon, lat, alt) / BAROMETER / H
            m = _GPS_OLD.search(body)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                lon, lat = (a, b) if abs(b) <= 90 else (b, a)
                r["latitude"], r["longitude"] = lat, lon
                if m.group(3):
                    r["abs_alt"] = float(m.group(3))
            m = _BARO.search(body)
            if m and "rel_alt" not in r:
                r["rel_alt"] = float(m.group(1))
        rows.append(r)
    df = pd.DataFrame(rows)
    if "difftime" in df:
        df = df.rename(columns={"difftime": "difftime_ms", "framecnt": "frame_cnt"})
    if "srt_datetime" in df:
        df["srt_datetime"] = pd.to_datetime(df["srt_datetime"], errors="coerce")
    if "shutter" in df:
        df = df.rename(columns={"shutter": "shutter_s"})
    return df


# ----------------------------------------------------------------------------- protobuf
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def pb_parse(b: bytes):
    """Schema-less protobuf decoder. Returns list of (field, wiretype, value)."""
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i)
        f, t = k >> 3, k & 7
        if f == 0:
            raise ValueError("field 0")
        if t == 0:
            v, i = _varint(b, i)
            if v >= 1 << 63:
                v -= 1 << 64
        elif t == 1:
            v = struct.unpack("<d", b[i:i + 8])[0]
            i += 8
        elif t == 5:
            v = struct.unpack("<f", b[i:i + 4])[0]
            i += 4
        elif t == 2:
            n, i = _varint(b, i)
            raw = b[i:i + n]
            i += n
            try:
                v = pb_parse(raw) if n else raw
            except Exception:
                v = raw
        else:
            raise ValueError(f"wiretype {t}")
        if i > len(b):
            raise ValueError("overrun")
        out.append((f, t, v))
    return out


def pb_flatten(msg, prefix="", out=None):
    out = {} if out is None else out
    for f, t, v in msg:
        key = f"{prefix}{f}"
        if isinstance(v, list):
            pb_flatten(v, key + "-", out)
        else:
            out.setdefault(key, v)
    return out


# Field map for dvtm_wm265e.proto (Mavic 3 Enterprise), established by decoding this
# file and cross-checking against the SRT. Angles are int deci-degrees.
# proto3 omits fields whose value is 0 -> an absent angle field means 0.0.
DJMD_MAP = {
    "3-1-1": ("djmd_sensor_frame_seq", 1.0),
    "3-1-2": ("djmd_clock_us", 1.0),
    # Velocity triplet: 1 and 2 verified against GPS-derived velocity on this file
    # (corr 0.999 with North and East respectively). 3 = vertical, sign NOT verified.
    "3-3-2-1": ("djmd_vel_north_ms", 1.0),
    "3-3-2-2": ("djmd_vel_east_ms", 1.0),
    "3-3-2-3": ("djmd_vel_vertical_raw_ms", 1.0),
    "3-3-3-1": ("drone_roll_deg", 0.1),     # ExifTool DJI DroneInfo: 1=roll, 2=pitch, 3=yaw
    "3-3-3-2": ("drone_pitch_deg", 0.1),
    "3-3-3-3": ("drone_yaw_deg", 0.1),
    "3-3-4-1-2": ("djmd_lat_rad", 1.0),
    "3-3-4-1-3": ("djmd_lon_rad", 1.0),
    "3-3-4-2": ("djmd_abs_alt_m", 0.001),
    "3-3-4-4": ("djmd_gps_status_raw", 1.0),
    "3-3-5-1": ("djmd_rel_alt_m", 0.001),
    "3-4-3-1": ("djmd_gimbal_pitch_deg", 0.1),
    "3-4-3-2": ("djmd_gimbal_roll_deg", 0.1),
    "3-4-3-3": ("djmd_gimbal_yaw_deg", 0.1),
    "3-2-2-1": ("djmd_iso", 1.0),
    "3-2-7-1": ("djmd_color_temp_k", 1.0),
    "3-2-6-1": ("djmd_digital_zoom", 1.0),
}
ZERO_DEFAULT = {"3-3-3-1", "3-3-3-2", "3-3-3-3", "3-4-3-1", "3-4-3-2", "3-4-3-3",
                "3-3-2-1", "3-3-2-2", "3-3-2-3"}
# Sub-trees that hold opaque binary blobs (mis-decode as protobuf -> nonsense numbers)
OPAQUE_PREFIXES = ("3-2-9-", "3-2-3-", "3-2-4-", "3-2-5-")


def ffprobe_packets(mp4: str, stream_index: int):
    cmd = ["ffprobe", "-v", "error", "-select_streams", str(stream_index), "-show_entries",
           "packet=pts_time,pos,size,flags", "-of", "json", mp4]
    js = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return js.get("packets", [])


def parse_djmd(mp4: str, stream_index: int):
    """Returns (DataFrame per packet, header dict, field_stats dict)."""
    pk = ffprobe_packets(mp4, stream_index)
    rows, header, stats = [], {}, defaultdict(lambda: [0, None, None, set()])
    with open(mp4, "rb") as fh:
        for k, p in enumerate(pk):
            fh.seek(int(p["pos"]))
            b = fh.read(int(p["size"]))
            flat = pb_flatten(pb_parse(b))
            if k == 0:
                for key, v in flat.items():
                    if key.startswith("1-") and isinstance(v, (bytes, bytearray)):
                        try:
                            header[key] = v.decode("ascii")
                        except UnicodeDecodeError:
                            pass
                    elif key.startswith("2-"):
                        header[key] = v
            r = {"djmd_packet": k, "djmd_pts_s": float(p["pts_time"])}
            for key, (name, sc) in DJMD_MAP.items():
                if key in flat and not isinstance(flat[key], (bytes, bytearray)):
                    r[name] = flat[key] * sc
                elif key in ZERO_DEFAULT:
                    r[name] = 0.0
                else:
                    r[name] = np.nan
            for key, v in flat.items():
                if (key.startswith("3-") and not key.startswith(OPAQUE_PREFIXES)
                        and not isinstance(v, (bytes, bytearray))):
                    s = stats[key]
                    s[0] += 1
                    s[1] = v if s[1] is None else min(s[1], v)
                    s[2] = v if s[2] is None else max(s[2], v)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["djmd_lat_deg"] = np.degrees(df["djmd_lat_rad"])
    df["djmd_lon_deg"] = np.degrees(df["djmd_lon_rad"])
    fstats = {k: {"count": v[0], "min": v[1], "max": v[2],
                  "mapped_to": DJMD_MAP.get(k, (None,))[0]} for k, v in sorted(stats.items())}
    return df, header, fstats
