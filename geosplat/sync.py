"""
Telemetry <-> frame synchronisation engine.

Model
-----
* Time base = video presentation time (s) of each frame, from the MP4 container.
* DJI writes one telemetry record per video frame (SRT block and 'djmd' packet), but
  the GNSS position/altitude only refresh at ~10 Hz, so consecutive records repeat
  the same coordinate. Treating each repeated value as valid "at that frame" would
  produce a staircase trajectory with up to ~0.1 s x speed of error.
* Therefore each channel is reduced to its UPDATE EPOCHS (records where the value
  changes); epoch time = time of the first frame carrying the new value.
* Query time  t_tel = t_frame + offset_s   (offset_s = synchronisation correction,
  positive if telemetry lags the image; to be estimated in Phase 2/3 from COLMAP).
* Positions are interpolated linearly in a local ENU frame (metres), then converted
  back to WGS84 — never by interpolating degrees directly.
* Angles are interpolated through their unwrapped form (no 359->0 jumps).
* Quality flags: GOOD    - bracketed by epochs, gap <= good_gap_s
                 DEGRADED- bracketed, good_gap_s < gap <= max_gap_s
                 BAD     - outside telemetry span or gap > max_gap_s or missing
  BAD rows get NaN coordinates unless allow_nearest=True (then nearest + flag).
Method: GeoSplat-specific epoch-based interpolation in local ENU (docs/METHODS.md section 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import geodesy as g


@dataclass
class SyncConfig:
    offset_s: float = 0.0
    good_gap_s: float = 0.15       # 1.5 x nominal 10 Hz GNSS update
    max_gap_s: float = 0.50
    allow_nearest: bool = False


def update_epochs(t, *cols, invalid_zero=True):
    """Mask of rows where any of cols changes value (first row included), skipping invalid."""
    arr = np.column_stack([np.asarray(c, float) for c in cols])
    valid = np.all(np.isfinite(arr), axis=1)
    if invalid_zero:
        valid &= ~np.all(arr == 0, axis=1)
    m = np.zeros(len(t), bool)
    last = None
    for i in range(len(t)):
        if not valid[i]:
            continue
        if last is None or np.any(arr[i] != last):
            m[i] = True
            last = arr[i]
    return m


class TelemetrySync:
    def __init__(self, tel: pd.DataFrame, cfg: SyncConfig = SyncConfig()):
        """tel: per-record telemetry with columns
           t, lat, lon, abs_alt, rel_alt, gimbal_yaw, gimbal_pitch, gimbal_roll,
           drone_yaw, drone_pitch, drone_roll (any angle col may be absent)."""
        self.cfg = cfg
        self.t = tel["t"].to_numpy(float)
        self.ref = (float(tel["lat"].dropna().iloc[0]), float(tel["lon"].dropna().iloc[0]),
                    float(tel["abs_alt"].dropna().iloc[0]))
        m = update_epochs(self.t, tel["lat"], tel["lon"])
        self.pos_epoch_mask = m
        self.te = self.t[m]
        e, n, u = g.geodetic_to_enu(tel["lat"][m], tel["lon"][m], tel["abs_alt"][m], *self.ref)
        self.E, self.N, self.U = e, n, u
        ma = update_epochs(self.t, tel["rel_alt"], invalid_zero=False)
        self.ta, self.rel = self.t[ma], tel["rel_alt"].to_numpy(float)[ma]
        self.angles = {c: tel[c].to_numpy(float) for c in
                       ["gimbal_yaw", "gimbal_pitch", "gimbal_roll", "drone_yaw", "drone_pitch", "drone_roll"]
                       if c in tel}
        dte = np.diff(self.te)
        self.nominal_dt = float(np.median(dte)) if len(dte) else np.nan

    # ------------------------------------------------------------------
    def query(self, t_frame):
        c = self.cfg
        tq = np.asarray(t_frame, float) + c.offset_s
        te = self.te
        j = np.searchsorted(te, tq, side="right")      # te[j-1] <= tq < te[j]
        inside = (j > 0) & (j < len(te))
        exact_end = np.isclose(tq, te[-1]) if len(te) else np.zeros_like(tq, bool)
        jl = np.clip(j - 1, 0, len(te) - 1)
        jr = np.clip(j, 0, len(te) - 1)
        gap = np.where(inside, te[jr] - te[jl], np.nan)
        dnear = np.minimum(np.abs(tq - te[jl]), np.abs(te[jr] - tq))

        E = np.interp(tq, te, self.E)
        N = np.interp(tq, te, self.N)
        U = np.interp(tq, te, self.U)
        lat, lon, h = g.enu_to_geodetic(E, N, U, *self.ref)

        q = np.where(inside & (gap <= c.good_gap_s), "GOOD",
                     np.where(inside & (gap <= c.max_gap_s), "DEGRADED", "BAD"))
        q = np.where(exact_end & (q == "BAD"), "GOOD", q)
        method = np.where(inside, "linear_enu", "outside_span")
        bad = q == "BAD"
        if bad.any():
            if c.allow_nearest:
                method = np.where(bad, "nearest_flagged", method)
            else:
                for arr in (lat, lon, h, E, N, U):
                    arr[bad] = np.nan
                method = np.where(bad, "rejected", method)

        out = pd.DataFrame({
            "t_frame_s": np.asarray(t_frame, float), "t_telemetry_s": tq,
            "lat_deg": lat, "lon_deg": lon, "abs_alt_m": h,
            "rel_alt_m": np.interp(tq, self.ta, self.rel),
            "enu_e_m": E, "enu_n_m": N, "enu_u_m": U,
            "pos_bracket_gap_s": gap, "pos_dt_to_nearest_epoch_s": dnear,
            "interp_method": method, "sync_quality": q,
        })
        for k, a in self.angles.items():
            out[k + "_deg"] = g.interp_angle_deg(tq, self.t, a)
        # speed from interpolated epochs (central difference on epoch track)
        if len(te) > 2:
            ve = np.interp(tq, te, np.gradient(self.E, te))
            vn = np.interp(tq, te, np.gradient(self.N, te))
            vu = np.interp(tq, te, np.gradient(self.U, te))
            out["gnss_speed_h_ms"] = np.hypot(ve, vn)
            out["gnss_speed_v_ms"] = vu
            out["gnss_course_deg"] = np.degrees(np.arctan2(ve, vn)) % 360
        return out

    def nearest_neighbour(self, t_frame):
        """Raw nearest per-frame record (for comparison only)."""
        tq = np.asarray(t_frame, float) + self.cfg.offset_s
        k = np.clip(np.searchsorted(self.t, tq), 1, len(self.t) - 1)
        k = np.where(np.abs(self.t[k - 1] - tq) <= np.abs(self.t[k] - tq), k - 1, k)
        return k
