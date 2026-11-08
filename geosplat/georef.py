"""
Georeferencing utilities: model transformation, CRS chain, PLY export.

Coordinate chain (all WGS84):
    COLMAP local (arbitrary scale/orientation)
      --Sim3 (s, R, t)-->            local ENU  [m], origin (lat0, lon0, h0)
      --rigid (R_enu->ecef, X0)-->   ECEF      [m]
      --geodetic inversion-->        WGS84 lat, lon, h
      --Transverse Mercator-->       UTM zone E, N [m] + h

Composite local->ECEF is still a Sim3:  s,  R_ecef = M R,  t_ecef = M t + X0,
where M = columns (E, N, U) of the tangent frame expressed in ECEF.
References: Bar-Itzhack (2000) J. Guidance Control Dyn. 23(6) (quaternion from rotation matrix);
WGS 84 / ENU / ECEF chain per docs/METHODS.md section 6.4.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from . import geodesy as g
from .colmap_io import qvec2rotmat


def rotmat2qvec(R):
    """Robust rotation matrix -> quaternion (w, x, y, z), w >= 0 (Shepperd's method)."""
    R = np.asarray(R, float)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[2, 1] - R[1, 2]],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[0, 2] - R[2, 0]],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[1, 0] - R[0, 1]],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def enu_to_ecef_matrix(lat0, lon0):
    return g._enu_matrix(lat0, lon0).T          # columns = E, N, U unit vectors in ECEF


def compose_local_to_ecef(s, R, t, lat0, lon0, h0):
    M = enu_to_ecef_matrix(lat0, lon0)
    X0 = np.array(g.geodetic_to_ecef(lat0, lon0, h0), float).ravel()
    return s, M @ R, M @ t + X0


# ----------------------------------------------------------------------------- COLMAP TXT transform
def transform_colmap_txt(src_dir, dst_dir, s, R, t):
    """Apply X' = s R X + t to a COLMAP TXT model. Cameras unchanged.
    Pose (world->cam) R_wc, t_wc:  C' = s R C + t,  R_wc' = R_wc R^T,  t_wc' = -R_wc' C'.
    Reprojection of every observation is invariant (verified by caller)."""
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(src_dir, "cameras.txt"), encoding="utf-8") as fi, \
            open(os.path.join(dst_dir, "cameras.txt"), "w", encoding="utf-8") as fo:
        fo.write(fi.read())
    with open(os.path.join(src_dir, "images.txt"), encoding="utf-8") as fi, \
            open(os.path.join(dst_dir, "images.txt"), "w", encoding="utf-8") as fo:
        expect_pose = True
        for line in fi:
            if line.startswith("#"):
                fo.write(line)
                continue
            if expect_pose:
                p = line.split()
                q = np.array([float(v) for v in p[1:5]])
                tv = np.array([float(v) for v in p[5:8]])
                Rwc = qvec2rotmat(q)
                C = -Rwc.T @ tv
                C2 = s * R @ C + t
                Rwc2 = Rwc @ R.T
                t2 = -Rwc2 @ C2
                q2 = rotmat2qvec(Rwc2)
                fo.write(f"{p[0]} {q2[0]:.17g} {q2[1]:.17g} {q2[2]:.17g} {q2[3]:.17g} "
                         f"{t2[0]:.17g} {t2[1]:.17g} {t2[2]:.17g} {' '.join(p[8:])}\n")
            else:
                fo.write(line)
            expect_pose = not expect_pose
    with open(os.path.join(src_dir, "points3D.txt"), encoding="utf-8") as fi, \
            open(os.path.join(dst_dir, "points3D.txt"), "w", encoding="utf-8") as fo:
        for line in fi:
            if line.startswith("#") or not line.strip():
                fo.write(line)
                continue
            p = line.split(" ", 4)
            X = np.array([float(p[1]), float(p[2]), float(p[3])])
            X2 = s * R @ X + t                      # RGB, reprojection error (px) and track unchanged
            fo.write(f"{p[0]} {X2[0]:.10f} {X2[1]:.10f} {X2[2]:.10f} {p[4]}")


# ----------------------------------------------------------------------------- PLY
def write_ply(path, xyz, rgb=None, double=True, comments=()):
    xyz = np.asarray(xyz, np.float64 if double else np.float32)
    n = len(xyz)
    t = "double" if double else "float"
    hdr = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments] + \
          [f"element vertex {n}", f"property {t} x", f"property {t} y", f"property {t} z"]
    dt = [("x", xyz.dtype), ("y", xyz.dtype), ("z", xyz.dtype)]
    if rgb is not None:
        hdr += ["property uchar red", "property uchar green", "property uchar blue"]
        dt += [("r", np.uint8), ("g", np.uint8), ("b", np.uint8)]
    hdr.append("end_header")
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(arr.tobytes())


# ----------------------------------------------------------------------------- UTM linearisation
def enu_to_utm_exact(enu, lat0, lon0, h0, zone, south):
    lat, lon, h = g.enu_to_geodetic(enu[:, 0], enu[:, 1], enu[:, 2], lat0, lon0, h0)
    E, N, _, _ = g.latlon_to_utm(lat, lon, zone, south)
    return np.column_stack([E, N, h]), lat, lon


def fit_enu_to_utm_similarity(enu_samples, lat0, lon0, h0, zone, south):
    """Best 3D similarity ENU -> (UTM E, N, h) over the scene, with its max misfit.
    Needed because Gaussians (positions + covariances) can only be mapped consistently by a
    similarity; UTM itself is conformal but not Euclidean (scale k and grid convergence)."""
    from .sim3 import umeyama, residuals
    utm, _, _ = enu_to_utm_exact(enu_samples, lat0, lon0, h0, zone, south)
    off = np.floor(utm.mean(0) / 1000.0) * 1000.0
    off[2] = 0.0
    s, R, t = umeyama(enu_samples, utm - off)
    r = residuals(s, R, t, enu_samples, utm - off)
    gamma = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))   # rotation about Up (grid convergence)
    return {"scale": s, "R": R, "t": t, "offset": off, "max_err_m": float(r.max()),
            "rms_err_m": float(np.sqrt(np.mean(r ** 2))), "grid_convergence_deg": gamma}
