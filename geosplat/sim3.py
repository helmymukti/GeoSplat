"""
3D similarity transformation (7-parameter Helmert / Sim3):   y = s * R @ x + t

* umeyama()      closed-form least squares (Umeyama 1991), reflection-safe
* ransac_sim3()  robust estimate: minimal 3-point RANSAC + least-squares refit on inliers
* apply_sim3(), residuals()
References: Umeyama (1991) IEEE TPAMI 13(4) 376-380 (closed-form similarity);
Fischler & Bolles (1981) CACM 24(6) 381-395 (RANSAC). See docs/METHODS.md section 6.1.
"""
from __future__ import annotations

import numpy as np


def umeyama(src, dst, with_scale=True):
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / n
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s, R, t, x):
    return s * (np.asarray(x, float) @ R.T) + t


def residuals(s, R, t, src, dst):
    return np.linalg.norm(apply_sim3(s, R, t, src) - dst, axis=1)


def ransac_sim3(src, dst, thresh_m=3.0, iters=2000, seed=0, min_spread_m=5.0):
    """Robust Sim3. thresh_m = inlier threshold in destination units (metres)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        d = dst[idx]
        # reject near-collinear / tiny samples
        if np.linalg.norm(np.cross(d[1] - d[0], d[2] - d[0])) < min_spread_m ** 2:
            continue
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inl = residuals(s, R, t, src, dst) < thresh_m
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        best = np.ones(n, bool)
    # refit + one re-classification round
    for _ in range(3):
        s, R, t = umeyama(src[best], dst[best])
        new = residuals(s, R, t, src, dst) < thresh_m
        if new.sum() < 3 or np.array_equal(new, best):
            break
        best = new
    r = residuals(s, R, t, src, dst)
    return {"scale": s, "R": R, "t": t, "inliers": best, "residuals": r,
            "rmse_inliers": float(np.sqrt(np.mean(r[best] ** 2))),
            "rmse_all": float(np.sqrt(np.mean(r ** 2))),
            "median": float(np.median(r)), "max": float(r.max()),
            "n_inliers": int(best.sum()), "n": int(n)}


def self_test():
    rng = np.random.default_rng(1)
    x = rng.uniform(-50, 50, (200, 3))
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    s, t = 3.7, np.array([10.0, -5.0, 2.0])
    y = apply_sim3(s, R, t, x) + rng.normal(0, 0.5, x.shape)
    y[:15] += 40.0                              # 7.5 % gross outliers
    out = ransac_sim3(x, y, thresh_m=2.0)
    return {"scale_err": abs(out["scale"] - s), "rot_err_deg": float(np.degrees(np.arccos(
        np.clip((np.trace(out["R"].T @ R) - 1) / 2, -1, 1)))), "outliers_found": int((~out["inliers"]).sum()),
        "rmse_inliers": out["rmse_inliers"]}
