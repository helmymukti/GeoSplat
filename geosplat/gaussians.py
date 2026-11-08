"""
3D Gaussian Splatting PLY I/O and rigorous similarity transformation.

Graphdeco PLY layout (all float32, binary little endian):
  x y z | nx ny nz | f_dc_0..2 | f_rest_0..(3*K-1) | opacity | scale_0..2 | rot_0..3
  * f_rest is channel-major: f_rest[c*K + k] = coefficient k+1 (k = 0..K-1) of colour channel c
  * scale_* are log-scales, rot_* is an (unnormalised) quaternion (w, x, y, z)
  * opacity is a logit

Similarity  X' = s R X + t  applied to a Gaussian:
  mean      mu'   = s R mu + t
  covariance Sigma' = s^2 R Sigma R^T   ->  rotation q' = q_R (x) q ,  log-scale' = log-scale + ln s
  SH colour  c'(d') = c(R^T d')  for every viewing direction d'  -> per-band rotation of the SH
             coefficients, solved exactly by least squares on the band-limited basis.
References: Kerbl et al. (2023) ACM TOG 42(4) - 3D Gaussian Splatting (PLY layout, SH convention);
Ivanic & Ruedenberg (1996) J. Phys. Chem. 100(15); Green (2003) GDC - real SH rotation.
See docs/METHODS.md section 7.1.
"""
from __future__ import annotations

import numpy as np

# ---- Graphdeco SH basis (utils/sh_utils.py eval_sh), same signs/constants
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_basis(d, deg=3):
    """Real SH basis values (M, (deg+1)^2) for unit directions d (M,3), Graphdeco convention."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    out = [np.full_like(x, C0)]
    if deg >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if deg >= 3:
        out += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(out, axis=1)


def sh_rotation_matrices(R, deg=3, n_dirs=512, seed=0):
    """Block matrices D_l such that new_coeffs_l = D_l @ old_coeffs_l represent c'(d) = c(R^T d)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_dirs, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Y_new = sh_basis(d, deg)                 # basis at d'
    Y_old = sh_basis(d @ R, deg)             # basis at R^T d'   (row-vector form: (R^T d)^T = d^T R)
    Ds = []
    for l in range(1, deg + 1):
        sl = slice(l * l, (l + 1) * (l + 1))
        # Solve Y(d') X = Y(R^T d')  =>  Y(d') (X c) = Y(R^T d') c  =>  c' = X c
        X, *_ = np.linalg.lstsq(Y_new[:, sl], Y_old[:, sl], rcond=None)
        Ds.append(X)
    return Ds


# ---- quaternions (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_to_rotmat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotmat_to_quat(R):
    from .georef import rotmat2qvec
    return rotmat2qvec(R)


# ---- PLY I/O
def read_gs_ply(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            hdr.append(line)
            if line == "end_header":
                break
        n = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
        props = [l.split() for l in hdr if l.startswith("property")]
        tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
        dt = np.dtype([(p[2], tmap[p[1]]) for p in props])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    comments = [l[8:] for l in hdr if l.startswith("comment ")]
    return data, comments


def write_gs_ply(path, data, comments=()):
    hdr = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments] + \
          [f"element vertex {len(data)}"]
    names = {"<f4": "float", "<f8": "double", "|u1": "uchar", "<i4": "int"}
    for name in data.dtype.names:
        hdr.append(f"property {names[data.dtype[name].str]} {name}")
    hdr.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def sh_degree_of(data):
    k = len([n for n in data.dtype.names if n.startswith("f_rest_")]) // 3
    return int(round(np.sqrt(k + 1))) - 1


def transform_gaussians(data, s, R, t, offset=None):
    """Return a new structured array with the similarity applied (float32 output).
    offset (3,) is subtracted after the transform (e.g. a UTM false origin) to keep float32 precision."""
    out = data.copy()
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    xyz2 = s * xyz @ R.T + t
    if offset is not None:
        xyz2 = xyz2 - np.asarray(offset, float)
    out["x"], out["y"], out["z"] = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    for n in ("nx", "ny", "nz"):
        if n in data.dtype.names:
            out[n] = 0.0
    q = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qR = rotmat_to_quat(R)
    q2 = quat_mul(np.broadcast_to(qR, q.shape), q)
    for i in range(4):
        out[f"rot_{i}"] = q2[:, i]
    ls = np.log(s)
    for i in range(3):
        out[f"scale_{i}"] = data[f"scale_{i}"].astype(np.float64) + ls
    deg = sh_degree_of(data)
    if deg > 0:
        K = (deg + 1) ** 2 - 1
        Ds = sh_rotation_matrices(R, deg)
        for c in range(3):
            coeff = np.column_stack([data[f"f_rest_{c * K + k}"] for k in range(K)]).astype(np.float64)
            new = np.empty_like(coeff)
            for l, D in enumerate(Ds, start=1):
                a, b = l * l - 1, (l + 1) * (l + 1) - 1
                new[:, a:b] = coeff[:, a:b] @ D.T
            for k in range(K):
                out[f"f_rest_{c * K + k}"] = new[:, k]
    return out


def gaussian_colour(data, idx, d):
    """SH colour (before +0.5 / clamp) of Gaussian idx for unit directions d (M,3)."""
    deg = sh_degree_of(data)
    K = (deg + 1) ** 2 - 1
    Y = sh_basis(d, deg)
    col = []
    for c in range(3):
        co = np.r_[data[f"f_dc_{c}"][idx], [data[f"f_rest_{c * K + k}"][idx] for k in range(K)]]
        col.append(Y @ co)
    return np.stack(col, 1)


def self_test():
    """Synthetic check: covariance and view-dependent colour must transform exactly."""
    rng = np.random.default_rng(0)
    n, deg = 50, 3
    K = (deg + 1) ** 2 - 1
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + \
            [f"f_rest_{i}" for i in range(3 * K)] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                     "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(nm, "<f4") for nm in names])
    for nm in names:
        data[nm] = rng.normal(size=n)
    a = 2.4
    R = quat_to_rotmat(np.array([np.cos(a / 2), *(np.sin(a / 2) * np.array([0.3, -0.5, 0.81]) /
                                                     np.linalg.norm([0.3, -0.5, 0.81]))]))
    s, t = 36.18, np.array([-74.7, 56.4, 0.6])
    out = transform_gaussians(data, s, R, t)
    # covariance
    def cov(dd, i):
        Rq = quat_to_rotmat(np.array([dd[f"rot_{k}"][i] for k in range(4)], float))
        S = np.diag(np.exp([dd[f"scale_{k}"][i] for k in range(3)]).astype(float))
        return Rq @ S @ S @ Rq.T
    ce = max(np.abs(cov(out, i) - s * s * R @ cov(data, i) @ R.T).max() / (s * s * np.abs(cov(data, i)).max())
             for i in range(n))
    # colour: c'(R d) == c(d)
    d = rng.normal(size=(100, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    colerr = max(np.abs(gaussian_colour(out, i, d @ R.T) - gaussian_colour(data, i, d)).max() for i in range(n))
    # inverse round trip
    back = transform_gaussians(out, 1 / s, R.T, -(R.T @ t) / s)
    rt = max(np.abs(back[nm].astype(float) - data[nm].astype(float)).max() for nm in names if not nm.startswith("n")
             and not nm.startswith("rot"))
    return {"cov_rel_err": float(ce), "sh_colour_err": float(colerr), "roundtrip_err": float(rt)}
