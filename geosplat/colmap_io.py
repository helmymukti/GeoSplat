"""
COLMAP command-line wrapper (version-aware) and text-model reader.

* Locates COLMAP (PATH or C:\\colmap-x64-windows-cuda). If the Windows release
  wrapper colmap.bat is found, bin\\colmap.exe is called directly with the same
  environment the .bat sets (lib on PATH, Qt plugins) - avoids cmd.exe quoting
  problems with paths that contain spaces ("ITS RESEARCH CENTER").
* Option names changed between COLMAP releases (e.g. SiftExtraction.use_gpu ->
  FeatureExtraction.use_gpu). Each command's -h output is parsed and the first
  supported spelling is used.
COLMAP: Schoenberger & Frahm (2016) CVPR; Schoenberger et al. (2016) ECCV. See docs/METHODS.md section 5.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import numpy as np

DEFAULT_DIRS = [r"C:\colmap-x64-windows-cuda", r"C:\colmap", r"C:\Program Files\COLMAP"]


class Colmap:
    def __init__(self, exe=None, log=None):
        self.log = log
        self.env = os.environ.copy()
        cand = exe or shutil.which("colmap")
        if not cand:
            for d in DEFAULT_DIRS:
                for n in ("COLMAP.bat", "colmap.bat", "colmap.exe", os.path.join("bin", "colmap.exe")):
                    if os.path.exists(os.path.join(d, n)):
                        cand = os.path.join(d, n)
                        break
                if cand:
                    break
        if not cand:
            raise FileNotFoundError("COLMAP not found on PATH or in " + ", ".join(DEFAULT_DIRS))
        root = os.path.dirname(cand)
        exe_direct = os.path.join(root, "bin", "colmap.exe")
        if cand.lower().endswith(".bat") and os.path.exists(exe_direct):
            self.env["PATH"] = os.path.join(root, "lib") + os.pathsep + self.env.get("PATH", "")
            if os.path.isdir(os.path.join(root, "plugins")):
                self.env["QT_PLUGIN_PATH"] = os.path.join(root, "plugins")
            self.cmd = [exe_direct]
        else:
            self.cmd = [cand]
        self._help = {}
        self.version = self._version()

    def _run(self, args, **kw):
        return subprocess.run(self.cmd + args, capture_output=True, text=True, env=self.env,
                              errors="replace", **kw)

    def _version(self):
        r = self._run(["help"])
        m = re.search(r"COLMAP\s+([0-9][0-9.]*[^\s]*)", r.stdout + r.stderr)
        cuda = "with CUDA" in (r.stdout + r.stderr) or "CUDA" in (r.stdout + r.stderr)
        return {"version": m.group(1) if m else "unknown", "cuda": cuda,
                "banner": (r.stdout + r.stderr).strip().splitlines()[:2]}

    def help(self, command):
        if command not in self._help:
            r = self._run([command, "-h"])
            self._help[command] = r.stdout + r.stderr
        return self._help[command]

    def opt(self, command, *names):
        """First option spelling supported by this COLMAP build (or None)."""
        h = self.help(command)
        for n in names:
            if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w.])", h):
                return n
        return None

    def args(self, command, pairs):
        """pairs: list of ((name1, name2, ...), value). Unsupported options are skipped + logged."""
        out = []
        for names, val in pairs:
            names = (names,) if isinstance(names, str) else names
            o = self.opt(command, *names)
            if o is None:
                if self.log:
                    self.log.warning("COLMAP %s: option %s not supported by this build - skipped",
                                     command, "/".join(names))
                continue
            out += [o, str(val)]
        return out

    def run(self, command, args, log_path, echo_every_s=20):
        """Run a COLMAP command, streaming output to log_path; raises on failure."""
        t0 = time.time()
        full = self.cmd + [command] + args
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(" ".join(f'"{a}"' if " " in a else a for a in full) + "\n\n")
            fh.flush()
            p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=self.env, errors="replace", bufsize=1)
            last = t0
            for line in p.stdout:
                fh.write(line)
                if time.time() - last > echo_every_s:
                    fh.flush()
                    if self.log:
                        self.log.info("  [%s %.0fs] %s", command, time.time() - t0, line.strip()[:150])
                    last = time.time()
            p.wait()
        dt = time.time() - t0
        if p.returncode != 0:
            raise RuntimeError(f"COLMAP {command} failed (exit {p.returncode}); see {log_path}")
        return dt


# ----------------------------------------------------------------------------- model I/O (TXT)
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_cameras_txt(path):
    cams = {}
    for l in open(path, encoding="utf-8"):
        if l.startswith("#") or not l.strip():
            continue
        p = l.split()
        cams[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                           "params": [float(v) for v in p[4:]]}
    return cams


def read_images_txt(path):
    imgs = {}
    lines = [l for l in open(path, encoding="utf-8") if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        pts = lines[i + 1].split()
        n_obs = len(pts) // 3
        n_tri = sum(1 for k in range(n_obs) if pts[3 * k + 2] != "-1")
        R = qvec2rotmat(q)
        imgs[int(p[0])] = {"name": " ".join(p[9:]), "camera_id": int(p[8]), "qvec": q, "tvec": t,
                           "R": R, "center": -R.T @ t, "n_obs": n_obs, "n_tri": n_tri}
    return imgs


def read_points3d_txt(path, max_points=None):
    xyz, rgb, err, tl = [], [], [], []
    for l in open(path, encoding="utf-8"):
        if l.startswith("#") or not l.strip():
            continue
        p = l.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
        err.append(float(p[7]))
        tl.append((len(p) - 8) // 2)
    return np.array(xyz), np.array(rgb, np.uint8), np.array(err), np.array(tl)
