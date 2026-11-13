"""
GeoSplat pipeline orchestrator (used by the GUI and the CLI).

Runs the five stages as separate processes (isolation, cancellable, resumable):

  1 telemetry  scripts/stage1_telemetry.py   probe video, decode SRT / embedded DJI track, extract frames, sync
  2 colmap     scripts/stage2_colmap.py      GNSS-guided matching + incremental SfM (COLMAP)
  3 georef     scripts/stage3_georef.py      robust Sim3 to GNSS, check points, CRS chain, ENU model
  4 gaussians  scripts/stage4_gaussians.py   Graphdeco 3D Gaussian Splatting + georeferenced exports
  5 web        scripts/stage5_web.py         satellite viewer assets (index.html, splat, orthophoto)
  6 vcorr      scripts/stage6_vcorr.py       height-datum check; with RTK checkpoints: vertical correction (optional)

A stage that finished successfully writes <project>/logs/_stage_<key>.done ; re-running the same project
skips finished stages (resume). Nothing is ever deleted.
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
SETTINGS = os.path.join(RUNTIME, "settings.json")
VIEWER_PORT = 8765

STAGES = [("telemetry", "Telemetri & ekstraksi frame"), ("colmap", "COLMAP Structure-from-Motion"),
          ("georef", "Georeferensi GNSS"), ("gaussians", "Gaussian Splatting"), ("web", "Viewer satelit & ekspor")]


# ----------------------------------------------------------------------------- settings / tools
def load_settings() -> dict:
    try:
        return json.load(open(SETTINGS, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_settings(d: dict):
    os.makedirs(RUNTIME, exist_ok=True)
    json.dump(d, open(SETTINGS, "w", encoding="utf-8"), indent=2)


def tool_paths() -> dict:
    """Locations of ffmpeg / colmap: runtime/tools.json (portable download) first, then PATH."""
    t = {}
    try:
        t = json.load(open(os.path.join(RUNTIME, "tools.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    for name in ("ffmpeg", "ffprobe", "colmap"):
        if not t.get(name) or not os.path.exists(t[name]):
            t[name] = shutil.which(name)
    return t


def child_env(gmaps_key: str = "") -> dict:
    env = os.environ.copy()
    t = tool_paths()
    extra = []
    for name in ("ffmpeg", "colmap"):
        if t.get(name):
            extra.append(os.path.dirname(t[name]))
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if gmaps_key:
        env["GEOSPLAT_GMAPS_KEY"] = gmaps_key
    return env


def gs_env() -> dict | None:
    try:
        d = json.load(open(os.path.join(RUNTIME, "gs_env.json"), encoding="utf-8"))
        return d if d.get("ok") and os.path.exists(d.get("python", "")) else None
    except Exception:  # noqa: BLE001
        return None


def check_prerequisites() -> dict:
    t = tool_paths()
    g = gs_env()
    smi = shutil.which("nvidia-smi")
    gpu = None
    if smi:
        try:
            gpu = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=15, creationflags=_NO_WINDOW).stdout.strip() or None
        except Exception:  # noqa: BLE001
            gpu = None
    return {"ffmpeg": t.get("ffmpeg"), "ffprobe": t.get("ffprobe"), "colmap": t.get("colmap"),
            "gpu": gpu, "gaussian_splatting": g["torch"] + " / " + g.get("commit", "")[:8] if g else None}


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_srt(video: str) -> str | None:
    stem = os.path.splitext(video)[0]
    for ext in (".SRT", ".srt", ".Srt"):
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


def default_project(video: str) -> str:
    return os.path.splitext(os.path.abspath(video))[0] + "_GeoSplat"


# ----------------------------------------------------------------------------- pipeline
@dataclass
class Job:
    video: str
    srt: str | None = None
    project: str | None = None
    fps: float = 2.0
    profile: str = "balanced"
    iterations: int = 30000
    gmaps_key: str = ""
    redo: bool = False
    stages: list = field(default_factory=lambda: [k for k, _ in STAGES])
    checkpoints: str | None = None          # optional RTK checkpoint CSV for stage 6

    def __post_init__(self):
        self.video = os.path.abspath(self.video)
        if self.srt is None:
            self.srt = find_srt(self.video)
        self.project = os.path.abspath(self.project or default_project(self.video))
        if self.checkpoints:
            self.checkpoints = os.path.abspath(self.checkpoints)


class Pipeline:
    def __init__(self, job: Job, on_log=print, on_stage=None):
        self.job = job
        self.on_log = on_log
        self.on_stage = on_stage or (lambda key, status, info="": None)
        self.proc = None
        self._stop = False

    # commands -------------------------------------------------------------
    def command(self, key: str):
        j, py = self.job, sys.executable
        if os.path.basename(py).lower() == "pythonw.exe":                 # GUI runs under pythonw
            py = os.path.join(os.path.dirname(py), "python.exe")
        s = lambda n: os.path.join(ROOT, "scripts", n)                     # noqa: E731
        if key == "telemetry":
            c = [py, s("stage1_telemetry.py"), "--video", j.video, "--project", j.project, "--fps", f"{j.fps:g}"]
            return c + (["--srt", j.srt] if j.srt else [])
        if key == "colmap":
            return [py, s("stage2_colmap.py"), "--project", j.project]
        if key == "georef":
            return [py, s("stage3_georef.py"), "--project", j.project]
        if key == "gaussians":
            g = gs_env()
            if not g:
                raise RuntimeError("Lingkungan Gaussian Splatting belum terpasang. Klik 'Siapkan GPU' dulu.")
            return [g["python"], s("stage4_gaussians.py"), "--project", j.project, "--profile", j.profile,
                    "--iterations", str(j.iterations), "--gs-env", os.path.join(RUNTIME, "gs_env.json")]
        if key == "web":
            return [py, s("stage5_web.py"), "--project", j.project]
        if key == "vcorr":
            return [py, s("stage6_vcorr.py"), "--project", j.project] + (
                ["--checkpoints", j.checkpoints] if j.checkpoints else [])
        raise KeyError(key)

    def marker(self, key):
        return os.path.join(self.job.project, "logs", f"_stage_{key}.done")

    # run ------------------------------------------------------------------
    def stop(self):
        self._stop = True
        p = self.proc
        if p and p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=_NO_WINDOW)
            else:
                p.terminate()

    def run(self) -> bool:
        j = self.job
        os.makedirs(os.path.join(j.project, "logs"), exist_ok=True)
        json.dump({"video": j.video, "srt": j.srt, "fps": j.fps, "profile": j.profile, "iterations": j.iterations,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(j.project, "logs", "job.json"), "w"), indent=2)
        env = child_env(j.gmaps_key)
        for key, label in STAGES:
            if key not in j.stages:
                continue
            if self._stop:
                self.on_stage(key, "cancelled")
                return False
            if os.path.exists(self.marker(key)) and not j.redo and not (key == "vcorr" and j.checkpoints):
                self.on_stage(key, "skipped", "sudah selesai sebelumnya")
                continue
            try:
                cmd = self.command(key)
            except Exception as e:  # noqa: BLE001
                self.on_stage(key, "failed", str(e))
                self.on_log(f"[{key}] {e}")
                return False
            self.on_stage(key, "running")
            self.on_log(f"=== {label} ===")
            t0 = time.time()
            log_path = os.path.join(j.project, "logs", f"stage_{key}.log")
            with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(f"\n$ {' '.join(cmd)}\n")
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                             encoding="utf-8", errors="replace", env=env, cwd=ROOT,
                                             creationflags=_NO_WINDOW, bufsize=1)
                for line in self.proc.stdout:
                    lf.write(line)
                    self.on_log(line.rstrip())
                rc = self.proc.wait()
            dt = time.time() - t0
            if self._stop:
                self.on_stage(key, "cancelled")
                return False
            if rc != 0:
                self.on_stage(key, "failed", f"exit {rc} - lihat {log_path}")
                return False
            open(self.marker(key), "w").write(json.dumps({"seconds": round(dt, 1), "finished": time.ctime()}))
            self.on_stage(key, "done", f"{dt / 60:.1f} menit")
        return True

    def viewer_dir(self):
        return os.path.join(self.job.project, "08_Exports", "web")


# ----------------------------------------------------------------------------- local viewer server
_server = None


def serve_viewer(web_dir: str, port: int = VIEWER_PORT) -> str:
    """Serve web_dir on http://127.0.0.1:<port>/ (loopback only). Returns the URL."""
    global _server
    if _server:
        _server.shutdown()
        _server.server_close()
        _server = None

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = partial(Quiet, directory=web_dir)
    socketserver.TCPServer.allow_reuse_address = True
    _server = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=_server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html"
