"""Read-only environment check for Gaussian Splatting on Windows. Installs nothing.
Writes GeoSplat_Output/01_Metadata/phase4_env.json"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
OUT = os.path.join(HERE, "runtime", "gpu_toolchain.json")


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


env = {}
smi = run(["nvidia-smi"])
m = re.search(r"Driver Version:\s*([\d.]+).*?CUDA Version:\s*([\d.]+)", smi, re.S)
env["driver"] = m.group(1) if m else None
env["driver_max_cuda"] = m.group(2) if m else None
env["gpu"] = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader"])

nvcc = shutil.which("nvcc")
env["nvcc_path"] = nvcc
env["nvcc_version"] = None
if nvcc:
    m = re.search(r"release ([\d.]+)", run([nvcc, "--version"]))
    env["nvcc_version"] = m.group(1) if m else None
env["CUDA_PATH"] = os.environ.get("CUDA_PATH")
env["cuda_toolkits_installed"] = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"))

vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
env["visual_studio"] = []
if os.path.exists(vswhere):
    try:
        vs = json.loads(run([vswhere, "-products", "*", "-requires",
                             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json"]) or "[]")
        for v in vs:
            p = v.get("installationPath")
            env["visual_studio"].append({"name": v.get("displayName"), "version": v.get("installationVersion"),
                                         "path": p,
                                         "vcvars64": os.path.exists(os.path.join(p, r"VC\Auxiliary\Build\vcvars64.bat"))})
    except Exception as e:  # noqa: BLE001
        env["visual_studio_error"] = str(e)
env["cl_on_path"] = shutil.which("cl")

conda = shutil.which("conda")
for c in [os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"), os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
          r"C:\ProgramData\anaconda3\Scripts\conda.exe", r"C:\ProgramData\miniconda3\Scripts\conda.exe"]:
    if not conda and os.path.exists(c):
        conda = c
env["conda"] = conda
env["conda_version"] = run([conda, "--version"]) if conda else None
env["conda_envs"] = run([conda, "env", "list"]) if conda else None
env["git"] = shutil.which("git")
env["git_version"] = run(["git", "--version"]) if env["git"] else None
env["winget"] = shutil.which("winget")
env["colmap"] = shutil.which("colmap")
env["venv_python"] = sys.executable
env["gs_repo_present"] = os.path.exists(os.path.join(HERE, "third_party", "gaussian-splatting", "train.py"))
env["py_launcher"] = shutil.which("py")
for d in ("C:\\", os.path.splitdrive(HERE)[0] + "\\"):
    try:
        env[f"disk_free_gb_{d[0]}"] = round(shutil.disk_usage(d).free / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass

ready = {
    "nvidia_driver": bool(env["driver"]),
    "cuda_toolkit_nvcc": bool(env["nvcc_version"]),
    "msvc_build_tools": any(v["vcvars64"] for v in env["visual_studio"]),
    "python_env_tool (conda or py launcher)": bool(conda or env["py_launcher"]), "git": bool(env["git"]),
}
env["ready"] = ready
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(env, open(OUT, "w"), indent=2)
print(json.dumps(env, indent=2))
print("\nRINGKASAN:")
for k, v in ready.items():
    print(f"  {'[OK]     ' if v else '[MISSING]'} {k}")
print(f"\nDisimpan ke {OUT}")
