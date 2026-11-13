"""
GeoSplat - Phase 4 installer for the official Graphdeco 3D Gaussian Splatting (Windows, local GPU).

What it does (only after you type Y):
  1. git clone --recursive graphdeco-inria/gaussian-splatting -> GeoSplat_App/third_party/gaussian-splatting
  2. Python environment inside GeoSplat_App/.conda_gs (conda, Python 3.10)
  3. PyTorch CUDA build matching the installed CUDA Toolkit (nvcc)
  4. Compile the CUDA submodules (diff-gaussian-rasterization, simple-knn, fused-ssim) with MSVC
  5. Verify imports + GPU, write GeoSplat_Output/logs/phase4_setup.json
It never installs system software (CUDA Toolkit / Visual Studio / git): if one is missing it stops and
prints the command to install it. It never deletes anything.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
RUNTIME = os.path.join(HERE, "runtime")
LOG = os.path.join(RUNTIME, "install_gaussian_splatting.log")
REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting"
REPO = os.path.join(HERE, "third_party", "gaussian-splatting")
ENV = os.path.join(HERE, ".conda_gs")
ENV_PY = os.path.join(ENV, "python.exe")
if not os.path.exists(ENV_PY) and os.path.exists(os.path.join(ENV, "Scripts", "python.exe")):
    ENV_PY = os.path.join(ENV, "Scripts", "python.exe")

# nvcc release -> (torch, torchvision, wheel index tag). Pinned versions that exist on download.pytorch.org.
TORCH_MAP = [((11, 8), (11, 8), "2.4.1", "0.19.1", "cu118"),
             ((12, 1), (12, 3), "2.4.1", "0.19.1", "cu121"),
             ((12, 4), (12, 5), "2.5.1", "0.20.1", "cu124"),
             ((12, 6), (12, 7), "2.7.1", "0.22.1", "cu126"),
             ((12, 8), (12, 9), "2.7.1", "0.22.1", "cu128")]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stream(cmd, shell=False, env=None):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace", env=env)
    with open(LOG, "a", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line)
            s = line.rstrip()
            if s and (len(s) < 200):
                print("   " + s, flush=True)
    p.wait()
    return p.returncode


def main():
    global ENV_PY
    os.makedirs(RUNTIME, exist_ok=True)
    log("=== GeoSplat GPU setup (3D Gaussian Splatting) ===")
    subprocess.run([sys.executable, os.path.join(HERE, "setup", "check_gpu_toolchain.py")], capture_output=True)
    env = json.load(open(os.path.join(RUNTIME, "gpu_toolchain.json")))
    r = env["ready"]
    missing = [k for k, v in r.items() if not v]
    drv = env.get("driver_max_cuda")
    if missing:
        log("Cannot install yet. Missing: " + ", ".join(missing))
        if not r["cuda_toolkit_nvcc"]:
            log(f"  * CUDA Toolkit: your driver supports up to CUDA {drv}. Install CUDA Toolkit 12.4 "
                "(matches PyTorch cu124) from https://developer.nvidia.com/cuda-12-4-0-download-archive "
                "(Windows > x86_64 > 11 > exe local). Do not install a version newer than " + str(drv) + ".")
        if not r["msvc_build_tools"]:
            log('  * Visual Studio 2022 Build Tools (C++):  winget install Microsoft.VisualStudio.2022.BuildTools '
                '--override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"')
        if not r["git"]:
            log("  * Git:  winget install Git.Git")
        if not r["python_env_tool (conda or py launcher)"]:
            log("  * Needs Miniconda/Anaconda OR Python 3.10-3.12 (py launcher):  winget install Python.Python.3.12")
        log("After installing, close this window and click 'Set up GPU…' in GeoSplat again.")
        return 2

    m = re.match(r"(\d+)\.(\d+)", env["nvcc_version"])
    nv = (int(m.group(1)), int(m.group(2)))
    pick = next((x for x in TORCH_MAP if x[0] <= nv <= x[1]), None)
    if pick is None:
        log(f"CUDA Toolkit {env['nvcc_version']} has no tested PyTorch mapping in this script. Stopping.")
        return 3
    torch_v, tv_v, tag = pick[2], pick[3], pick[4]
    vs = next(v for v in env["visual_studio"] if v["vcvars64"])
    vcvars = os.path.join(vs["path"], r"VC\Auxiliary\Build\vcvars64.bat")
    cuda_home = os.path.dirname(os.path.dirname(env["nvcc_path"]))
    vs_minor = tuple(int(x) for x in vs["version"].split(".")[:2])
    # CUDA <= 12.4 officially supports VS 2022 up to 17.9; newer MSVC needs -allow-unsupported-compiler
    allow_unsup = nv <= (12, 4) and vs_minor >= (17, 10)
    if allow_unsup:
        log(f"VS {vs['version']} is newer than officially supported by CUDA {env['nvcc_version']} "
            "-> using -allow-unsupported-compiler")
    log(f"CUDA_HOME pinned to {cuda_home}")
    log(f"nvcc {env['nvcc_version']} | driver max CUDA {drv} | {vs['name']} {vs['version']}")
    log(f"Plan:    Graphdeco repository -> {REPO}")
    log(f"         Python 3.10 environment -> {ENV}")
    log(f"         torch {torch_v} + torchvision {tv_v} ({tag}), then compile 3 CUDA extensions")
    log("Estimate: ~3 GB download, ~7 GB disk, 15-40 minutes.")
    ans = "y" if "--yes" in sys.argv else input("\nContinue with the installation? Type Y and press Enter: ").strip().lower()
    if ans != "y":
        log("Cancelled by the user.")
        return 1

    # 1. repo
    if not os.path.exists(os.path.join(REPO, "train.py")):
        os.makedirs(os.path.dirname(REPO), exist_ok=True)
        if stream(["git", "clone", "--recursive", REPO_URL, REPO]) != 0:
            log("git clone failed.")
            return 4
    else:
        stream(["git", "-C", REPO, "submodule", "update", "--init", "--recursive"])
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"Graphdeco commit: {commit}")

    # 2. env
    if not os.path.exists(ENV_PY):
        if env.get("conda"):
            rc = stream([env["conda"], "create", "-p", ENV, "python=3.10", "-y"])
        else:   # no conda: plain venv from a py-launcher Python 3.10-3.12 (PyTorch wheels exist for these)
            rc = 1
            for ver in ("3.10", "3.11", "3.12"):
                if subprocess.run(["py", f"-{ver}", "--version"], capture_output=True).returncode == 0:
                    rc = stream(["py", f"-{ver}", "-m", "venv", ENV])
                    break
        if not os.path.exists(ENV_PY) and os.path.exists(os.path.join(ENV, "Scripts", "python.exe")):
            ENV_PY = os.path.join(ENV, "Scripts", "python.exe")
        if rc != 0 or not os.path.exists(ENV_PY):
            log("Creating the Python environment failed (needs conda or Python 3.10-3.12).")
            return 5
    pip = [ENV_PY, "-m", "pip"]
    stream(pip + ["install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel"])

    # 3. torch (no pip cache: keeps the ~3 GB download off the nearly-full C: drive)
    rc = stream(pip + ["install", "--no-cache-dir", f"torch=={torch_v}", f"torchvision=={tv_v}",
                       "--index-url", f"https://download.pytorch.org/whl/{tag}"])
    if rc != 0:
        log("PyTorch installation failed.")
        return 6
    stream(pip + ["install", "--no-cache-dir", "plyfile", "tqdm", "opencv-python", "joblib", "pandas",
                  "matplotlib", "lpips", "ninja"])

    # 4. CUDA extensions (MSVC environment)
    cap = re.search(r"(\d+)\.(\d+)", env.get("gpu", "").split(",")[-1] if env.get("gpu") else "")
    arch = f"{cap.group(1)}.{cap.group(2)}" if cap else "8.9"
    built = {}
    # Build through a temporary drive letter without spaces ("ITS RESEARCH CENTER" breaks some nvcc/ninja
    # command lines). subst only maps a letter to the folder; nothing is copied or deleted.
    drv = None
    for letter in "WVUTSRQP":
        if not os.path.exists(letter + ":\\"):
            if subprocess.run(["subst", letter + ":", HERE], capture_output=True).returncode == 0:
                drv = letter + ":"
                break
    if drv:
        log(f"App folder temporarily mapped to {drv}\\ (no spaces) for compiling")
        b_env, b_py, b_repo = drv + "\\.conda_gs", drv + "\\.conda_gs\\python.exe", drv + "\\third_party\\gaussian-splatting"
    else:
        log("subst not available - compiling from the original path")
        b_env, b_py, b_repo = ENV, ENV_PY, REPO
    try:
        rc_all = build_all(vcvars, cuda_home, allow_unsup, arch, b_env, b_py, b_repo, built)
    finally:
        if drv:
            subprocess.run(["subst", drv, "/D"], capture_output=True)
            log(f"Mapping {drv} released")
    if rc_all != 0:
        return rc_all
    return verify(env, commit, torch_v, tag, arch, built)


IMPORT_NAME = {"diff-gaussian-rasterization": "diff_gaussian_rasterization", "simple-knn": "simple_knn._C",
               "fused-ssim": "fused_ssim"}


def build_all(vcvars, cuda_home, allow_unsup, arch, ENV, ENV_PY, REPO, built):
    for sub in ("diff-gaussian-rasterization", "simple-knn", "fused-ssim"):
        path = os.path.join(REPO, "submodules", sub)
        if not os.path.isdir(path):
            log(f"  (submodule {sub} not in this commit - skipped)")
            continue
        if subprocess.run([ENV_PY, "-c", f"import torch, {IMPORT_NAME[sub]}"], capture_output=True).returncode == 0:
            log(f"  {sub}: already compiled - skipped")
            built[sub] = True
            continue
        log(f"  Compiling {sub} ...")
        nvflags = "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
        flags = f'set "NVCC_APPEND_FLAGS={nvflags}" && ' if allow_unsup else ""
        # cmd.exe pitfalls handled here:
        #  * %VAR% is expanded when the whole line is parsed (BEFORE vcvars64 runs) -> use delayed
        #    expansion (cmd /V:ON, !PATH!) so the MSVC paths added by vcvars64 survive.
        #  * `set X=v && ...` stores "v " WITH the trailing space (gave TORCH_CUDA_ARCH_LIST="8.9 " ->
        #    "Unknown CUDA arch ()") -> every assignment is quoted: set "X=v".
        line = (f'call "{vcvars}" >nul && set "DISTUTILS_USE_SDK=1" && set "CUDA_HOME={cuda_home}" && '
                f'set "CUDA_PATH={cuda_home}" && set "PATH={cuda_home}\\bin;{ENV}\\Scripts;!PATH!" && {flags}'
                f'set "TORCH_CUDA_ARCH_LIST={arch}" && echo ARCH=[!TORCH_CUDA_ARCH_LIST!] && '
                f'where cl && where nvcc && where ninja & '
                f'"{ENV_PY}" -m pip install --no-cache-dir --no-build-isolation "{path}"')
        # passed as one raw string: /S makes cmd strip only the outer quotes and keep the inner ones
        rc = stream('cmd /V:ON /S /C "' + line + '"')
        if rc != 0 and not allow_unsup:
            log(f"  Compiling {sub} failed, retrying with -allow-unsupported-compiler ...")
            line2 = line.replace('set "TORCH_CUDA_ARCH_LIST',
                                 f'set "NVCC_APPEND_FLAGS={nvflags}" && set "TORCH_CUDA_ARCH_LIST')
            rc = stream('cmd /V:ON /S /C "' + line2 + '"')
        built[sub] = rc == 0
        if rc != 0:
            log(f"FAILED to compile {sub}. See {LOG}.")
            return 7
    return 0


def verify(env, commit, torch_v, tag, arch, built):
    test = ("import torch, json; import diff_gaussian_rasterization, simple_knn; "
            "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'gpu': torch.cuda.get_device_name(0), 'cuda_ok': torch.cuda.is_available()}))")
    vr = subprocess.run([ENV_PY, "-c", test], capture_output=True, text=True)
    log(vr.stdout.strip() or vr.stderr.strip()[-800:])
    ok = vr.returncode == 0 and '"cuda_ok": true' in vr.stdout
    json.dump({"ok": ok, "python": ENV_PY, "repo": REPO, "commit": commit, "torch": torch_v, "index": tag,
               "nvcc": env["nvcc_version"], "arch": arch, "built": built,
               "verify": vr.stdout.strip()}, open(os.path.join(RUNTIME, "gs_env.json"), "w"), indent=2)
    log("SETUP COMPLETE - Gaussian Splatting is ready for GeoSplat" if ok else f"Verification FAILED - see {LOG}")
    return 0 if ok else 8


if __name__ == "__main__":
    sys.exit(main())
