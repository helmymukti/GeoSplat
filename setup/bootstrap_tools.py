"""
Portable tool bootstrap: makes FFmpeg and COLMAP available to GeoSplat without a system install.

* Uses tools already on PATH if present (nothing is downloaded then).
* Otherwise downloads official Windows builds into <app>/tools/ and records them in runtime/tools.json:
    - FFmpeg  : gyan.dev release "essentials" build  (fallback: BtbN GitHub build)
    - COLMAP  : official GitHub release, Windows CUDA build (version pinned, fallback: latest)
Nothing outside the app folder is modified; nothing is deleted.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
RUNTIME = os.path.join(ROOT, "runtime")
COLMAP_TAG = "3.11.1"
FFMPEG_URLS = ["https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
               "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"]
UA = {"User-Agent": "GeoSplat-bootstrap"}


def download(url, label):
    print(f"  downloading {label}: {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf, got = io.BytesIO(), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
        print()
    return buf


def unzip_to(buf, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(dest)


def find(dirpath, names):
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower() in names:
                return os.path.join(root, f)
    return None


def ok(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def colmap_asset_url():
    for api in (f"https://api.github.com/repos/colmap/colmap/releases/tags/{COLMAP_TAG}",
                "https://api.github.com/repos/colmap/colmap/releases/latest"):
        try:
            with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=30) as r:
                rel = json.load(r)
            for a in rel.get("assets", []):
                n = a["name"].lower()
                if "windows" in n and "cuda" in n and "nocuda" not in n and n.endswith(".zip"):
                    return a["browser_download_url"], rel.get("tag_name")
        except Exception as e:  # noqa: BLE001
            print("  GitHub API:", e)
    return None, None


def main():
    os.makedirs(RUNTIME, exist_ok=True)
    tj = os.path.join(RUNTIME, "tools.json")
    tools = json.load(open(tj)) if os.path.exists(tj) else {}

    # ---------------- FFmpeg
    ff = tools.get("ffmpeg") if tools.get("ffmpeg") and os.path.exists(tools["ffmpeg"]) else shutil.which("ffmpeg")
    if not ff:
        ff = find(os.path.join(TOOLS, "ffmpeg"), {"ffmpeg.exe"}) if os.path.isdir(os.path.join(TOOLS, "ffmpeg")) else None
    if not ff:
        print("[FFmpeg] not found - downloading portable build")
        for url in FFMPEG_URLS:
            try:
                unzip_to(download(url, "FFmpeg"), os.path.join(TOOLS, "ffmpeg"))
                ff = find(os.path.join(TOOLS, "ffmpeg"), {"ffmpeg.exe"})
                if ff:
                    break
            except Exception as e:  # noqa: BLE001
                print("  failed:", e)
    if ff and ok([ff, "-version"]):
        tools["ffmpeg"] = ff
        fp = os.path.join(os.path.dirname(ff), "ffprobe.exe" if ff.lower().endswith(".exe") else "ffprobe")
        tools["ffprobe"] = fp if os.path.exists(fp) else shutil.which("ffprobe")
        print("[FFmpeg] OK:", ff)
    else:
        print("[FFmpeg] NOT AVAILABLE - install manually: winget install Gyan.FFmpeg")

    # ---------------- COLMAP
    cm = tools.get("colmap") if tools.get("colmap") and os.path.exists(tools["colmap"]) else shutil.which("colmap")
    for d in (r"C:\colmap-x64-windows-cuda", os.path.join(TOOLS, "colmap")):
        if not cm and os.path.isdir(d):
            cm = find(d, {"colmap.bat", "colmap.exe"})
    if not cm:
        url, tag = colmap_asset_url()
        if url:
            print(f"[COLMAP] not found - downloading {tag} (Windows CUDA build, several hundred MB)")
            try:
                unzip_to(download(url, f"COLMAP {tag}"), os.path.join(TOOLS, "colmap"))
                cm = find(os.path.join(TOOLS, "colmap"), {"colmap.bat"}) or find(os.path.join(TOOLS, "colmap"), {"colmap.exe"})
            except Exception as e:  # noqa: BLE001
                print("  failed:", e)
    if cm and ok([cm, "help"]):
        tools["colmap"] = cm
        print("[COLMAP] OK:", cm)
    else:
        print("[COLMAP] NOT AVAILABLE - download 'colmap-x64-windows-cuda.zip' from "
              "https://github.com/colmap/colmap/releases and unzip to C:\\colmap-x64-windows-cuda")

    json.dump(tools, open(tj, "w"), indent=2)
    print("tools.json:", json.dumps(tools, indent=2))
    return 0 if tools.get("ffmpeg") and tools.get("colmap") else 1


if __name__ == "__main__":
    sys.exit(main())
