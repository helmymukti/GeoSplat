"""
GeoSplat command line:  python -m geosplat.cli VIDEO.MP4 [--srt FILE.SRT] [--project DIR] [--fps 2]
                                               [--profile balanced|low|quality] [--iterations 30000] [--serve]
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser

from .pipeline import STAGES, Job, Pipeline, check_prerequisites, load_settings, serve_viewer


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geosplat", description="DJI video -> georeferenced 3D Gaussian Splatting")
    ap.add_argument("video")
    ap.add_argument("--srt", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--profile", default="balanced", choices=["balanced", "low", "quality"])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--stages", default=",".join(k for k, _ in STAGES))
    ap.add_argument("--redo", action="store_true", help="re-run stages already marked done")
    ap.add_argument("--serve", action="store_true", help="open the satellite viewer when finished")
    a = ap.parse_args(argv)
    print("Prerequisites:", check_prerequisites())
    job = Job(a.video, a.srt, a.project, a.fps, a.profile, a.iterations, load_settings().get("gmaps_key", ""),
              a.redo, a.stages.split(","))
    print(f"Video   : {job.video}\nSRT     : {job.srt or '(none - embedded telemetry)'}\nProject : {job.project}")
    pl = Pipeline(job, on_log=print, on_stage=lambda k, s, i="": print(f"[{k}] {s} {i}"))
    ok = pl.run()
    if ok and a.serve:
        url = serve_viewer(pl.viewer_dir())
        print("Viewer:", url, "(Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
