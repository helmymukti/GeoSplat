"""Plots, HTML and text reports for Phase 1 (matplotlib, fully offline)."""
from __future__ import annotations

import base64
import html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_trajectory(tel, frames, utm_label, out_png):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    E, N = tel["utm_e"].to_numpy(), tel["utm_n"].to_numpy()
    e0, n0 = np.floor(E.min() / 100) * 100, np.floor(N.min() / 100) * 100
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("blues_trunc", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
    sc = ax.scatter(E - e0, N - n0, c=tel["t"], cmap=cm, s=4, lw=0)
    ax.plot(E - e0, N - n0, color=C1, lw=0.6, alpha=0.5)
    if frames is not None and len(frames):
        ok = frames["sync_quality"] != "BAD"
        ax.scatter(frames["utm_e"][ok] - e0, frames["utm_n"][ok] - n0, s=14, facecolor="none",
                   edgecolor=C2, lw=1.0, label=f"extracted frames ({ok.sum()})", zorder=3)
    ax.scatter([E[0] - e0], [N[0] - n0], s=80, marker="^", color=C3, zorder=4, label="start")
    ax.scatter([E[-1] - e0], [N[-1] - n0], s=80, marker="s", color=INK, zorder=4, label="end")
    # heading arrows every ~10 s (gimbal yaw = camera viewing azimuth)
    step = max(1, int(len(tel) / 11))
    for i in range(0, len(tel), step):
        a = np.radians(tel["gimbal_yaw"].iloc[i])
        ax.annotate("", xy=(E[i] - e0 + 18 * np.sin(a), N[i] - n0 + 18 * np.cos(a)),
                    xytext=(E[i] - e0, N[i] - n0),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel(f"Easting − {e0:.0f} m   ({utm_label})")
    ax.set_ylabel(f"Northing − {n0:.0f} m")
    ax.set_title("GNSS camera trajectory (djmd track, 10 Hz epochs)  ·  arrows = gimbal yaw",
                 fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("video time (s)", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_altitude(tel, out_png):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)
    t = tel["t"]
    axs[0].plot(t, tel["rel_alt"], color=C1, lw=1.5)
    axs[0].set_ylabel("relative alt (m)\nabove take-off")
    axs[0].set_title("Altitude, speed and attitude profile", fontsize=10, color=INK, loc="left")
    axs[1].plot(t, np.hypot(tel["vel_n"], tel["vel_e"]), color=C1, lw=1.5, label="horizontal (djmd)")
    if "gnss_speed_h_ms" in tel:
        axs[1].plot(t, tel["gnss_speed_h_ms"], color=C2, lw=1.0, label="GNSS-derived")
    axs[1].set_ylabel("speed (m/s)")
    axs[1].legend(fontsize=8, frameon=False)
    axs[2].plot(t, tel["drone_pitch"], color=C1, lw=1.0, label="drone pitch")
    axs[2].plot(t, tel["drone_roll"], color=C2, lw=1.0, label="drone roll")
    axs[2].plot(t, tel["gimbal_pitch"], color=C3, lw=1.5, label="gimbal pitch")
    axs[2].set_ylabel("angle (deg)")
    axs[2].set_xlabel("video time (s)")
    axs[2].legend(fontsize=8, frameon=False, ncol=3)
    for a in axs:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_sync(sync_all, out_png):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), dpi=150, sharex=True)
    t = sync_all["t"]
    axs[0].plot(t, sync_all["staircase_err_m"] * 100, color=C2, lw=0.8)
    axs[0].set_ylabel("cm")
    axs[0].set_title("Horizontal error avoided by epoch interpolation vs raw per-frame value",
                     fontsize=10, color=INK, loc="left")
    axs[1].plot(t, sync_all["srt_vs_djmd_h_m"] * 100, color=C1, lw=0.8)
    axs[1].set_ylabel("cm")
    axs[1].set_xlabel("video time (s)")
    axs[1].set_title("SRT vs embedded djmd position difference (SRT rounding to 1e-6 deg)",
                     fontsize=10, color=INK, loc="left")
    for a in axs:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def _table(rows):
    h = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table>{h}</table>"


def write_html(out_html, title, sections, images):
    css = """
    body{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#0f1b2d;color:#e8edf5}
    header{padding:22px 32px;background:#0a1322;border-bottom:1px solid #22324d}
    h1{margin:0;font-size:22px} h2{font-size:15px;margin:28px 0 8px;color:#8fb6ff}
    .sub{color:#9fb0c8;font-size:13px;margin-top:4px}
    main{padding:8px 32px 40px;max-width:1200px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
    .card{background:#14233a;border:1px solid #22324d;border-radius:8px;padding:14px 16px}
    table{border-collapse:collapse;width:100%;font-size:12.5px}
    th{text-align:left;color:#9fb0c8;font-weight:500;padding:4px 8px 4px 0;vertical-align:top;width:45%}
    td{padding:4px 0;font-family:Consolas,monospace;color:#e8edf5;word-break:break-word}
    tr{border-bottom:1px solid #1d2d47}
    img{width:100%;background:#fff;border-radius:6px}
    .warn{color:#ffcf70}.ok{color:#7fe0b0}
    """
    from .about import credit_line
    from . import __version__
    body = [f"<header><h1>{html.escape(title)}</h1><div class='sub'>Generated locally by GeoSplat {__version__} · "
            f"no data left this computer</div></header><main>"]
    body.append("<div class='grid'>")
    for name, rows in sections:
        body.append(f"<div class='card'><h2>{html.escape(name)}</h2>{_table(rows)}</div>")
    body.append("</div>")
    for cap, p in images:
        body.append(f"<h2>{html.escape(cap)}</h2><div class='card'><img src='{_img(p)}'></div>")
    body.append(f"<footer style='padding:14px 32px;color:#6f84a3;font-size:12px'>{html.escape(credit_line())}"
                "</footer></main>")
    open(out_html, "w", encoding="utf-8").write(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{css}</style></head><body>{''.join(body)}</body></html>")


def write_text_report(path, blocks):
    with open(path, "w", encoding="utf-8") as f:
        for name, rows in blocks:
            f.write("=" * 78 + "\n" + name + "\n" + "=" * 78 + "\n")
            for k, v in rows:
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, default=str)
                f.write(f"{k:<42}: {v}\n")
            f.write("\n")
