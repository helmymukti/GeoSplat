"""
GeoSplat desktop app (PySide6). Drop a DJI video (+ optional .SRT and RTK checkpoint .CSV) into the window
and press Start.

    python -m geosplat.gui [VIDEO.MP4] [FILE.SRT] [CHECKPOINTS.CSV]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QGuiApplication
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QTextBrowser, QVBoxLayout, QWidget)

from . import __version__, about
from .pipeline import (ROOT, STAGES, Job, Pipeline, check_prerequisites, default_project, find_srt, load_settings,
                       save_settings, serve_viewer)

ACC = "#ff6a13"
MUTED = "#8f8f8f"
OK, BAD = "#53c27a", "#ff5b5b"
QSS = f"""
QWidget {{ background:#18181a; color:#e8e8e8; font-family:'Segoe UI'; font-size:10pt; }}
QMenuBar {{ background:#18181a; color:#cfcfcf; padding:2px 6px; }}
QMenuBar::item:selected {{ background:#2c2c2e; border-radius:4px; }}
QMenu {{ background:#232325; border:1px solid #333; padding:4px; }}
QMenu::item {{ padding:6px 22px; border-radius:4px; }}
QMenu::item:selected {{ background:#333336; }}
QFrame#card {{ background:#222224; border:1px solid #2e2e30; border-radius:10px; }}
QLabel#title {{ font-size:17pt; font-weight:600; letter-spacing:0.3px; background:transparent; }}
QLabel#badge {{ background:#2e2e30; color:#bdbdbd; border-radius:9px; padding:2px 9px; font-size:8.5pt; }}
QLabel#key {{ color:{MUTED}; background:transparent; }}
QLabel#section {{ color:{MUTED}; font-size:8.5pt; font-weight:600; letter-spacing:1px; background:transparent; }}
QLabel#footer {{ color:#6f6f6f; font-size:8.5pt; background:transparent; }}
QFrame#drop {{ background:#1d1d1f; border:2px dashed #3d3d40; border-radius:12px; }}
QFrame#drop[hover="true"] {{ border-color:{ACC}; background:#261d15; }}
QPushButton {{ background:#2f2f32; border:0; border-radius:6px; padding:7px 16px; }}
QPushButton:hover {{ background:#3a3a3d; }}
QPushButton:disabled {{ color:#5f5f5f; background:#252527; }}
QPushButton#primary {{ background:{ACC}; color:#111; font-weight:600; }}
QPushButton#primary:hover {{ background:#ff7d33; }}
QPushButton#primary:disabled {{ background:#4d3424; color:#8a8a8a; }}
QPushButton#link {{ background:transparent; color:#9fb4ff; padding:0 4px; font-size:8.5pt; }}
QPushButton#link:hover {{ color:#c7d3ff; text-decoration:underline; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:#141416; border:1px solid #353538; border-radius:5px;
    padding:4px 6px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color:{ACC}; }}
QPlainTextEdit {{ background:#111113; border:1px solid #2a2a2c; border-radius:8px; font-family:Consolas;
    font-size:9pt; color:#bdbdbd; }}
QTextBrowser {{ background:#1d1d1f; border:0; }}
QProgressBar {{ background:#141416; border:0; border-radius:3px; max-height:6px; }}
QProgressBar::chunk {{ background:{ACC}; border-radius:3px; }}
QToolTip {{ background:#2a2a2c; color:#e8e8e8; border:1px solid #3a3a3c; padding:4px; }}
"""
ICON = {"pending": ("○", "#6a6a6a"), "running": ("◐", ACC), "done": ("✓", OK), "skipped": ("✓", OK),
        "failed": ("✗", BAD), "cancelled": ("■", "#999")}
NO_CHECKPOINTS = "None (optional; without checkpoints the absolute height of the model is not corrected)"


class Worker(QObject):
    log = Signal(str)
    stage = Signal(str, str, str)
    finished = Signal(bool)

    def __init__(self, job):
        super().__init__()
        self.pl = Pipeline(job, on_log=self.log.emit, on_stage=lambda k, s, i="": self.stage.emit(k, s, i))

    def run(self):
        try:
            ok = self.pl.run()
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"ERROR: {e}")
            ok = False
        self.finished.emit(ok)


class DropZone(QFrame):
    def __init__(self, on_files):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.on_files = on_files
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 22, 20, 22)
        lay.setSpacing(8)
        t = QLabel("Drop a DJI video (.MP4) here")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size:13pt; font-weight:600; background:transparent;")
        s = QLabel("Add the matching .SRT and, optionally, an RTK checkpoint .CSV. "
                   "Without an SRT, telemetry is read from the video itself.")
        s.setAlignment(Qt.AlignCenter)
        s.setWordWrap(True)
        s.setStyleSheet(f"color:{MUTED}; background:transparent;")
        b = QPushButton("Browse…")
        b.clicked.connect(self.pick)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addWidget(b, alignment=Qt.AlignCenter)

    def pick(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select DJI video, SRT and (optional) checkpoint CSV", "",
            "DJI video / telemetry / checkpoints (*.mp4 *.MP4 *.mov *.MOV *.srt *.SRT *.csv *.CSV)")
        if files:
            self.on_files(files)

    def _hover(self, v):
        self.setProperty("hover", v)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        self.on_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


class AboutDialog(QDialog):
    """Authors, affiliations, licence, acknowledgements and citation."""

    def __init__(self, parent=None, focus_cite=False):
        super().__init__(parent)
        self.setWindowTitle(f"About {about.NAME}")
        self.resize(640, 600)
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        txt = QTextBrowser()
        txt.setOpenExternalLinks(True)
        txt.setHtml(self._html())
        v.addWidget(txt, 1)
        row = QHBoxLayout()
        for label, fn in (("Copy citation (APA)", about.citation_apa), ("Copy BibTeX", about.citation_bibtex)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, f=fn, btn=b: self._copy(f(), btn))
            row.addWidget(b)
        row.addStretch()
        close = QPushButton("Close")
        close.setObjectName("primary")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)
        if focus_cite:
            txt.scrollToAnchor("cite")

    @staticmethod
    def _copy(text, btn):
        QGuiApplication.clipboard().setText(text)
        old = btn.text()
        btn.setText("Copied ✓")
        QTimer.singleShot(1500, lambda: btn.setText(old))

    @staticmethod
    def _html():
        authors = ", ".join(f"<b>{n}</b><sup>{a}{',*' if c else ''}</sup>" for n, a, c in about.AUTHORS)
        affs = "<br>".join(f"<sup>{k}</sup> {v}" for k, v in about.AFFILIATIONS.items())
        ack = "".join(f"<li><b>{n}</b> — {who} <span style='color:{MUTED}'>({lic})</span></li>"
                      for n, who, lic in about.ACKNOWLEDGEMENTS)
        return f"""
        <div style="font-family:'Segoe UI'; color:#e8e8e8;">
        <p style="font-size:18pt; font-weight:600; margin:0">{about.NAME}
           <span style="font-size:10pt; color:{MUTED}; font-weight:400">version {__version__}</span></p>
        <p style="color:{MUTED}; margin-top:2px">{about.TAGLINE}</p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px">AUTHORS</p>
        <p style="margin-top:0">{authors}</p>
        <p style="font-size:9pt; color:#c8c8c8">{affs}<br><sup>*</sup> Corresponding author:
           <a style="color:#9fb4ff" href="mailto:{about.CORRESPONDENCE}">{about.CORRESPONDENCE}</a></p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px"><a name="cite"></a>HOW TO CITE</p>
        <p style="margin-top:0; font-size:9.5pt">{about.citation_apa()}</p>
        <p style="font-size:9pt; color:#c8c8c8">Please also cite the methods GeoSplat builds on
           (3D Gaussian Splatting, COLMAP, Umeyama 1991; full list in docs/METHODS.md).</p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px">LICENCE</p>
        <p style="margin-top:0; font-size:9.5pt">GeoSplat is released under the {about.LICENSE} licence.
           Third-party software is downloaded from its official sources at install time, is not bundled,
           and keeps its own licence:</p>
        <ul style="font-size:9.5pt">{ack}</ul>
        <p style="font-size:9.5pt">Source code: <a style="color:#9fb4ff" href="{about.REPO_URL}">{about.REPO_URL}</a></p>
        </div>"""


class Main(QMainWindow):
    def __init__(self, argv_files):
        super().__init__()
        self.setWindowTitle(f"{about.NAME} {__version__}")
        self.resize(1060, 860)
        self.settings = load_settings()
        self.video = self.srt = None
        self.checkpoints = None
        self.project = None
        self.thread = self.worker = None
        self.t_start = None
        self._menus()

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(20, 12, 20, 10)
        v.setSpacing(12)

        # header ---------------------------------------------------------------
        head = QHBoxLayout()
        mark = QLabel("◆")
        mark.setStyleSheet(f"color:{ACC}; font-size:17pt; background:transparent;")
        t = QLabel(about.NAME)
        t.setObjectName("title")
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("badge")
        head.addWidget(mark)
        head.addWidget(t)
        head.addSpacing(4)
        head.addWidget(ver, alignment=Qt.AlignVCenter)
        head.addStretch()
        self.pre = QLabel()
        self.pre.setStyleSheet("background:transparent;")
        self.pre.setToolTip("Required components detected on this computer")
        head.addWidget(self.pre)
        head.addSpacing(10)
        bset = QPushButton("Set up GPU…")
        bset.setToolTip("Install Graphdeco 3D Gaussian Splatting + PyTorch CUDA (one-time setup)")
        bset.clicked.connect(self.setup_gpu)
        head.addWidget(bset)
        v.addLayout(head)

        self.drop = DropZone(self.set_files)
        v.addWidget(self.drop)

        # project card -----------------------------------------------------------
        card = QFrame()
        card.setObjectName("card")
        g = QGridLayout(card)
        g.setContentsMargins(18, 14, 18, 14)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(9)
        sec = QLabel("PROJECT")
        sec.setObjectName("section")
        g.addWidget(sec, 0, 0, 1, 7)
        self.lv, self.ls, self.lp = QLabel("–"), QLabel("–"), QLabel("–")
        for lab in (self.lv, self.ls, self.lp):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bout = QPushButton("Change…")
        bout.clicked.connect(self.pick_project)
        for r, (name, w) in enumerate((("Video", self.lv), ("Telemetry", self.ls), ("Output folder", self.lp)), 1):
            g.addWidget(self._key(name), r, 0)
            g.addWidget(w, r, 1, 1, 5)
        g.addWidget(bout, 3, 6)

        self.lc = QLabel(NO_CHECKPOINTS)
        self.lc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lc.setWordWrap(True)
        bcp = QPushButton("Browse…")
        bcp.clicked.connect(self.pick_checkpoints)
        k = self._key("RTK checkpoints")
        k.setToolTip("Optional: GNSS/RTK checkpoint CSV (id,E,N,h or id,lat,lon,h; ellipsoidal heights).\n"
                     "Used by stage 6 to correct the height datum of the model.")
        g.addWidget(k, 4, 0)
        g.addWidget(self.lc, 4, 1, 1, 5)
        g.addWidget(bcp, 4, 6)

        sec2 = QLabel("SETTINGS")
        sec2.setObjectName("section")
        g.addWidget(sec2, 5, 0, 1, 7)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.5, 5.0)
        self.fps.setSingleStep(0.5)
        self.fps.setSuffix(" fps")
        self.fps.setValue(float(self.settings.get("fps", 2.0)))
        self.fps.setToolTip("Frames extracted per second of video")
        self.prof = QComboBox()
        self.prof.addItems(["balanced", "low", "quality"])
        self.prof.setCurrentText(self.settings.get("profile", "balanced"))
        self.prof.setToolTip("low: ≤ 6 GB VRAM · balanced: 8 GB · quality: ≥ 12 GB")
        self.it = QSpinBox()
        self.it.setRange(1000, 60000)
        self.it.setSingleStep(1000)
        self.it.setValue(int(self.settings.get("iterations", 30000)))
        self.it.setToolTip("3D Gaussian Splatting training iterations")
        for c, (name, w) in enumerate((("Frame rate", self.fps), ("GPU profile", self.prof), ("Iterations", self.it))):
            g.addWidget(self._key(name), 6, c * 2)
            g.addWidget(w, 6, c * 2 + 1)
        self.key = QLineEdit(self.settings.get("gmaps_key", ""))
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Google Map Tiles API key (optional; Esri World Imagery is used without a key)")
        g.addWidget(self._key("Satellite imagery"), 7, 0)
        g.addWidget(self.key, 7, 1, 1, 6)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setColumnStretch(5, 1)
        v.addWidget(card)

        # stages ------------------------------------------------------------------
        stc = QFrame()
        stc.setObjectName("card")
        sg = QGridLayout(stc)
        sg.setContentsMargins(18, 12, 18, 12)
        sg.setVerticalSpacing(5)
        sec3 = QLabel("PROCESSING")
        sec3.setObjectName("section")
        sg.addWidget(sec3, 0, 0, 1, 3)
        self.st = {}
        for r, (key, label) in enumerate(STAGES):
            ic = QLabel()
            ic.setFixedWidth(22)
            nm = QLabel(f"{r + 1}.  {label}")
            nm.setStyleSheet("background:transparent;")
            inf = QLabel("")
            inf.setStyleSheet(f"color:{MUTED}; background:transparent;")
            sg.addWidget(ic, r + 1, 0)
            sg.addWidget(nm, r + 1, 1)
            sg.addWidget(inf, r + 1, 2)
            self.st[key] = (ic, inf)
            self.set_stage(key, "pending")
        sg.setColumnStretch(2, 1)
        v.addWidget(stc)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, len(STAGES))
        self.now = QLabel("Ready.")
        self.now.setStyleSheet(f"color:{MUTED};")
        v.addWidget(self.bar)
        v.addWidget(self.now)

        self.logw = QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(5000)
        self.logw.setPlaceholderText("Processing log")
        v.addWidget(self.logw, 1)

        btns = QHBoxLayout()
        self.bstart = QPushButton("Start")
        self.bstart.setObjectName("primary")
        self.bstart.setMinimumWidth(150)
        self.bstart.clicked.connect(self.start)
        self.bstop = QPushButton("Cancel")
        self.bstop.clicked.connect(self.cancel)
        self.bstop.setEnabled(False)
        self.bview = QPushButton("Open satellite viewer")
        self.bview.clicked.connect(self.open_viewer)
        self.bfold = QPushButton("Open output folder")
        self.bfold.clicked.connect(self.open_folder)
        btns.addWidget(self.bstart)
        btns.addWidget(self.bstop)
        btns.addStretch()
        btns.addWidget(self.bfold)
        btns.addWidget(self.bview)
        v.addLayout(btns)

        # footer: permanent author credit ---------------------------------------------
        foot = QHBoxLayout()
        cred = QLabel(about.credit_line())
        cred.setObjectName("footer")
        cred.setToolTip("\n".join(f"{n} — {about.AFFILIATIONS[a]}" for n, a, _ in about.AUTHORS))
        foot.addWidget(cred)
        foot.addStretch()
        for label, cite in (("How to cite", True), ("About", False)):
            b = QPushButton(label)
            b.setObjectName("link")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, c=cite: self.show_about(c))
            foot.addWidget(b)
        v.addLayout(foot)

        self.refresh_prereq()
        self.update_buttons()
        if argv_files:
            self.set_files(argv_files)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    @staticmethod
    def _key(text):
        k = QLabel(text)
        k.setObjectName("key")
        return k

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for label, fn in (("Open video…", lambda: self.drop.pick()), ("Open RTK checkpoints…", self.pick_checkpoints),
                          ("Change output folder…", self.pick_project)):
            a = QAction(label, self)
            a.triggered.connect(fn)
            f.addAction(a)
        f.addSeparator()
        q = QAction("Exit", self)
        q.triggered.connect(self.close)
        f.addAction(q)
        h = mb.addMenu("&Help")
        items = (("Methods and references", lambda: self._open(os.path.join(ROOT, "docs", "METHODS.md"))),
                 ("User guide (README)", lambda: self._open(os.path.join(ROOT, "README.md"))),
                 ("Project website", lambda: webbrowser.open(about.REPO_URL)),
                 (None, None),
                 ("How to cite GeoSplat…", lambda: self.show_about(True)),
                 ("About GeoSplat…", lambda: self.show_about(False)))
        for label, fn in items:
            if label is None:
                h.addSeparator()
                continue
            a = QAction(label, self)
            a.triggered.connect(fn)
            h.addAction(a)

    @staticmethod
    def _open(path):
        if os.name == "nt":
            os.startfile(path)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", path])

    def show_about(self, cite=False):
        AboutDialog(self, cite).exec()

    # ------------------------------------------------------------------ state
    def refresh_prereq(self):
        p = check_prerequisites()
        part = lambda ok, name: f"<span style='color:{OK if ok else BAD}'>{'✓' if ok else '✗'} {name}</span>"  # noqa: E731
        self.prereq = p
        self.pre.setText("&nbsp;&nbsp;".join([part(p["ffmpeg"], "FFmpeg"), part(p["colmap"], "COLMAP"),
                                              part(p["gpu"], "GPU"), part(p["gaussian_splatting"], "3DGS")]))

    def set_files(self, files):
        vids = [f for f in files if os.path.splitext(f)[1].lower() in (".mp4", ".mov")]
        srts = [f for f in files if os.path.splitext(f)[1].lower() == ".srt"]
        csvs = [f for f in files if os.path.splitext(f)[1].lower() == ".csv"]
        if csvs:
            self.set_checkpoints(csvs[0])
        if vids:
            self.video = os.path.abspath(vids[0])
            self.srt = os.path.abspath(srts[0]) if srts else find_srt(self.video)
            self.project = default_project(self.video)
        elif srts and self.video:
            self.srt = os.path.abspath(srts[0])
        self.lv.setText(self.video or "–")
        self.ls.setText(self.srt if self.srt else
                        ("No SRT — using the telemetry track embedded in the video" if self.video else "–"))
        self.lp.setText(self.project or "–")
        self.mark_done_stages()
        self.update_buttons()

    def mark_done_stages(self):
        if not self.project:
            return
        for key, _ in STAGES:
            done = os.path.exists(os.path.join(self.project, "logs", f"_stage_{key}.done"))
            self.set_stage(key, "done" if done else "pending", "completed earlier" if done else "")

    def set_checkpoints(self, path):
        self.checkpoints = os.path.abspath(path) if path else None
        self.lc.setText(self.checkpoints or NO_CHECKPOINTS)

    def pick_checkpoints(self):
        f, _ = QFileDialog.getOpenFileName(self, "RTK checkpoint CSV", os.path.dirname(self.video or ROOT),
                                           "CSV (*.csv)")
        if f:
            self.set_checkpoints(f)

    def pick_project(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", os.path.dirname(self.project or ROOT))
        if d:
            self.project = d
            self.lp.setText(d)
            self.mark_done_stages()
            self.update_buttons()

    def set_stage(self, key, status, info=""):
        ic, inf = self.st[key]
        sym, col = ICON.get(status, ("?", "#999"))
        ic.setText(sym)
        ic.setStyleSheet(f"color:{col}; font-size:13pt; background:transparent;")
        inf.setText(info)
        if status in ("done", "skipped"):
            self.bar.setValue(max(self.bar.value(), [k for k, _ in STAGES].index(key) + 1))
        if status == "running":
            self.cur = (key, time.time())

    def update_buttons(self):
        running = self.thread is not None
        self.bstart.setEnabled(bool(self.video) and not running)
        self.bstop.setEnabled(running)
        has_web = bool(self.project) and os.path.exists(os.path.join(self.project, "08_Exports", "web", "index.html"))
        self.bview.setEnabled(has_web and not running)
        self.bfold.setEnabled(bool(self.project) and os.path.isdir(self.project or ""))

    def tick(self):
        if self.thread and getattr(self, "cur", None):
            k, t0 = self.cur
            el = time.time() - t0
            self.st[k][1].setText(f"running {int(el // 60)}:{int(el % 60):02d}")

    # ------------------------------------------------------------------ actions
    def start(self):
        if not self.prereq["ffmpeg"] or not self.prereq["colmap"]:
            QMessageBox.warning(self, about.NAME, "FFmpeg or COLMAP was not found. Run GeoSplat.bat again "
                                                  "(automatic setup) or install them manually.")
            return
        if not self.prereq["gaussian_splatting"]:
            r = QMessageBox.question(self, about.NAME, "Gaussian Splatting (GPU) is not installed.\n\n"
                                     "Telemetry, COLMAP, georeferencing and the height check will still run. "
                                     "Continue?")
            if r != QMessageBox.Yes:
                return
        self.settings.update({"fps": self.fps.value(), "profile": self.prof.currentText(), "iterations": self.it.value(),
                              "gmaps_key": self.key.text().strip()})
        save_settings(self.settings)
        stages = [k for k, _ in STAGES]
        if not self.prereq["gaussian_splatting"]:
            stages = ["telemetry", "colmap", "georef", "vcorr"]
        job = Job(self.video, self.srt, self.project, self.fps.value(), self.prof.currentText(), self.it.value(),
                  self.key.text().strip(), False, stages, self.checkpoints)
        self.logw.clear()
        self.bar.setValue(0)
        for key, _ in STAGES:
            self.set_stage(key, "pending")
        self.thread = QThread()
        self.worker = Worker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.on_log)
        self.worker.stage.connect(self.set_stage)
        self.worker.finished.connect(self.on_done)
        self.thread.start()
        self.t_start = time.time()
        self.now.setText("Running… You can leave it unattended; an interrupted run resumes where it stopped.")
        self.update_buttons()

    def on_log(self, line):
        if line.strip():
            self.logw.appendPlainText(line)
            if "Training progress" in line or line.startswith("==="):
                self.now.setText(line.strip()[-140:])

    def on_done(self, ok):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        el = (time.time() - self.t_start) / 60 if self.t_start else 0
        self.now.setText(("Finished" if ok else "Stopped") + f" · {el:.1f} min")
        self.update_buttons()
        if ok and self.bview.isEnabled():
            self.open_viewer()

    def cancel(self):
        if self.worker:
            self.worker.pl.stop()
            self.now.setText("Cancelling…")

    def open_viewer(self):
        url = serve_viewer(os.path.join(self.project, "08_Exports", "web"))
        webbrowser.open(url)

    def open_folder(self):
        self._open(self.project)

    def setup_gpu(self):
        bat = os.path.join(ROOT, "setup", "install_gpu.bat")
        if os.name == "nt":
            subprocess.Popen(["cmd", "/c", "start", "GeoSplat GPU setup", "cmd", "/k", bat], cwd=ROOT)
        QMessageBox.information(self, about.NAME, "The GPU setup is running in a separate window.\n"
                                                  "When it shows 'SETUP COMPLETE', close that window and click OK.")
        self.refresh_prereq()

    def closeEvent(self, e):
        if self.thread:
            if QMessageBox.question(self, about.NAME, "Processing is still running. Stop it and exit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self.cancel()
            self.thread.quit()
            self.thread.wait(5000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(about.NAME)
    app.setApplicationVersion(__version__)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    w = Main([a for a in sys.argv[1:] if os.path.isfile(a)])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
