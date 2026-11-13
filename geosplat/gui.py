"""
GeoSplat desktop app (PySide6).  Drag a DJI video (+ optional .SRT) into the window and press Mulai.

    python -m geosplat.gui [VIDEO.MP4] [FILE.SRT]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .pipeline import (ROOT, STAGES, Job, Pipeline, check_prerequisites, default_project, find_srt, load_settings,
                       save_settings, serve_viewer)

ACC = "#ff6a13"
QSS = f"""
QWidget {{ background:#1b1b1b; color:#e6e6e6; font-family:'Segoe UI'; font-size:10pt; }}
QFrame#card {{ background:#262626; border-radius:10px; }}
QLabel#title {{ font-size:15pt; font-weight:600; }}
QLabel#muted, QLabel.muted {{ color:#9a9a9a; }}
QFrame#drop {{ background:#202020; border:2px dashed #4a4a4a; border-radius:12px; }}
QFrame#drop[hover="true"] {{ border-color:{ACC}; background:#2a2016; }}
QPushButton {{ background:#333; border:0; border-radius:6px; padding:7px 14px; }}
QPushButton:hover {{ background:#3d3d3d; }}
QPushButton:disabled {{ color:#666; background:#2a2a2a; }}
QPushButton#primary {{ background:{ACC}; color:#111; font-weight:600; }}
QPushButton#primary:disabled {{ background:#5a3a26; color:#999; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:#151515; border:1px solid #3a3a3a; border-radius:5px; padding:4px 6px; }}
QPlainTextEdit {{ background:#111; border:0; border-radius:8px; font-family:Consolas; font-size:9pt; color:#bdbdbd; }}
QProgressBar {{ background:#151515; border:0; border-radius:4px; height:8px; text-align:center; }}
QProgressBar::chunk {{ background:{ACC}; border-radius:4px; }}
"""
ICON = {"pending": ("○", "#777"), "running": ("◐", ACC), "done": ("✓", "#53c27a"), "skipped": ("✓", "#53c27a"),
        "failed": ("✗", "#ff5b5b"), "cancelled": ("■", "#999")}


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
        lay.setContentsMargins(20, 26, 20, 26)
        t = QLabel("Seret video DJI (.MP4) ke sini")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size:13pt; font-weight:600; background:transparent;")
        s = QLabel("File .SRT ikut diseret kalau ada · tanpa SRT, telemetri dibaca dari dalam video")
        s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet("color:#9a9a9a; background:transparent;")
        b = QPushButton("Pilih file…")
        b.clicked.connect(self.pick)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addWidget(b, alignment=Qt.AlignCenter)

    def pick(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Pilih video DJI, SRT, dan (opsional) CSV check point", "", "DJI video/telemetri/check point (*.mp4 *.MP4 *.mov *.MOV *.srt *.SRT *.csv *.CSV)")
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


class Main(QMainWindow):
    def __init__(self, argv_files):
        super().__init__()
        self.setWindowTitle("GeoSplat — DJI video → Gaussian Splatting tergeoreferensi")
        self.resize(1040, 820)
        self.settings = load_settings()
        self.video = self.srt = None
        self.checkpoints = None
        self.project = None
        self.thread = self.worker = None
        self.t_start = None

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(12)

        head = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{ACC}; font-size:14pt;")
        t = QLabel("GeoSplat")
        t.setObjectName("title")
        sub = QLabel("lokal · tanpa upload · COLMAP + 3D Gaussian Splatting + georeferensi GNSS")
        sub.setStyleSheet("color:#9a9a9a;")
        head.addWidget(dot)
        head.addWidget(t)
        head.addSpacing(8)
        head.addWidget(sub)
        head.addStretch()
        self.pre = QLabel()
        self.pre.setStyleSheet("color:#9a9a9a;")
        head.addWidget(self.pre)
        bset = QPushButton("Siapkan GPU…")
        bset.setToolTip("Pasang Graphdeco Gaussian Splatting + PyTorch CUDA (sekali saja)")
        bset.clicked.connect(self.setup_gpu)
        head.addWidget(bset)
        v.addLayout(head)

        self.drop = DropZone(self.set_files)
        v.addWidget(self.drop)

        card = QFrame()
        card.setObjectName("card")
        g = QGridLayout(card)
        g.setContentsMargins(16, 12, 16, 12)
        g.setHorizontalSpacing(14)
        self.lv, self.ls, self.lp = QLabel("–"), QLabel("–"), QLabel("–")
        for lab in (self.lv, self.ls, self.lp):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bout = QPushButton("Ubah…")
        bout.clicked.connect(self.pick_project)
        for r, (name, w) in enumerate((("Video", self.lv), ("Telemetri", self.ls), ("Output", self.lp))):
            k = QLabel(name)
            k.setStyleSheet("color:#9a9a9a;")
            g.addWidget(k, r, 0)
            g.addWidget(w, r, 1, 1, 5)
        g.addWidget(bout, 2, 6)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.5, 5.0)
        self.fps.setSingleStep(0.5)
        self.fps.setValue(float(self.settings.get("fps", 2.0)))
        self.prof = QComboBox()
        self.prof.addItems(["balanced", "low", "quality"])
        self.prof.setCurrentText(self.settings.get("profile", "balanced"))
        self.it = QSpinBox()
        self.it.setRange(1000, 60000)
        self.it.setSingleStep(1000)
        self.it.setValue(int(self.settings.get("iterations", 30000)))
        self.key = QLineEdit(self.settings.get("gmaps_key", ""))
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Google API key (Map Tiles API) – opsional")
        for c, (name, w) in enumerate((("Frame/detik", self.fps), ("Profil GPU", self.prof), ("Iterasi", self.it))):
            k = QLabel(name)
            k.setStyleSheet("color:#9a9a9a;")
            g.addWidget(k, 3, c * 2)
            g.addWidget(w, 3, c * 2 + 1)
        k = QLabel("Satelit")
        k.setStyleSheet("color:#9a9a9a;")
        g.addWidget(k, 4, 0)
        g.addWidget(self.key, 4, 1, 1, 6)
        k = QLabel("Check point RTK")
        k.setStyleSheet("color:#9a9a9a;")
        k.setToolTip("Opsional: CSV check point GNSS/RTK (id,E,N,h atau id,lat,lon,h; tinggi ellipsoid). "
                     "Dipakai tahap 6 untuk mengoreksi datum tinggi model.")
        self.lc = QLabel("– (opsional; tanpa ini tinggi absolut model tidak dikoreksi)")
        self.lc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bcp = QPushButton("Pilih CSV…")
        bcp.clicked.connect(self.pick_checkpoints)
        g.addWidget(k, 5, 0)
        g.addWidget(self.lc, 5, 1, 1, 5)
        g.addWidget(bcp, 5, 6)
        v.addWidget(card)

        stc = QFrame()
        stc.setObjectName("card")
        sg = QGridLayout(stc)
        sg.setContentsMargins(16, 10, 16, 10)
        self.st = {}
        for r, (key, label) in enumerate(STAGES):
            ic = QLabel()
            ic.setFixedWidth(22)
            nm = QLabel(f"{r + 1}. {label}")
            inf = QLabel("")
            inf.setStyleSheet("color:#9a9a9a;")
            sg.addWidget(ic, r, 0)
            sg.addWidget(nm, r, 1)
            sg.addWidget(inf, r, 2)
            self.st[key] = (ic, inf)
            self.set_stage(key, "pending")
        sg.setColumnStretch(2, 1)
        v.addWidget(stc)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, len(STAGES))
        self.now = QLabel("Siap.")
        self.now.setStyleSheet("color:#9a9a9a;")
        v.addWidget(self.bar)
        v.addWidget(self.now)

        self.logw = QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(5000)
        v.addWidget(self.logw, 1)

        btns = QHBoxLayout()
        self.bstart = QPushButton("Mulai")
        self.bstart.setObjectName("primary")
        self.bstart.setMinimumWidth(140)
        self.bstart.clicked.connect(self.start)
        self.bstop = QPushButton("Batal")
        self.bstop.clicked.connect(self.cancel)
        self.bstop.setEnabled(False)
        self.bview = QPushButton("Buka Viewer Satelit")
        self.bview.clicked.connect(self.open_viewer)
        self.bfold = QPushButton("Buka Folder Output")
        self.bfold.clicked.connect(self.open_folder)
        btns.addWidget(self.bstart)
        btns.addWidget(self.bstop)
        btns.addStretch()
        btns.addWidget(self.bfold)
        btns.addWidget(self.bview)
        v.addLayout(btns)

        self.refresh_prereq()
        self.update_buttons()
        if argv_files:
            self.set_files(argv_files)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    # ------------------------------------------------------------------ state
    def refresh_prereq(self):
        p = check_prerequisites()
        part = lambda ok, name: f"<span style='color:{'#53c27a' if ok else '#ff5b5b'}'>{'✓' if ok else '✗'} {name}</span>"  # noqa: E731
        self.prereq = p
        self.pre.setText("  ".join([part(p["ffmpeg"], "FFmpeg"), part(p["colmap"], "COLMAP"), part(p["gpu"], "GPU"),
                                    part(p["gaussian_splatting"], "3DGS")]))

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
        self.ls.setText(self.srt if self.srt else ("Tidak ada SRT — memakai track telemetri di dalam video" if self.video else "–"))
        self.lp.setText(self.project or "–")
        if self.project:
            for key, _ in STAGES:
                done = os.path.exists(os.path.join(self.project, "logs", f"_stage_{key}.done"))
                self.set_stage(key, "done" if done else "pending", "selesai sebelumnya" if done else "")
        self.update_buttons()

    def set_checkpoints(self, path):
        self.checkpoints = os.path.abspath(path) if path else None
        self.lc.setText(self.checkpoints or "– (opsional; tanpa ini tinggi absolut model tidak dikoreksi)")

    def pick_checkpoints(self):
        f, _ = QFileDialog.getOpenFileName(self, "CSV check point RTK", os.path.dirname(self.video or ROOT),
                                           "CSV (*.csv)")
        if f:
            self.set_checkpoints(f)

    def pick_project(self):
        d = QFileDialog.getExistingDirectory(self, "Folder output", os.path.dirname(self.project or ROOT))
        if d:
            self.project = d
            self.lp.setText(d)

    def set_stage(self, key, status, info=""):
        ic, inf = self.st[key]
        sym, col = ICON.get(status, ("?", "#999"))
        ic.setText(sym)
        ic.setStyleSheet(f"color:{col}; font-size:13pt;")
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
            self.st[k][1].setText(f"berjalan {int(el // 60)}:{int(el % 60):02d}")

    # ------------------------------------------------------------------ actions
    def start(self):
        if not self.prereq["ffmpeg"] or not self.prereq["colmap"]:
            QMessageBox.warning(self, "GeoSplat", "FFmpeg/COLMAP belum ditemukan. Jalankan GeoSplat.bat sekali lagi "
                                                  "(setup otomatis) atau pasang manual.")
            return
        if not self.prereq["gaussian_splatting"]:
            r = QMessageBox.question(self, "GeoSplat", "Gaussian Splatting (GPU) belum terpasang.\n\nTahap 1–3 (telemetri, "
                                     "COLMAP, georeferensi) tetap dijalankan. Lanjutkan?")
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
        self.now.setText("Berjalan… (aman ditinggal; bisa dilanjutkan bila terhenti)")
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
        self.now.setText(("Selesai" if ok else "Berhenti") + f" · {el:.1f} menit")
        self.update_buttons()
        if ok and self.bview.isEnabled():
            self.open_viewer()

    def cancel(self):
        if self.worker:
            self.worker.pl.stop()
            self.now.setText("Membatalkan…")

    def open_viewer(self):
        url = serve_viewer(os.path.join(self.project, "08_Exports", "web"))
        webbrowser.open(url)

    def open_folder(self):
        if os.name == "nt":
            os.startfile(self.project)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", self.project])

    def setup_gpu(self):
        bat = os.path.join(ROOT, "setup", "install_gpu.bat")
        if os.name == "nt":
            subprocess.Popen(["cmd", "/c", "start", "GeoSplat GPU setup", "cmd", "/k", bat], cwd=ROOT)
        QMessageBox.information(self, "GeoSplat", "Instalasi GPU berjalan di jendela terpisah.\nSetelah muncul "
                                                  "'SETUP SELESAI', tutup jendela itu lalu klik OK.")
        self.refresh_prereq()

    def closeEvent(self, e):
        if self.thread:
            if QMessageBox.question(self, "GeoSplat", "Proses masih berjalan. Hentikan dan keluar?") != QMessageBox.Yes:
                e.ignore()
                return
            self.cancel()
            self.thread.quit()
            self.thread.wait(5000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    w = Main([a for a in sys.argv[1:] if os.path.isfile(a)])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
