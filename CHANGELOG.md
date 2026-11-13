# Changelog

## [0.2.0] - 2026-11-13
Vertical datum correction from RTK checkpoints, English user interface, author credits.
### Added
- Stage 6 (`scripts/stage6_vcorr.py`, `geosplat/vcorr.py`): height datum check in every project and, with a
  checkpoint CSV (`id,E,N,h` or `id,lat,lon,h`; ellipsoidal heights), vertical correction of the model. ASPRS Ed. 2
  reporting: leave-one-out RMSE, NMAD, bootstrap intervals, pairwise height test. Corrected copies are written to
  `10_Vertical_Correction/`; original outputs are never modified.
- Optional checkpoint harmonisation (NTRIP base re-referencing, epoch shift) from a JSON file next to the CSV.
- Altitude-reference diagnosis: warns when DJI `abs_alt` is home-point referenced.
- Checkpoint input in the app (button or drag & drop) and CLI (`--checkpoints`); templates in `docs/examples/`.
- Author credits that stay with the software and its results: footer and **Help → About GeoSplat** in the app
  (authors, affiliations, licence, third-party acknowledgements, copy-ready APA and BibTeX citation),
  `geosplat --version` and a banner in the command line, a `provenance` block in every project's `logs/job.json`,
  a footer in the HTML reports and a credit line in the satellite viewer (`geosplat/about.py`).
### Changed
- The whole application is in English: desktop app, command line, setup scripts, satellite viewer and reports.
- Cleaner main window: menu bar (File, Help), version badge, grouped Project / Settings / Processing panels.
- Google satellite tiles use the browser's language and region.
### Fixed
- Viewer metadata no longer hard-codes EPSG:32749.

## [0.1.0] - 2026-11-09
First public version.
### Added
- One-click Windows app (`GeoSplat.bat`): drag & drop a DJI video (+ optional SRT), automatic five-stage pipeline.
- Telemetry from DJI SRT (current and legacy format) and from the embedded `djmd` protobuf track (verified for
  Mavic 3 Enterprise `dvtm_wm265e`).
- Frame extraction with exact-PTS selection, hardware-decode probing, resume, input-duration limit.
- Epoch-based telemetry synchronisation in local ENU with quality flags.
- GNSS-guided pair selection for COLMAP; version-aware COLMAP wrapper.
- Robust Sim3 georeferencing (Umeyama + RANSAC), time-offset scan, 5-fold check points, ENU/ECEF/WGS84/UTM chain.
- Graphdeco 3D Gaussian Splatting trained in metric ENU; exact similarity transform of Gaussians incl. SH rotation;
  UTM export with false origin; GS orthophoto with world file.
- WebGL2 satellite viewer (Google Map Tiles API / Esri fallback) with measurement and blink comparison.
- Authors and affiliations (`AUTHORS.md`, `CITATION.cff`).
- Installers for portable FFmpeg/COLMAP and the GPU environment; unit tests; methods documentation with references.
