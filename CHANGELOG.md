# Changelog

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
