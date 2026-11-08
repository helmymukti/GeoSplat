# Contributing to GeoSplat

Thank you for helping. Three rules keep the project honest and maintainable:

1. **Cite the method.** Every algorithm added to GeoSplat must be described in `docs/METHODS.md` with the equations
   it implements and a reference to the original publication. If an implementation follows an existing open-source
   project, name it in the code and in `THIRD_PARTY_NOTICES.md`.
2. **Do not vendor third-party code.** External tools (Graphdeco 3DGS, COLMAP, FFmpeg, …) are downloaded from their
   official sources by the installers and keep their own licences.
3. **No data or secrets in git.** Videos, outputs, API keys and local settings are excluded by `.gitignore`.

## Development
```bat
GeoSplat.bat                       :: first run creates .venv
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m geosplat.cli VIDEO.MP4 --stages telemetry
```
Each pipeline stage in `scripts/` can be run on its own with `--project <folder>`; see the docstring at the top
of each script.
