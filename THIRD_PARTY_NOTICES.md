# Third-party software and data

GeoSplat **does not redistribute** any of the software below. The installers download it from the official sources
on the user's machine. Each component keeps its own licence; users are responsible for complying with it.

| Component | Used for | Obtained from | Licence |
|---|---|---|---|
| **Graphdeco 3D Gaussian Splatting** (Kerbl et al. 2023), incl. submodules `diff-gaussian-rasterization`, `simple-knn`, `fused-ssim` | training / rendering / metrics | `git clone https://github.com/graphdeco-inria/gaussian-splatting` at install | **Gaussian-Splatting License — non-commercial research and evaluation use only**; commercial use requires permission from Inria / MPII. `fused-ssim` has its own licence (see its repository). |
| **COLMAP** (Schönberger et al.) | SfM, undistortion | GitHub releases (Windows CUDA build) or existing install | BSD 3-Clause ("new BSD"); bundled third-party components as listed by COLMAP |
| **FFmpeg** | decoding / frame extraction | gyan.dev or BtbN builds, or existing install | LGPL 2.1+ / GPL (these Windows builds are GPL-enabled) |
| **PyTorch / torchvision** | GPU training (GS environment) | download.pytorch.org | BSD-style |
| **PySide6 (Qt for Python)** | desktop GUI | PyPI | LGPL v3 |
| NumPy, pandas, Matplotlib, OpenCV, Pillow, psutil, plyfile, tqdm, lpips | processing | PyPI | BSD / PSF-based / Apache-2.0 / HPND / BSD |

## Code attribution
* `web/viewer.html` — the splat renderer is an independent implementation of the approach used by
  **antimatter15/splat** by Kevin Kwok (MIT licence): Gaussian data packed into float textures, screen-space quads
  from the eigen-decomposition of the projected covariance, and a 16-bit counting sort by depth in a Web Worker.
  GeoSplat's version adds an exact Jacobian for an arbitrary model-view-projection and a georeferenced satellite
  ground. Copyright notice of the original approach: Copyright (c) 2023 Kevin Kwok, MIT licence.
* SH basis constants and sign conventions in `geosplat/gaussians.py` and `web/viewer.html` match the Graphdeco
  implementation (`utils/sh_utils.py`) so that exported Gaussians stay compatible. These are mathematical constants of
  the real spherical-harmonic basis.
* The DJI `djmd` field naming follows ExifTool's DJI tag documentation (Phil Harvey). The decoder itself is GeoSplat
  code.

## Imagery and map services (viewer)
* **Google Map Tiles API** (satellite 2D tiles): use is subject to the Google Maps Platform Terms of Service; the viewer
  shows the required Google attribution and copyright string. A user-supplied API key is stored only locally
  (`runtime/settings.json`, `08_Exports/web/config.local.js`) and is never committed.
* **Esri World Imagery** (fallback): subject to Esri's terms of use; attribution "Esri, Maxar, Earthstar Geographics"
  is shown.

## Trained models
Gaussian models are produced by the Graphdeco software. Check its licence before any commercial use of trained models
or renders.
