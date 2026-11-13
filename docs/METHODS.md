# GeoSplat — Methods and References

This document describes every processing step in GeoSplat: the equations it implements, the design choices behind them, and the published work each method comes from. GeoSplat **orchestrates** established tools (FFmpeg, COLMAP, Graphdeco 3D Gaussian Splatting). The geodesy, telemetry decoding, synchronisation, georeferencing and export code is GeoSplat's own implementation of the published methods cited below. Third-party software is **not** bundled; see `THIRD_PARTY_NOTICES.md`.

Notation: column vectors; **R** is a rotation matrix; *s* is a scale; **t** is a translation; ENU means East-North-Up.

---

## 1. Telemetry decoding

### 1.1 DJI SRT subtitles
DJI writes one subtitle cue per video frame. Each cue holds `FrameCnt`, `DiffTime`, a local timestamp and bracketed `key: value` pairs (`latitude`, `longitude`, `rel_alt`, `abs_alt`, `gb_yaw`, `gb_pitch`, `gb_roll`, exposure). GeoSplat parses these with regular expressions. It also understands the older `GPS(lon, lat, alt)` / `BAROMETER` format. DJI does not publish a specification for this format; the parser is based on the files themselves.

### 1.2 Embedded DJI metadata track (`djmd`)
Recent DJI aircraft embed a timed data track in the MP4, with handler `DJI meta` and codec tag `djmd`. Each packet is a Protocol Buffers message [Google, Protocol Buffers encoding]. GeoSplat:

* gets packet byte offsets and sizes from `ffprobe` and reads each packet with a file seek, so the video is never loaded into memory;
* decodes the protobuf wire format without a schema (varint / fixed64 / length-delimited / fixed32);
* maps field paths to quantities. The naming follows ExifTool's DJI protobuf module [Harvey, ExifTool], and the mapping was **verified empirically for `dvtm_wm265e.proto` (Mavic 3 Enterprise)**:
  * lat/lon are float64 **radians**;
  * absolute altitude is in mm;
  * relative altitude is a float in mm;
  * attitude angles are int **deci-degrees**;
  * the velocity N/E fields were confirmed by correlating them with GNSS-derived velocity (r = 0.999).

  For other proto versions, positions are used only if they pass plausibility checks.

Proto3 omits fields whose value is zero, so an absent angle or velocity field is read as 0.

## 2. Frame extraction
Frames are decoded with FFmpeg [FFmpeg Developers]. GeoSplat tries hardware decoding (CUDA/D3D11VA/DXVA2/VAAPI) and falls back to software decoding. A frame is selected when its presentation time *t_k* is nearest to the target time *k/f_out*; selection uses the exact container PTS (`select=lt(abs(t-T_k), ½Δt)`). Output files are named after the source frame index and the PTS. Interrupted runs resume from the last complete frame.

## 3. Telemetry ↔ image synchronisation
The GNSS position refreshes at about 10 Hz, while video runs at about 30 fps, so consecutive records repeat the same position. GeoSplat therefore:

1. reduces each channel to its **update epochs**, i.e. the records where the value changes;
2. interpolates positions **linearly in a local ENU frame** (§4.3), not in degrees;
3. interpolates angles through their unwrapped form;
4. flags each frame as GOOD / DEGRADED / BAD depending on the gap between the bracketing epochs.

Frames outside the telemetry span are rejected instead of being extrapolated. The query time is *t_tel = t_frame + Δt*, and Δt is estimated in §6.2.

## 4. Geodesy (WGS 84)
Ellipsoid: *a* = 6 378 137 m, *f* = 1/298.257 223 563 [NIMA TR8350.2].

### 4.1 Geodetic → ECEF
*N(φ) = a / √(1 − e² sin²φ)*,
*X = (N+h) cosφ cosλ*, *Y = (N+h) cosφ sinλ*, *Z = (N(1−e²)+h) sinφ*.

### 4.2 ECEF → geodetic
Bowring's parametric starting value [Bowring 1976], followed by three fixed-point refinements of φ. Round-trip error is below 10⁻⁸ m in the unit tests.

### 4.3 Local ENU
**x**_ENU = **M**ᵀ(**X** − **X**₀). The columns of **M** are the East, North and Up unit vectors at (φ₀, λ₀):
E = (−sinλ₀, cosλ₀, 0), N = (−sinφ₀cosλ₀, −sinφ₀sinλ₀, cosφ₀), U = (cosφ₀cosλ₀, cosφ₀sinλ₀, sinφ₀).

### 4.4 UTM
Transverse Mercator using Krüger's series to order *n⁴* [Krüger 1912], as given by Karney [2011]; *k₀* = 0.9996, with a false northing of 10 000 km in the southern hemisphere. Round-trip error is below 10⁻⁹ degrees inside a zone.

### 4.5 Web Mercator tiles (viewer)
Tile *(z, x, y)* corner longitude is *λ = 360·x/2ᶻ − 180*, and latitude is *φ = atan(sinh(π(1 − 2y/2ᶻ)))* [EPSG:3857]. The tile corners are converted to ENU (§4.3), so the satellite imagery and the model share one metric frame.

## 5. Structure-from-Motion (COLMAP)
COLMAP [Schönberger & Frahm 2016; Schönberger et al. 2016] performs:

* SIFT features [Lowe 2004], GPU implementation [Wu 2007];
* matching of a **GNSS-guided pair list**. Two images are paired if |i − j| ≤ w (temporal window), **or** if their camera viewing azimuths differ by no more than θ and the points where their optical axes hit the ground lie within *d* of each other. The ground point is **C** + **v**·h/sin(−pitch), using the gimbal angles. This rule captures loop closure on orbits at about 35 % of the cost of exhaustive matching;
* geometric verification, incremental mapping and bundle adjustment;
* a shared OPENCV camera model. The focal prior comes from the 35 mm-equivalent focal length, *f_px = f₃₅/43.27 mm · √(W²+H²)*, and is refined by bundle adjustment.

## 6. GNSS georeferencing

### 6.1 Robust similarity transform
The camera centres from SfM are **C**ᵢ = −**R**ᵢᵀ**t**ᵢ. GeoSplat finds the similarity **x**_ENU = s**R**·**C** + **t** that best maps them onto the GNSS positions **G**ᵢ, using the closed-form least-squares solution of Umeyama [1991] (SVD with a reflection guard). It is wrapped in RANSAC [Fischler & Bolles 1981] with 3-point minimal samples, an inlier threshold in metres, and a least-squares refit on the inliers.

### 6.2 Time-offset estimation
For each candidate Δt in a grid (default ±0.5 s, 10 ms steps), GNSS positions are re-interpolated at *t+Δt* and the Sim3 is refitted on a fixed inlier set. The minimum of RMSE(Δt) is refined parabolically. An offset is applied only if it improves the RMSE by more than 5 % and more than 2 cm. On a circular orbit a constant shift partly maps into a rotation of the Sim3; the offset is still observable because the flight speed varies.

### 6.3 Accuracy assessment
* Residuals per axis (E, N, U), RMSE, 95th percentile.
* **5-fold check-point cross-validation**: the Sim3 is fitted on 4/5 of the cameras and evaluated on the held-out fifth.
* Camera heading compared with gimbal yaw, which estimates the compass bias.

These residuals measure **relative** consistency between SfM and GNSS. A common GNSS bias (metre level without RTK) cannot be observed this way and needs ground control points.

### 6.4 Coordinate chain
COLMAP-local → ENU (Sim3) → ECEF (rigid, §4.3) → WGS 84 → UTM. The local→ECEF composite is still a Sim3: *s*, **M R**, **M t** + **X**₀. The model is also written out as a COLMAP model in metric ENU. Every image observation reprojects to identical pixels before and after the transform (this is checked).

### 6.5 Quaternion extraction
Rotation matrix → quaternion uses the symmetric 4×4 matrix eigenvector method [Bar-Itzhack 2000], which is robust near 180° rotations.

## 7. 3D Gaussian Splatting
Training uses the official implementation of Kerbl et al. [2023], cloned at install time from its repository; it is not redistributed. Training runs directly on the **metric ENU** COLMAP model after undistortion to PINHOLE cameras (COLMAP `image_undistorter`). The trained Gaussians are therefore georeferenced by construction. GeoSplat does not simply attach GPS metadata to a model trained in an arbitrary frame. The trainer's SSIM loss uses the fused-SSIM kernel [Mallick et al. 2024]. Evaluation on held-out views (every 8th image) reports PSNR, SSIM [Wang et al. 2004] and LPIPS [Zhang et al. 2018].

### 7.1 Transforming Gaussians consistently
Applying a similarity *x' = s**R**x + **t*** to a Gaussian (μ, Σ = **R**_g **S**² **R**_gᵀ, spherical-harmonic colour **c**):

* μ' = s**R**μ + **t**
* Σ' = s²**R**Σ**R**ᵀ, which gives rotation quaternion q' = q_R ⊗ q_g and log-scales + ln *s*
* SH colour must satisfy **c**'(**d**') = **c**(**R**ᵀ**d**') for every viewing direction. Each SH band *l* is rotated by a (2*l*+1)×(2*l*+1) matrix **D**_l. GeoSplat obtains **D**_l by least squares on the band-limited basis: it evaluates **Y**_l(**d**) and **Y**_l(**R**ᵀ**d**) on random directions and solves **Y**_l(**d**)·**D**_l = **Y**_l(**R**ᵀ**d**). This is exact for band-limited functions and equivalent to analytic real-SH rotation [Ivanic & Ruedenberg 1996; Green 2003]. The basis constants and signs follow the Graphdeco implementation.

Unit tests check covariance, view-dependent colour and the round-trip error.

### 7.2 UTM export
UTM is conformal but not Euclidean: it has a point scale *k* and a grid convergence γ. Gaussians can only be mapped consistently by a similarity, so GeoSplat fits one ENU→UTM similarity over the core scene and reports its misfit (typically below 2 cm over a few hundred metres). Cameras and points also get the exact projection. A false origin is subtracted from the PLY coordinates, and recorded in the sidecar JSON, to keep float32 precision.

## 8. Orthophoto from Gaussians
For a top-down orthographic view, the projected covariance is the xy block of Σ. A Gaussian covers pixel offset **d** with α = o·exp(−½**d**ᵀΣ_xy⁻¹**d**). Each pixel takes the colour of the highest Gaussian with α ≥ 0.25 (two-pass z-buffer). The orthophoto is written with a world file (EPSG:4326).

## 9. Web viewer
The WebGL2 viewer renders each Gaussian as a screen-space ellipse. The projected 2-D covariance uses the EWA splatting approximation [Zwicker et al. 2001] as used by Kerbl et al. [2023]: the Jacobian of the full model-view-projection, *J = (r₀w − c_x r₃, r₁w − c_y r₃)/w²*, gives Σ₂D = JΣJᵀ. Quads are sized from the eigen-decomposition of Σ₂D and are blended back-to-front after a 16-bit counting sort on view depth in a Web Worker. **The structure of this renderer (data textures, eigen-axis quads, worker-side counting sort) follows the open-source viewer by Kevin Kwok, `antimatter15/splat` (MIT licence); GeoSplat's renderer is an independent implementation of that approach**, extended with a general-MVP Jacobian and a georeferenced satellite ground. Satellite imagery comes from the Google Map Tiles API (2D tiles) with the required attribution, or from Esri World Imagery as a fallback.

---

## 10. Vertical datum correction from GNSS checkpoints (stage 6, optional)
**Why.** On DJI aircraft the per-frame `abs_alt` equals a home-point altitude fixed at take-off plus `rel_alt`
(stage 6 checks this: `abs_alt − rel_alt` is constant to millimetres in the ITS data set). An error in the
home-point altitude therefore shifts the whole model vertically by one constant. The Sim3 of §6 absorbs such a
shift, so the internal residuals cannot reveal it. In the ITS Research Center data set the offset was +25.42 m.

**Checkpoint harmonisation (optional).** RTK measures the baseline from the reference coordinate the base
broadcasts. If that differs from the station's official datum coordinate, every rover coordinate is re-referenced
in ECEF: **X**ᵣ = **X**ᵣ,RTK − **X**_base,used + **X**_base,official. Coordinates can be moved between epochs with a
constant station velocity: **X**(t₂) = **X**(t₁) + **V**(t₂ − t₁) [Altamimi et al. 2011 for the ITRF2008 frame]. Both steps are read from a JSON file next to the checkpoint CSV, or from command-line options.

**Model surface height.** For each checkpoint, Gaussians within 2.5 m horizontally (4 m if fewer than five) with
opacity σ(o) > 0.3 are selected. The surface height is the mode of their opacity-weighted Gaussian kernel density
(bandwidth 0.15 m), which suppresses floaters. Without Gaussians the sparse SfM points are used with equal weights.

**Offset and accuracy.** ΔZᵢ = Z_model − h_checkpoint. Points are classed as non-vegetated (NVA) or vegetated (VVA)
by the user; unlabelled points are NVA candidates. Points more than 3 m above the provisional (median) offset are
excluded as canopy tops, points without model support as "no coverage". The offset *b* is the mean ΔZ of the NVA
class. Following the ASPRS Positional Accuracy Standards, Edition 2 [ASPRS 2024], the report gives mean error,
standard deviation, RMSE without correction, and RMSE after correction. The NVA RMSE uses leave-one-out residuals
eᵢ = ΔZᵢ − mean(ΔZⱼ, j ≠ i), so no point is tested against an offset fitted to itself; VVA points are corrected
with the NVA offset. The checkpoint survey error is added as RMSE_V = √(RMSE_data² + RMSE_survey²) when VRMS values
are given. Robust measures (median, NMAD = 1.4826 · median|ΔZ − median ΔZ|) follow [Höhle & Höhle 2009]; 95 %
intervals come from 10 000 bootstrap resamples. A datum-free check compares height differences of all NVA pairs,
eᵢⱼ = (Zᵢ − Zⱼ) − (hᵢ − hⱼ), in which any constant offset cancels. With fewer than 30 NVA checkpoints the result is
labelled a limited assessment.

**Output.** `10_Vertical_Correction/` receives height-corrected copies: UTM products (sparse points, camera centres,
Gaussians) with *z* + (−*b*); local-ENU products with unchanged values and the origin height in the header changed to
*h*₀ − *b*; camera trajectory (*h* and ECEF); CRS sidecar and viewer metadata. Horizontal coordinates are not
changed. Originals are never modified.

## References

* Altamimi, Z., Collilieux, X., & Métivier, L. (2011). ITRF2008: an improved solution of the international terrestrial reference frame. *Journal of Geodesy*, 85(8), 457–473.
* ASPRS (2024). *ASPRS Positional Accuracy Standards for Digital Geospatial Data*, Edition 2, Version 2.0. American Society for Photogrammetry and Remote Sensing.
* Bar-Itzhack, I. Y. (2000). New method for extracting the quaternion from a rotation matrix. *Journal of Guidance, Control, and Dynamics*, 23(6), 1085–1087.
* Bowring, B. R. (1976). Transformation from spatial to geographical coordinates. *Survey Review*, 23(181), 323–327.
* EPSG Geodetic Parameter Dataset. EPSG:3857 (WGS 84 / Pseudo-Mercator), EPSG:4326, EPSG:4978, EPSG:326xx/327xx (UTM).
* FFmpeg Developers. *FFmpeg*. https://ffmpeg.org
* Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381–395.
* Google. Protocol Buffers — Encoding. https://protobuf.dev/programming-guides/encoding/
* Google. Map Tiles API — 2D Tiles. https://developers.google.com/maps/documentation/tile
* Green, R. (2003). Spherical Harmonic Lighting: The Gritty Details. *Game Developers Conference*.
* Harvey, P. *ExifTool* — DJI tags (protobuf `dvtm_*`). https://exiftool.org/TagNames/DJI.html
* Höhle, J., & Höhle, M. (2009). Accuracy assessment of digital elevation models by means of robust statistical methods. *ISPRS Journal of Photogrammetry and Remote Sensing*, 64(4), 398–406.
* Ivanic, J., & Ruedenberg, K. (1996). Rotation matrices for real spherical harmonics. Direct determination by recursion. *Journal of Physical Chemistry*, 100(15), 6342–6347.
* Karney, C. F. F. (2011). Transverse Mercator with an accuracy of a few nanometers. *Journal of Geodesy*, 85(8), 475–485.
* Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics*, 42(4), Article 139.
* Krüger, L. (1912). *Konforme Abbildung des Erdellipsoids in der Ebene*. Royal Prussian Geodetic Institute, New Series 52.
* Kwok, K. *splat* — WebGL 3D Gaussian Splat viewer. https://github.com/antimatter15/splat (MIT licence).
* Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
* Mallick, S. S., Goel, R., Kerbl, B., Steinberger, M., Vicente Carrasco, F., & De La Torre, F. (2024). Taming 3DGS: High-Quality Radiance Fields with Limited Resources. *SIGGRAPH Asia 2024 Conference Papers*.
* NIMA (2000). *Department of Defense World Geodetic System 1984*, Technical Report TR8350.2, 3rd ed.
* Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-Motion Revisited. *IEEE CVPR*.
* Schönberger, J. L., Zheng, E., Pollefeys, M., & Frahm, J.-M. (2016). Pixelwise View Selection for Unstructured Multi-View Stereo. *ECCV*.
* Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(4), 376–380.
* Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.
* Wu, C. (2007). SiftGPU: A GPU implementation of Scale Invariant Feature Transform (SIFT). https://github.com/pitzer/SiftGPU
* Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *IEEE CVPR*.
* Zwicker, M., Pfister, H., van Baar, J., & Gross, M. (2001). EWA Volume Splatting. *IEEE Visualization 2001*, 29–36.
