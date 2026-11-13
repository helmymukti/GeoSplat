"""Unit tests for the vertical datum correction (geosplat/vcorr.py, scripts/stage6_vcorr.py)."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from geosplat import gaussians, geodesy, georef, vcorr  # noqa: E402

LAT0, LON0, H0 = -7.2816475, 112.7982082, 194.478
ZONE, SOUTH = 49, True


def test_self_test():
    assert vcorr.self_test()


def test_loo_residuals_match_explicit():
    rng = np.random.default_rng(1)
    d = rng.normal(25.4, 0.2, 9)
    explicit = np.array([d[i] - np.delete(d, i).mean() for i in range(len(d))])
    assert np.allclose(vcorr.loo_residuals(d), explicit)


def test_ntrip_base_correction_is_ecef_translation():
    used = np.array([-2443857.719, 5835257.946, -808826.478])
    official = np.array([-2443857.505, 5835258.077, -808826.379])
    lat, lon, h = np.array([-7.28]), np.array([112.80]), np.array([31.7])
    la, lo, hh = vcorr.ntrip_base_correction(lat, lon, h, used, official)
    x0 = np.array(geodesy.geodetic_to_ecef(lat, lon, h)).ravel()
    x1 = np.array(geodesy.geodetic_to_ecef(la, lo, hh)).ravel()
    assert np.allclose(x1 - x0, official - used, atol=1e-6)


def test_epoch_shift_and_decimal_year():
    v = np.array([-0.024, -0.014, -0.011])
    lat, lon, h = np.array([-7.28]), np.array([112.80]), np.array([31.7])
    la, lo, hh = vcorr.epoch_shift(lat, lon, h, v, 2012.0, 2022.0)
    d = np.array(geodesy.geodetic_to_ecef(la, lo, hh)).ravel() - np.array(geodesy.geodetic_to_ecef(lat, lon, h)).ravel()
    assert np.allclose(d, v * 10, atol=1e-6)
    assert abs(vcorr.decimal_year("2025-06-12") - 2025.445) < 0.001


def test_assess_classes_offset_and_exclusions():
    dz = np.array([25.3, 25.5, 25.4, 25.45, 25.35, 30.9, np.nan, 25.0])
    cls = ["NVA", "NVA", "", "NVA", "", "", "", "VVA"]
    final, status, st = vcorr.assess(dz, cls)
    assert final[5] == vcorr.EXCLUDED and status[5] == "canopy top"
    assert final[6] == vcorr.EXCLUDED and status[6] == "no model coverage"
    assert st["NVA"]["n"] == 5 and abs(st["offset_m"] - 25.4) < 1e-9
    assert abs(st["VVA"]["ME_after_correction_m"] - (-0.4)) < 1e-9
    assert any("ASPRS" in w for w in st["warnings"])


def test_pairwise_errors_ignore_constant_offset():
    h = np.array([31.7, 32.8, 31.5, 32.0])
    assert np.allclose(vcorr.pairwise_errors(h + 25.42, h), 0)


def _project(tmp):
    """Minimal synthetic GeoSplat project: flat ground h = 31.7 seen 25.42 m too high."""
    P = os.path.join(tmp, "proj")
    G, X, L, T = (os.path.join(P, d) for d in ("06_Georeferencing", "08_Exports", "logs", "02_Telemetry"))
    for d in (G, X, os.path.join(X, "web"), L, T):
        os.makedirs(d)
    off = 25.42
    E0, N0 = (float(v[0]) for v in geodesy.latlon_to_utm(np.array([LAT0]), np.array([LON0]), ZONE, SOUTH)[:2])
    rng = np.random.default_rng(3)
    n = 30000
    E = E0 + rng.uniform(-100, 100, n)
    N = N0 + rng.uniform(-100, 100, n)
    Z = 31.7 + off + rng.normal(0, 0.05, n)
    georef.write_ply(os.path.join(G, "sparse_points_utm.ply"), np.column_stack([E, N, Z]),
                     np.zeros((n, 3), np.uint8), comments=["EPSG:32749 E N + DJI abs_alt-referenced height h (datum unconfirmed)"])
    georef.write_ply(os.path.join(G, "camera_centres_utm.ply"), np.array([[E0, N0, H0]]), np.zeros((1, 3), np.uint8),
                     comments=["EPSG:32749"])
    georef.write_ply(os.path.join(G, "sparse_points_enu.ply"), np.column_stack([E - E0, N - N0, Z - H0]),
                     np.zeros((n, 3), np.uint8), comments=[f"WGS84 local ENU origin lat {LAT0:.9f} lon {LON0:.9f} h {H0:.3f}"])
    dt = np.dtype([(k, "<f4") for k in ("x", "y", "z", "opacity")])
    gsd = np.zeros(n, dt)
    gsd["x"], gsd["y"], gsd["z"], gsd["opacity"] = E - 698000, N - 9194000, Z, 3.0
    gaussians.write_gs_ply(os.path.join(X, "gaussians_utm_epsg32749_offset.ply"), gsd, ["test"])
    json.dump({"files": {"gaussians_utm_epsg32749_offset.ply": {"add_offset_to_get_utm": [698000.0, 9194000.0, 0.0],
                                                                "from_enu": {"t": [0.0, 0.0, H0]}}}},
              open(os.path.join(X, "gaussians_crs.json"), "w"))
    json.dump({"enu_origin": {"lat_deg": LAT0, "lon_deg": LON0, "h_m": H0},
               "conventions": {"utm": "WGS 84 / UTM zone 49S (EPSG:32749)"}}, open(os.path.join(G, "transform.json"), "w"))
    json.dump({"zone": ZONE, "south": SOUTH, "epsg": 32749}, open(os.path.join(L, "_state.json"), "w"))
    with open(os.path.join(T, "telemetry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["srt_datetime", "abs_alt", "rel_alt"])
        for i in range(50):
            w.writerow(["2025-06-12 17:01:23", 194.4 + 0.01 * i, 130.2 + 0.01 * i])
    cp = os.path.join(tmp, "cp.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Easting", "Northing", "Height", "Class", "VRMS"])
        for i, (de, dn) in enumerate([(-50, -50), (50, -40), (-40, 60), (30, 30), (0, -70), (70, 70)]):
            w.writerow([i + 1, E0 + de, N0 + dn, 31.7, "NVA", 0.02])
    return P, cp, off


def test_stage6_end_to_end(tmp_path):
    import stage6_vcorr
    P, cp, off = _project(str(tmp_path))
    assert stage6_vcorr.main(["--project", P, "--checkpoints", cp]) == 0
    O = os.path.join(P, vcorr.OUT_DIR)
    rep = json.load(open(os.path.join(O, "vcorr_report.json")))
    assert abs(rep["statistics"]["offset_m"] - off) < 0.03
    assert rep["altitude_reference"]["home_point_referenced"] is True
    src, _ = gaussians.read_gs_ply(os.path.join(P, "08_Exports", "gaussians_utm_epsg32749_offset.ply"))
    cor, com = gaussians.read_gs_ply(os.path.join(O, "gaussians_utm_epsg32749_offset_hcorr.ply"))
    assert np.allclose(cor["z"] - src["z"], -rep["statistics"]["offset_m"], atol=1e-4)
    assert np.array_equal(cor["x"], src["x"]) and any("vertical datum corrected" in c for c in com)
    sp, com = gaussians.read_gs_ply(os.path.join(O, "sparse_points_utm_hcorr.ply"))
    assert abs(np.median(sp["z"]) - 31.7) < 0.03 and not any("abs_alt-referenced" in c for c in com)
    enu0 = open(os.path.join(P, "06_Georeferencing", "sparse_points_enu.ply"), "rb").read()
    enu1 = open(os.path.join(O, "sparse_points_enu_hcorr.ply"), "rb").read()
    k0, k1 = enu0.index(b"end_header\n") + 11, enu1.index(b"end_header\n") + 11
    assert enu0[k0:] == enu1[k1:] and b"h 194.478" not in enu1[:k1]
    assert os.path.exists(os.path.join(O, "README.txt")) and os.path.exists(os.path.join(O, "checkpoints_dz.csv"))


def test_stage6_without_checkpoints_only_diagnoses(tmp_path):
    import stage6_vcorr
    P, _, _ = _project(str(tmp_path))
    assert stage6_vcorr.main(["--project", P]) == 0
    O = os.path.join(P, vcorr.OUT_DIR)
    assert sorted(os.listdir(O)) == ["vcorr_report.json"]


def test_negative_triplets_are_accepted():
    import stage6_vcorr
    out = stage6_vcorr._join_negative_triplets(["--velocity", "-0.024,-0.014,-0.011", "--epoch-from", "2012.0"])
    assert out == ["--velocity=-0.024,-0.014,-0.011", "--epoch-from", "2012.0"]


def test_stage6_sidecar_json_applies_epoch_shift(tmp_path):
    import stage6_vcorr
    P, cp, off = _project(str(tmp_path))
    json.dump({"velocity": [0.0, 0.0, -0.1], "epoch_from": 2015.0, "epoch_to": 2025.0},
              open(os.path.splitext(cp)[0] + ".json", "w"))
    assert stage6_vcorr.main(["--project", P, "--checkpoints", cp, "--no-write"]) == 0
    rep = json.load(open(os.path.join(P, vcorr.OUT_DIR, "vcorr_report.json")))
    assert rep["harmonisation_settings_file"] == "cp.json"
    assert rep["checkpoint_harmonisation"][0]["epoch_to"] == 2025.0
    # 1 m of ECEF Z motion changes h by about sin(lat) m; the checkpoint heights and hence the offset follow it
    _, _, h1 = vcorr.epoch_shift(np.array([LAT0]), np.array([LON0]), np.array([31.7]), [0.0, 0.0, -0.1], 2015.0, 2025.0)
    assert abs(rep["statistics"]["offset_m"] - (off - (h1[0] - 31.7))) < 0.03
