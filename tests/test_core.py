"""Unit tests for GeoSplat's own numerical code (run: python -m pytest tests)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geosplat import gaussians, geodesy, georef, sim3, telemetry_parsers  # noqa: E402
from geosplat.colmap_io import qvec2rotmat  # noqa: E402


def test_geodesy_roundtrips():
    r = geodesy.self_test()
    assert r["ecef_roundtrip_h_m"] < 1e-6
    assert r["ecef_roundtrip_lat_deg"] < 1e-10
    assert r["utm_roundtrip_deg"] < 1e-8
    assert abs(r["utm_cm_check_E"] - 500000.0) < 1e-6
    assert r["enu_roundtrip_h_m"] < 1e-6


def test_utm_reference_point():
    # Central meridian of zone 49 at the equator: E = 500 000 m exactly; north of equator N = 0
    E, N, z, s = geodesy.latlon_to_utm(np.array([0.0]), np.array([111.0]), 49, False)
    assert abs(E[0] - 500000) < 1e-6 and abs(N[0]) < 1e-6


def test_sim3_ransac_recovers_transform_and_outliers():
    r = sim3.self_test()
    assert r["scale_err"] < 0.01 and r["rot_err_deg"] < 0.1 and r["outliers_found"] == 15


def test_quaternion_roundtrip_including_180deg():
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = qvec2rotmat(q)
        assert np.abs(qvec2rotmat(georef.rotmat2qvec(R)) - R).max() < 1e-9
    for R in (np.diag([1, -1, -1.0]), np.diag([-1, 1, -1.0]), np.diag([-1, -1, 1.0])):
        assert np.abs(qvec2rotmat(georef.rotmat2qvec(R)) - R).max() < 1e-9


def test_gaussian_similarity_transform_exact():
    r = gaussians.self_test()
    assert r["cov_rel_err"] < 1e-5          # float32 storage
    assert r["sh_colour_err"] < 1e-5        # view-dependent colour preserved under rotation
    assert r["roundtrip_err"] < 1e-4


def test_protobuf_decoder():
    # field 1 varint 150, field 2 string "hi", field 3 fixed64 double 1.5
    import struct
    msg = bytes([0x08, 0x96, 0x01, 0x12, 0x02]) + b"hi" + bytes([0x19]) + struct.pack("<d", 1.5)
    out = telemetry_parsers.pb_parse(msg)
    assert out[0] == (1, 0, 150) and out[2] == (3, 1, 1.5)


def test_srt_parsers(tmp_path):
    new = tmp_path / "new.srt"
    new.write_text("1\n00:00:00,000 --> 00:00:00,033\n<font size=\"28\">FrameCnt: 1, DiffTime: 33ms\n"
                   "2025-06-12 17:01:23.863\n[iso: 100] [shutter: 1/128.0] [latitude: -7.281648] "
                   "[longitude: 112.798208] [rel_alt: 130.282 abs_alt: 194.479] [gb_yaw: -60.6 gb_pitch: -41.9 gb_roll: 0.0] </font>\n\n")
    d = telemetry_parsers.parse_srt(str(new))
    assert abs(d["latitude"][0] + 7.281648) < 1e-9 and abs(d["abs_alt"][0] - 194.479) < 1e-9 and d["frame_cnt"][0] == 1
    old = tmp_path / "old.srt"
    old.write_text("1\n00:00:00,000 --> 00:00:01,000\nGPS(112.7951,-7.2817,19) BAROMETER:120.5m\n\n")
    d = telemetry_parsers.parse_srt(str(old))
    assert abs(d["latitude"][0] + 7.2817) < 1e-9 and abs(d["longitude"][0] - 112.7951) < 1e-9 and d["rel_alt"][0] == 120.5


def test_colmap_model_transform_reprojection_invariant(tmp_path):
    """Writing a COLMAP TXT model through a Sim3 must not change any reprojection."""
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    rng = np.random.default_rng(2)
    (src / "cameras.txt").write_text("1 PINHOLE 1000 800 900 900 500 400\n")
    P = rng.normal(0, 5, (50, 3)) + [0, 0, 30]
    lines = []
    for i in range(3):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q[0] = abs(q[0])
        t = rng.normal(0, 1, 3)
        lines.append(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 img{i}.jpg\n\n")
    (src / "images.txt").write_text("# images\n" + "".join(lines))
    (src / "points3D.txt").write_text("".join(f"{k} {p[0]} {p[1]} {p[2]} 1 2 3 0.5 1 0\n" for k, p in enumerate(P)))
    a = 0.8
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    s, t = 12.5, np.array([100.0, -50.0, 7.0])
    georef.transform_colmap_txt(str(src), str(dst), s, R, t)
    from geosplat.colmap_io import read_images_txt
    i1, i2 = read_images_txt(str(src / "images.txt")), read_images_txt(str(dst / "images.txt"))
    for k in i1:
        x1 = (i1[k]["R"] @ P.T).T + i1[k]["tvec"]
        x2 = (i2[k]["R"] @ sim3.apply_sim3(s, R, t, P).T).T + i2[k]["tvec"]
        assert np.abs(x1[:, :2] / x1[:, 2:] - x2[:, :2] / x2[:, 2:]).max() < 1e-9
