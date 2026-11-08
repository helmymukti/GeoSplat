"""
GeoSplat geodesy core (no external deps beyond NumPy).

Conventions
-----------
* Datum / ellipsoid : WGS84 (a = 6378137.0 m, f = 1/298.257223563)
* Geodetic          : latitude, longitude in DEGREES, height h in metres (ellipsoidal,
                      unless the caller documents otherwise)
* ECEF              : X, Y, Z in metres (Earth-Centred, Earth-Fixed, WGS84)
* Local ENU         : East, North, Up in metres, tangent plane at (lat0, lon0, h0)
* UTM               : Transverse Mercator, k0 = 0.9996, Kruger series to n^4
                      (sub-millimetre within a zone). Southern hemisphere uses
                      false northing 10 000 000 m.
References: NIMA TR8350.2 (WGS 84); Bowring (1976) Survey Review 23(181); Krueger (1912);
Karney (2011) J. Geodesy 85(8) - Transverse Mercator series to n^4. See docs/METHODS.md section 4.
"""
from __future__ import annotations

import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
B = A * (1.0 - F)
EP2 = (A * A - B * B) / (B * B)


# --------------------------------------------------------------------------- ECEF
def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    sl, cl = np.sin(lat), np.cos(lat)
    N = A / np.sqrt(1.0 - E2 * sl * sl)
    x = (N + h) * cl * np.cos(lon)
    y = (N + h) * cl * np.sin(lon)
    z = (N * (1.0 - E2) + h) * sl
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """Bowring initial value + 3 Newton refinements (sub-mm)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    th = np.arctan2(z * A, p * B)
    lat = np.arctan2(z + EP2 * B * np.sin(th) ** 3, p - E2 * A * np.cos(th) ** 3)
    for _ in range(3):
        N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - E2 * N / (N + h)))
    N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


# --------------------------------------------------------------------------- ENU
def _enu_matrix(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    # rows = E, N, U unit vectors expressed in ECEF
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    R = _enu_matrix(lat0, lon0)
    d = np.vstack([np.ravel(x) - x0, np.ravel(y) - y0, np.ravel(z) - z0])
    e, n, u = R @ d
    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, h0):
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    R = _enu_matrix(lat0, lon0)
    d = R.T @ np.vstack([np.ravel(e), np.ravel(n), np.ravel(u)])
    return d[0] + x0, d[1] + y0, d[2] + z0


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0):
    return ecef_to_enu(*geodetic_to_ecef(lat, lon, h), lat0, lon0, h0)


def enu_to_geodetic(e, n, u, lat0, lon0, h0):
    return ecef_to_geodetic(*enu_to_ecef(e, n, u, lat0, lon0, h0))


# --------------------------------------------------------------------------- UTM
_n = F / (2.0 - F)
_AA = A / (1.0 + _n) * (1.0 + _n ** 2 / 4.0 + _n ** 4 / 64.0)
_ALPHA = [
    _n / 2 - 2 / 3 * _n ** 2 + 5 / 16 * _n ** 3 + 41 / 180 * _n ** 4,
    13 / 48 * _n ** 2 - 3 / 5 * _n ** 3 + 557 / 1440 * _n ** 4,
    61 / 240 * _n ** 3 - 103 / 140 * _n ** 4,
    49561 / 161280 * _n ** 4,
]
_BETA = [
    _n / 2 - 2 / 3 * _n ** 2 + 37 / 96 * _n ** 3 - 1 / 360 * _n ** 4,
    1 / 48 * _n ** 2 + 1 / 15 * _n ** 3 - 437 / 1440 * _n ** 4,
    17 / 480 * _n ** 3 - 37 / 840 * _n ** 4,
    4397 / 161280 * _n ** 4,
]
_DELTA = [
    2 * _n - 2 / 3 * _n ** 2 - 2 * _n ** 3,
    7 / 3 * _n ** 2 - 8 / 5 * _n ** 3,
    56 / 15 * _n ** 3,
]
K0 = 0.9996


def utm_zone(lon):
    return int(np.floor((float(np.mean(lon)) + 180.0) / 6.0) + 1)


def utm_epsg(zone, south):
    return (32700 if south else 32600) + zone


def latlon_to_utm(lat, lon, zone=None, south=None):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if zone is None:
        zone = utm_zone(lon)
    if south is None:
        south = bool(np.mean(lat) < 0)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    phi = np.radians(lat)
    lam = np.radians(lon) - lon0
    c = 2 * np.sqrt(_n) / (1 + _n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    E = 500000.0 + K0 * _AA * eta
    N = K0 * _AA * xi + (10000000.0 if south else 0.0)
    return E, N, zone, south


def utm_to_latlon(E, N, zone, south):
    E = np.asarray(E, float)
    N = np.asarray(N, float)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    xi = (N - (10000000.0 if south else 0.0)) / (K0 * _AA)
    eta = (E - 500000.0) / (K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, d in enumerate(_DELTA, start=1):
        phi = phi + d * np.sin(2 * j * chi)
    lam = lon0 + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(phi), np.degrees(lam)


# --------------------------------------------------------------------------- angles
def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def interp_angle_deg(tq, t, a):
    """Linear interpolation of an angle series in degrees through its unwrapped form."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return np.full(np.shape(tq), np.nan)
    au = np.degrees(np.unwrap(np.radians(a[ok])))
    return wrap180(np.interp(tq, np.asarray(t)[ok], au))


def self_test():
    """Round-trip checks; returns dict of max errors (metres / degrees)."""
    rng = np.random.default_rng(0)
    lat = rng.uniform(-80, 80, 2000)
    lon = rng.uniform(-179, 179, 2000)
    h = rng.uniform(-100, 9000, 2000)
    x, y, z = geodetic_to_ecef(lat, lon, h)
    la2, lo2, h2 = ecef_to_geodetic(x, y, z)
    out = {"ecef_roundtrip_lat_deg": float(np.max(np.abs(la2 - lat))),
           "ecef_roundtrip_lon_deg": float(np.max(np.abs(wrap180(lo2 - lon)))),
           "ecef_roundtrip_h_m": float(np.max(np.abs(h2 - h)))}
    # UTM within zone (|dlon| < 3 deg)
    zone = 49
    lonz = rng.uniform(108.0, 114.0, 2000)
    latz = rng.uniform(-10.0, -1.0, 2000)
    E, N, _, _ = latlon_to_utm(latz, lonz, zone, True)
    la3, lo3 = utm_to_latlon(E, N, zone, True)
    out["utm_roundtrip_deg"] = float(max(np.max(np.abs(la3 - latz)), np.max(np.abs(lo3 - lonz))))
    # Known reference: lat 0, lon 111 (zone 49 central meridian) -> E=500000, N=10000000 (south)
    E0, N0, _, _ = latlon_to_utm(np.array([-0.0]), np.array([111.0]), 49, True)
    out["utm_cm_check_E"] = float(E0[0])
    # ENU round trip
    e, n, u = geodetic_to_enu(latz[:50], lonz[:50], h[:50], -7.28, 112.79, 50.0)
    la4, lo4, h4 = enu_to_geodetic(e, n, u, -7.28, 112.79, 50.0)
    out["enu_roundtrip_h_m"] = float(np.max(np.abs(h4 - h[:50])))
    return out
