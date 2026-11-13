"""
Authorship, affiliations and citation of GeoSplat (single source used by the GUI, CLI, viewer and reports).

Keep in sync with AUTHORS.md and CITATION.cff (tests/test_about.py checks this).
"""
from __future__ import annotations

from . import __version__

NAME = "GeoSplat"
TAGLINE = "DJI video to georeferenced 3D Gaussian Splatting"
REPO_URL = "https://github.com/helmymukti/GeoSplat"
LICENSE = "MIT"
YEAR = 2026

AFFILIATIONS = {
    1: "Department of Geomatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia",
    2: "Department of Geomatics Engineering, Universitas Gadjah Mada (UGM), Yogyakarta, Indonesia",
}

# (display name, affiliation key, corresponding author)
AUTHORS = [
    ("Helmy Mukti Wijaya", 1, True),
    ("Hepi Hapsari Handayani", 1, False),
    ("Teguh Hariyanto", 1, False),
    ("Harintaka", 2, False),
]
CORRESPONDENCE = "helmy.mukti.wijaya@gmail.com"

# third-party software that GeoSplat runs (fetched at install time, not bundled)
ACKNOWLEDGEMENTS = [
    ("3D Gaussian Splatting", "Kerbl, Kopanas, Leimkühler & Drettakis (2023), Inria / GRAPHDECO",
     "non-commercial research licence"),
    ("COLMAP", "Schönberger & Frahm (2016)", "BSD licence"),
    ("FFmpeg", "FFmpeg developers", "LGPL/GPL"),
    ("splat (WebGL viewer, adapted)", "Kevin Kwok", "MIT licence"),
]


def short_names() -> str:
    """'H. M. Wijaya, H. H. Handayani, T. Hariyanto, Harintaka'"""
    out = []
    for name, _, _ in AUTHORS:
        parts = name.split()
        out.append(name if len(parts) == 1 else " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1])
    return ", ".join(out)


def credit_line() -> str:
    return f"© {YEAR} {short_names()} · ITS & UGM · {LICENSE} licence"


def citation_apa() -> str:
    names = []
    for name, _, _ in AUTHORS:
        parts = name.split()
        names.append(name if len(parts) == 1 else f"{parts[-1]}, " + " ".join(p[0] + "." for p in parts[:-1]))
    who = ", ".join(names[:-1]) + ", & " + names[-1]
    return f"{who} ({YEAR}). {NAME}: local DJI video to georeferenced 3D Gaussian Splatting " \
           f"(Version {__version__}) [Computer software]. {REPO_URL}"


def citation_bibtex() -> str:
    authors = " and ".join(
        (n if len(n.split()) == 1 else f"{n.split()[-1]}, {' '.join(n.split()[:-1])}") for n, _, _ in AUTHORS)
    return ("@software{geosplat,\n"
            f"  author  = {{{authors}}},\n"
            f"  title   = {{{NAME}: local DJI video to georeferenced 3D Gaussian Splatting}},\n"
            f"  version = {{{__version__}}},\n"
            f"  year    = {{{YEAR}}},\n"
            f"  url     = {{{REPO_URL}}},\n"
            f"  license = {{{LICENSE}}}\n}}")


def provenance() -> dict:
    """Embedded in every project's logs/job.json so outputs stay attributable."""
    return {"software": NAME, "version": __version__, "repository": REPO_URL, "licence": LICENSE,
            "authors": [{"name": n, "affiliation": AFFILIATIONS[a], **({"corresponding": CORRESPONDENCE} if c else {})}
                        for n, a, c in AUTHORS],
            "cite": citation_apa()}


def banner() -> str:
    lines = [f"{NAME} {__version__} - {TAGLINE}"]
    for n, a, c in AUTHORS:
        lines.append(f"  {n}{'*' if c else ''} ({a})")
    for k, v in AFFILIATIONS.items():
        lines.append(f"  ({k}) {v}")
    lines.append(f"  * {CORRESPONDENCE}   ·   {REPO_URL}")
    return "\n".join(lines)
