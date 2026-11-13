"""Credits stay consistent across the app, AUTHORS.md and CITATION.cff; stage 6 is part of the pipeline."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geosplat import __version__, about  # noqa: E402
from geosplat.pipeline import STAGES, Job, Pipeline  # noqa: E402


def _read(name):
    return open(os.path.join(ROOT, name), encoding="utf-8").read()


def test_authors_match_authors_md_and_citation():
    md, cff = _read("AUTHORS.md"), _read("CITATION.cff")
    for name, aff, _ in about.AUTHORS:
        assert name in md
        assert name.split()[-1] in cff
        assert about.AFFILIATIONS[aff] in md and about.AFFILIATIONS[aff] in cff
    assert about.CORRESPONDENCE in md and about.CORRESPONDENCE in cff


def test_version_consistent():
    assert f'version = "{__version__}"' in _read("pyproject.toml")
    assert f"version: {__version__}" in _read("CITATION.cff")
    assert f"[{__version__}]" in _read("CHANGELOG.md")


def test_citations_and_credit_line():
    apa, bib = about.citation_apa(), about.citation_bibtex()
    assert "Wijaya, H. M." in apa and "Harintaka" in apa and __version__ in apa
    assert "Wijaya, Helmy Mukti and Handayani, Hepi Hapsari and Hariyanto, Teguh and Harintaka" in bib
    assert about.short_names() == "H. M. Wijaya, H. H. Handayani, T. Hariyanto, Harintaka"
    prov = about.provenance()
    assert prov["version"] == __version__ and len(prov["authors"]) == 4
    json.dumps(prov)


def test_viewer_carries_credit():
    html = _read(os.path.join("web", "viewer.html"))
    assert "H. M. Wijaya, H. H. Handayani, T. Hariyanto, Harintaka" in html
    assert 'lang="en"' in html


def test_stage6_in_pipeline(tmp_path):
    keys = [k for k, _ in STAGES]
    assert keys == ["telemetry", "colmap", "georef", "gaussians", "web", "vcorr"]
    video = tmp_path / "v.MP4"
    video.write_bytes(b"")
    cp = tmp_path / "cp.csv"
    cp.write_text("id,E,N,h\n")
    job = Job(str(video), checkpoints=str(cp))
    assert "vcorr" in job.stages
    cmd = Pipeline(job).command("vcorr")
    assert cmd[1].endswith("stage6_vcorr.py") and cmd[-2:] == ["--checkpoints", str(cp)]
    assert "--checkpoints" not in Pipeline(Job(str(video))).command("vcorr")
