from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_reproduction_manifest_is_exact():
    meta = json.loads(
        (ROOT / "paper_reproduction/SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert meta["program_count"] == 11
    for item in meta["files"]:
        p = ROOT / item["released_path"]
        assert p.is_file()
        assert p.stat().st_size == item["size"]
        assert sha256(p) == item["sha256"]

def test_final_paper_physionet_reference_is_patient_aware_v8():
    refs = json.loads(
        (ROOT / "paper_reproduction/FINAL_PAPER_SOURCE_REFERENCES.json").read_text(
            encoding="utf-8"
        )
    )
    v8 = refs["physionet_R4_3_v8"]
    assert v8["title"] == "pcr_saits_R4_3_physionet_patientaware_full_v8.py"
    assert v8["file_library_object"] == "file_00000000024881fa815a3696d67e952f"
    assert v8["bundled"] is False

def test_citation_has_article_and_software_dois():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "10.1016/j.eswa.2026.134510" in text
    assert "10.5281/zenodo.22973879" in text
    assert 'version: "1.0.0"' in text
    assert 'family-names: "Somnugpong"' in text

def test_pyproject_is_release_candidate():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "1.0.0"' in text
    assert 'license = "MIT"' in text

def test_reproduction_dependencies_are_separate():
    core = set(
        x.strip() for x in (ROOT / "requirements.txt").read_text().splitlines()
        if x.strip()
    )
    repro = set(
        x.strip() for x in
        (ROOT / "requirements-reproduction.txt").read_text().splitlines()
        if x.strip()
    )
    assert {"numpy", "torch", "pypots"} <= core
    assert {"pandas", "scikit-learn", "xgboost", "shap", "benchpots", "tsdb"} <= repro
def test_final_license_is_mit():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License")
    assert "Copyright (c) 2026 Sawet Somnugpong" in license_text
    assert 'license = "MIT"' in pyproject
    assert ("All rights" + " reserved") not in license_text
    assert "MIT License" in readme
