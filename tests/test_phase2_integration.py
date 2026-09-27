from pathlib import Path
import re

import pcrsaits

ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        text,
    )
    assert match is not None
    return match.group(1)


def _citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r'(?m)^version: "([^"]+)"$', text)
    assert m is not None
    return m.group(1)


def test_third_adapter_public_exports():
    assert hasattr(pcrsaits, "CSDIBackbone")
    assert hasattr(pcrsaits, "PCRCSDI")


def test_release_lifecycle_version_and_citation_are_consistent():
    project_version = _project_version()
    citation_version = _citation_version()
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    if project_version == "1.1.0.dev0":
        assert citation_version == "1.0.0"
        assert "last published software release" in cff
    elif project_version == "1.1.0":
        assert citation_version == "1.1.0"
        assert "last published software release" not in cff
    else:
        raise AssertionError(
            f"Unsupported Phase-3 lifecycle version: {project_version!r}"
        )


def test_phase2_docs_make_no_uq_claim():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Phase 2 does **not** add the experimental PCR-CSDI-UQ" in normalized
