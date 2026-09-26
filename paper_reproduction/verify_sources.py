from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_MANIFEST.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for item in meta["files"]:
        p = ROOT.parent / item["released_path"]
        actual = sha256(p)
        if actual != item["sha256"]:
            failures.append({
                "path": str(p),
                "expected": item["sha256"],
                "actual": actual,
            })
    if failures:
        raise SystemExit("SOURCE MANIFEST FAILURE:\n" + json.dumps(failures, indent=2))
    print(f"PASS: {len(meta['files'])} reproduction source files verified")

if __name__ == "__main__":
    main()
