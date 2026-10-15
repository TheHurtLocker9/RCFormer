"""Verify the distributed files against the release manifest."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(name)
    if failures:
        raise ValueError("Release integrity check failed: " + ", ".join(failures))
    return {"verified_files": len(manifest["files"]), "failures": []}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
