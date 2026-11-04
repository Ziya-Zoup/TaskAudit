"""Verify the explicit publication allowlist and SHA-256 manifest; standard library only."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    allowed = set((root / "PUBLISH_ALLOWLIST.txt").read_text(encoding="utf-8").splitlines())
    manifest_path = root / "PUBLICATION_MANIFEST.json"
    body = manifest_path.read_bytes()
    expected = (root / "PUBLICATION_MANIFEST.sha256").read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(body).hexdigest() == expected, "Manifest sidecar mismatch"
    records = json.loads(body)["files"]
    assert allowed == {r["path"] for r in records} | {
        "PUBLICATION_MANIFEST.json", "PUBLICATION_MANIFEST.sha256"
    }, "Allowlist and manifest disagree"
    actual = set()
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        assert not p.is_symlink(), "Symlinks are not publication members"
        if p.is_file():
            actual.add(rel.as_posix())
    assert actual == allowed, {"unlisted": sorted(actual - allowed), "missing": sorted(allowed - actual)}
    for r in records:
        p = root / r["path"]
        assert not Path(r["path"]).is_absolute() and ".." not in Path(r["path"]).parts
        data = p.read_bytes()
        assert len(data) == r["bytes"] and hashlib.sha256(data).hexdigest() == r["sha256"], r["path"]
    print(json.dumps({"status": "PASS", "listed_files": len(allowed), "hashed_payload_files": len(records)}))


if __name__ == "__main__":
    main()
