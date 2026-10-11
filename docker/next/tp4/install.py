"""Plan the exact preserved TP4 payload; do not modify files before validation."""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(payload, site_packages):
    manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
    changes = []
    for name, meta in manifest["runtime_files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py":
            raise ValueError("Invalid TP4 runtime path")
        if rel.parts[0] == "runtime" and len(rel.parts) == 2:
            target = site_packages / "sm75_next_tp4_runtime" / rel.name
        elif name == "sm75_next_tp4.py":
            target = site_packages / name
        else:
            raise ValueError("Invalid TP4 runtime destination")
        data = (payload / name).read_bytes()
        if digest(data) != meta["sha256"]:
            raise ValueError("TP4 source hash mismatch: " + name)
        ast.parse(data, filename=name)
        if target.exists() and digest(target.read_bytes()) != meta["sha256"]:
            raise ValueError("Unreviewed installed TP4 source: " + name)
        changes.append((data, target))
    for name, meta in manifest["flashinfer_files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid PCIe IPC path")
        target = site_packages / "flashinfer" / name
        data = (payload / "flashinfer" / name).read_bytes()
        if digest(data) != meta["sha256"]:
            raise ValueError("PCIe IPC payload mismatch: " + name)
        current = digest(target.read_bytes()) if target.exists() else None
        allowed = {meta["sha256"], meta.get("base_sha256")} if meta["kind"] == "append" else {None, meta["sha256"]}
        if current not in allowed or (meta["kind"] == "append" and current is None):
            raise ValueError("Unreviewed FlashInfer source: " + name)
        if rel.suffix == ".py":
            ast.parse(data, filename=name)
        changes.append((data, target))
    return changes, manifest
