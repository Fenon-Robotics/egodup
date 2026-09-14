from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np


def feature_dir(db_root: Path, profile_id: str, asset_sha: str) -> Path:
    return db_root / "features" / profile_id / asset_sha


def store_features(
    db_root: Path,
    profile_id: str,
    asset_sha: str,
    raw: np.ndarray,
    timestamps: np.ndarray,
    quality: np.ndarray,
    manifest: dict,
) -> Path:
    final = feature_dir(db_root, profile_id, asset_sha)
    if final.exists():
        return final
    staging = db_root / "staging" / f"features-{asset_sha}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    np.save(staging / "raw.npy", np.asarray(raw, dtype=np.float32), allow_pickle=False)
    np.save(
        staging / "timestamps.npy", np.asarray(timestamps, dtype=np.float64), allow_pickle=False
    )
    np.save(staging / "quality.npy", np.asarray(quality, dtype=np.float32), allow_pickle=False)
    manifest = {**manifest, "complete": True, "raw_sha256": _digest(staging / "raw.npy")}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    return final


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest.get("complete") or _digest(path / "raw.npy") != manifest["raw_sha256"]:
        raise RuntimeError(f"Incomplete or corrupt feature artifact: {path}")
    return np.load(path / "raw.npy", mmap_mode="r"), np.load(path / "timestamps.npy", mmap_mode="r")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
