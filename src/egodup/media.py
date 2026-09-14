from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def local_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"Not a local file: {p}")
    return p


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    data = json.loads(proc.stdout)
    if not data.get("streams"):
        raise RuntimeError("No video stream")
    stream = data["streams"][0]
    duration = stream.get("duration") or data.get("format", {}).get("duration")
    return {
        "duration_seconds": float(duration) if duration is not None else None,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name"),
        "time_base": stream.get("time_base"),
        "pixel_format": stream.get("pix_fmt"),
        "color_space": stream.get("color_space"),
        "rotation": next(
            (
                int(x.get("rotation", 0))
                for x in stream.get("side_data_list", [])
                if "rotation" in x
            ),
            0,
        ),
    }


@dataclass
class SampledVideo:
    frames: list[Image.Image]
    timestamps: np.ndarray
    frame_ids: np.ndarray
    errors: np.ndarray
    diagnostics: list[str]


def sample_video(path: Path, interval: float = 1.0) -> SampledVideo:
    """Select closest decoded presentation timestamp; equal distances prefer earlier."""
    before = path.stat()
    candidates: list[tuple[float, int, Image.Image]] = []
    diagnostics: list[str] = []
    with av.open(str(path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
        streams = [s for s in container.streams.video]
        if not streams:
            raise RuntimeError("No video stream")
        stream = streams[0]
        for idx, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time is None:
                diagnostics.append(f"missing_pts_frame:{idx}")
                continue
            candidates.append((float(frame.time), idx, frame.to_image().convert("RGB")))
    if not candidates:
        raise RuntimeError("No decodable timestamped video frames")
    candidates.sort(key=lambda x: (x[0], x[1]))
    zero = candidates[0][0]
    rel = np.asarray([x[0] - zero for x in candidates], dtype=np.float64)
    targets = np.arange(0.0, rel[-1] + interval * 0.5, interval)
    frames: list[Image.Image] = []
    times: list[float] = []
    ids: list[int] = []
    errors: list[float] = []
    last_idx = None
    for target in targets:
        pos = int(np.searchsorted(rel, target, side="left"))
        choices = [i for i in (pos - 1, pos) if 0 <= i < len(rel)]
        chosen = min(choices, key=lambda i: (abs(rel[i] - target), rel[i]))
        if chosen == last_idx:
            continue
        last_idx = chosen
        frames.append(candidates[chosen][2])
        times.append(float(rel[chosen]))
        ids.append(candidates[chosen][1])
        errors.append(float(abs(rel[chosen] - target)))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while decoding: {path}")
    return SampledVideo(frames, np.asarray(times), np.asarray(ids), np.asarray(errors), diagnostics)


def discover(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
