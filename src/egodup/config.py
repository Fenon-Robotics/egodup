from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureProfile:
    name: str = "vsc_raw_v1"
    model_sha256: str = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"
    preprocessing: str = "resize_shorter_320_center_crop_320_bilinear_antialias_imagenet"
    sample_interval_seconds: float = 1.0
    timestamp_policy: str = "closest_pts_tie_earlier_v1"
    dtype: str = "float32"

    @property
    def identity(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class MatchingProfile:
    frame_top_k: int = 50
    min_supported_seconds: float = 10.0
    min_distinct_samples: int = 8
    min_coverage: float = 0.60
    median_cosine: float = 0.80
    p10_cosine: float = 0.65
    max_internal_gap_seconds: float = 3.0
    tn_max_step: int = 5
    tn_top_k: int = 5
    tn_min_length: int = 4
    calibration_id: str = "development-v1"

    @property
    def identity(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {requested!r}, but CUDA is unavailable")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be cpu, auto, or cuda[:index]")
    return requested
