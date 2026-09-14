from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .config import FeatureProfile


MODEL_NAME = "sscd-disc-mixup"
MODEL_FILE = "sscd_disc_mixup.torchscript.pt"
ADAPTED_FILE = "sscd_disc_mixup.no_l2_norm.torchscript.pt"
MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"


def cache_dir() -> Path:
    return Path(os.environ.get("EGODUP_CACHE", Path.home() / ".cache" / "egodup"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_l2_norm(model):
    import collections
    import torch
    from torch import nn

    embeddings = model.eval().embeddings
    if embeddings.original_name == "L2Norm":
        modules = [("backbone", model.backbone)]
    else:
        components = list(embeddings.children())
        names = [c.original_name for c in components]
        if names != ["GlobalGeMPool2d", "Linear", "L2Norm"]:
            raise RuntimeError(f"Unexpected SSCD checkpoint structure: {names}")
        modules = [
            ("backbone", model.backbone),
            ("pool", components[0]),
            ("project", components[1]),
        ]
    adapted = nn.Sequential(collections.OrderedDict(modules)).eval()
    probe = torch.randn([2, 3, 64, 64])
    with torch.inference_mode():
        expected, raw = model(probe), adapted(probe)
    if not torch.allclose(torch.nn.functional.normalize(raw), expected, atol=1e-3, rtol=1e-3):
        raise RuntimeError("Adapted SSCD structural parity failed")
    return torch.jit.trace(adapted, probe)


def fetch_model(*, offline: bool = False, source: Path | None = None) -> Path:
    import torch

    root = cache_dir() / "models"
    root.mkdir(parents=True, exist_ok=True)
    original, adapted_path = root / MODEL_FILE, root / ADAPTED_FILE
    expected = FeatureProfile().model_sha256
    if source:
        if digest(source) != expected:
            raise RuntimeError("Local model checksum mismatch")
        if source.resolve() != original.resolve():
            shutil.copyfile(source, original)
    elif not original.exists():
        if offline:
            raise RuntimeError("Model is not cached and --offline prevents download")
        tmp = original.with_suffix(".partial")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, original)
    if digest(original) != expected:
        raise RuntimeError("Downloaded model checksum mismatch")
    if not adapted_path.exists():
        original_model = torch.jit.load(str(original), map_location="cpu").eval()
        adapted = remove_l2_norm(original_model)
        tmp = adapted_path.with_suffix(".partial")
        torch.jit.save(adapted, str(tmp))
        os.replace(tmp, adapted_path)
        _real_image_parity(original_model, adapted)
    return adapted_path


def _preprocess(images):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import center_crop, normalize, pil_to_tensor, resize

    tensors = []
    for image in images:
        x = resize(image, 320, interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = center_crop(x, [320, 320])
        x = pil_to_tensor(x).float().div_(255.0)
        tensors.append(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return torch.stack(tensors)


def _real_image_parity(original, adapted) -> None:
    from PIL import Image
    import torch

    arr = np.indices((361, 487)).sum(axis=0).astype(np.uint8)
    image = Image.fromarray(np.stack((arr, np.roll(arr, 17, 0), np.roll(arr, 29, 1)), axis=-1))
    x = _preprocess([image])
    with torch.inference_mode():
        expected, raw = original(x), adapted(x)
    if not torch.allclose(torch.nn.functional.normalize(raw), expected, atol=2e-4, rtol=2e-4):
        raise RuntimeError("Adapted SSCD real-image parity failed")


def infer(images, device: str, batch_size: int = 32) -> tuple[np.ndarray, int]:
    import torch

    model = torch.jit.load(str(fetch_model(offline=True)), map_location=device).eval().to(device)
    outputs = []
    effective = batch_size
    start = 0
    while start < len(images):
        try:
            x = _preprocess(images[start : start + effective]).to(device)
            with torch.inference_mode():
                y = model(x).float().cpu().numpy()
            if (
                y.ndim != 2
                or y.shape[1] != 512
                or not np.isfinite(y).all()
                or np.any(np.linalg.norm(y, axis=1) == 0)
            ):
                raise RuntimeError(f"Invalid SSCD output shape/values: {y.shape}")
            outputs.append(y)
            start += len(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if effective <= 1:
                raise
            effective = max(1, effective // 2)
    return np.concatenate(outputs).astype(np.float32), effective


def model_metadata() -> dict:
    path = fetch_model(offline=True)
    return {
        "name": MODEL_NAME,
        "original_sha256": FeatureProfile().model_sha256,
        "adapted_sha256": digest(path),
    }
