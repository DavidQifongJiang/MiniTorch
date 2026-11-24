"""Portable MLP checkpoints: JSON metadata and NumPy arrays, never pickle."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .models import MLP, from_numpy, get_backend


def save_checkpoint(model: MLP, path, metadata=None):
    """Atomically save weights and experiment metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"schema_version": 1, "model": "MLP", "hidden": model.hidden,
            "seed": model.seed, "metadata": metadata or {}}
    arrays = {name: param.value.to_numpy() for name, param in model.named_parameters()}
    arrays["__metadata__"] = np.array(json.dumps(info))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, backend="cpu"):
    """Validate the whole checkpoint before loading weights onto a backend."""
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(str(archive["__metadata__"]))
        if info.get("schema_version") != 1 or info.get("model") != "MLP":
            raise ValueError("Unsupported checkpoint format or model")
        if not isinstance(info.get("hidden"), int) or info["hidden"] < 1:
            raise ValueError("Invalid hidden size in checkpoint")
        target = get_backend(backend) if isinstance(backend, str) else backend
        model = MLP(info["hidden"], target, info["seed"])
        params = dict(model.named_parameters())
        if set(archive.files) != set(params) | {"__metadata__"}:
            raise ValueError("Checkpoint parameter names do not match the model")
        arrays = {}
        for name, param in params.items():
            value = archive[name]
            if value.shape != param.value.shape or value.dtype != np.dtype("float64"):
                raise ValueError(f"Invalid shape or dtype for checkpoint parameter {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint parameter {name}")
            arrays[name] = value.copy()
    for name, param in params.items():
        param.update(from_numpy(arrays[name], target))
    model.eval()
    return model, info.get("metadata", {})
