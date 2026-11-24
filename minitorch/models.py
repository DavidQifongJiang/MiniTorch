"""Small reference models shared by the CLI, benchmarks, and examples."""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from .fast_ops import FastOps
from .module import Module, Parameter
from .tensor import Tensor
from .tensor_ops import TensorBackend


@lru_cache(maxsize=2)
def get_backend(name: str = "cpu") -> TensorBackend:
    """Select a backend without initializing CUDA for CPU-only users."""
    if name == "cpu":
        return TensorBackend(FastOps)
    if name == "cuda":
        from numba import cuda
        from .cuda_ops import CudaOps

        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select --backend cpu.")
        return TensorBackend(CudaOps)
    raise ValueError(f"Unknown backend: {name!r}")


def from_numpy(values: np.ndarray, backend: TensorBackend) -> Tensor:
    """Create an owned float64 tensor from a NumPy array."""
    values = np.array(values, dtype=np.float64, order="C", copy=True)
    return Tensor.make(values.reshape(-1), values.shape, backend=backend)


class Linear(Module):
    """Dense layer; broadcast mode is retained for performance experiments."""

    def __init__(self, weight, bias, backend, implementation="matmul"):
        super().__init__()
        if implementation not in ("matmul", "broadcast"):
            raise ValueError("Linear implementation must be matmul or broadcast")
        self.weights = Parameter(from_numpy(weight, backend))
        self.bias = Parameter(from_numpy(bias, backend))
        self.implementation = implementation

    def forward(self, x):
        weight = self.weights.value
        if self.implementation == "broadcast":
            batch, in_size = x.shape
            out_size = weight.shape[1]
            out = (x.view(batch, in_size, 1) * weight.view(1, in_size, out_size))
            out = out.sum(1).view(batch, out_size)
        else:
            out = x @ weight
        return out + self.bias.value


class MLP(Module):
    """A reproducibly initialized 2 -> hidden -> hidden -> 1 classifier."""

    def __init__(self, hidden=64, backend=None, seed=1337, implementation="matmul"):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.backend = backend if backend is not None else get_backend()
        self.seed = seed
        self.implementation = implementation
        rng = np.random.default_rng(seed)
        for i, (in_size, out_size) in enumerate(((2, hidden), (hidden, hidden), (hidden, 1)), 1):
            weight = rng.uniform(-0.5, 0.5, size=(in_size, out_size))
            bias = np.full((out_size,), 0.1)
            setattr(self, f"layer{i}", Linear(weight, bias, self.backend, implementation))

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x).relu()).relu()).sigmoid()


def make_dataset(points=1000, seed=1337, dataset="xor"):
    """Generate independent synthetic examples with a local random generator."""
    if points < 1:
        raise ValueError("points must be positive")
    x = np.random.default_rng(seed).random((points, 2))
    if dataset == "xor":
        y = (x[:, 0] > 0.5) != (x[:, 1] > 0.5)
    elif dataset == "simple":
        y = x[:, 0] < 0.5
    elif dataset == "split":
        y = (x[:, 0] < 0.2) | (x[:, 0] > 0.8)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return x, y.astype(np.float64).reshape(-1, 1)


def binary_loss(prediction, target):
    """Mean binary cross entropy with an explicit log epsilon (1e-6)."""
    probability = prediction * target + (prediction - 1.0) * (target - 1.0)
    return -(probability + 1e-6).log().mean()
