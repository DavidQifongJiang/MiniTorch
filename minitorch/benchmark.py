"""Matched float64 training benchmark with PyTorch numerical validation.

Tensor construction, compilation, and validation are outside measured training.
The result includes source fingerprints so uncommitted experiments are traceable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

from .models import MLP, binary_loss, from_numpy, get_backend, make_dataset
from .optim import SGD


def synchronize(backend):
    if getattr(backend, "cuda", False):
        from numba import cuda
        cuda.synchronize()


def host_array(tensor):
    # Also supports historical revisions whose to_numpy returned a device array.
    value = tensor.to_numpy()
    return value.copy_to_host() if hasattr(value, "copy_to_host") else np.asarray(value)


def source_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "python_source_sha256": digest.hexdigest()}


def run_benchmark(*, points=10000, hidden=64, batch_size=1000, epochs=3,
                  runs=3, warmups=1, threads=4, seed=1337, learning_rate=0.05,
                  backends=("cpu", "torch"), implementation="matmul", label="current"):
    """Return timings only after final parameters agree with a PyTorch reference."""
    import torch

    for name, value in (("points", points), ("hidden", hidden), ("batch_size", batch_size),
                        ("epochs", epochs), ("runs", runs), ("warmups", warmups), ("threads", threads)):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if not set(backends).issubset({"cpu", "cuda", "torch"}):
        raise ValueError("backends must be cpu, cuda, or torch")
    numba.set_num_threads(threads)
    torch.set_num_threads(threads)
    x, y = make_dataset(points, seed)
    initial = MLP(hidden, get_backend("cpu"), seed)
    weights = {name: host_array(p.value).copy() for name, p in initial.named_parameters()}
    batches = [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, points, batch_size)]
    orders = [np.random.default_rng(seed + epoch + 1).permutation(len(batches)) for epoch in range(epochs)]

    def prepare_torch():
        parameters = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True) for name, value in weights.items()}
        inputs = [(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)) for a, b in batches]
        optimizer = torch.optim.SGD(list(parameters.values()), lr=learning_rate)

        def train():
            for order in orders:
                for index in order:
                    a, target = inputs[index]
                    optimizer.zero_grad(set_to_none=True)
                    out = a
                    for layer in range(1, 4):
                        out = out @ parameters[f"layer{layer}.weights"] + parameters[f"layer{layer}.bias"]
                        out = out.relu() if layer < 3 else out.sigmoid()
                    prob = out * target + (out - 1) * (target - 1)
                    loss = -(prob + 1e-6).log().mean()
                    loss.backward()
                    optimizer.step()
            return loss

        def state():
            # Outside timing; also works with older PyTorch builds whose NumPy
            # bridge is incompatible with an installed NumPy major version.
            return {name: np.asarray(p.detach().tolist(), dtype=np.float64) for name, p in parameters.items()}
        return train, state, None

    def prepare_minitorch(name):
        backend = get_backend(name)
        model = MLP(hidden, backend, seed, implementation)
        inputs = [(from_numpy(a, backend), from_numpy(b, backend)) for a, b in batches]
        optimizer = SGD(model.parameters(), learning_rate)
        synchronize(backend)

        def train():
            for order in orders:
                for index in order:
                    a, target = inputs[index]
                    optimizer.zero_grad()
                    loss = binary_loss(model(a), target)
                    loss.backward()
                    optimizer.step()
            synchronize(backend)
            return loss

        def state():
            return {name: host_array(p.value).copy() for name, p in model.named_parameters()}
        return train, state, backend

    reference_train, reference_state, _ = prepare_torch()
    reference_loss = float(reference_train().detach())
    reference = reference_state()
    results = []
    for name in backends:
        print(f"Benchmarking {name} ({implementation}, {label})...", flush=True)
        times, max_error, final_loss = [], 0.0, None
        try:
            for iteration in range(warmups + runs):
                train, state, backend = prepare_torch() if name == "torch" else prepare_minitorch(name)
                start = time.perf_counter()
                loss = train()
                elapsed = time.perf_counter() - start
                final_loss = float(loss.detach()) if name == "torch" else loss.item()
                actual = state()
                for key in reference:
                    max_error = max(max_error, float(np.max(np.abs(actual[key] - reference[key]))))
                    np.testing.assert_allclose(actual[key], reference[key], rtol=1e-6, atol=1e-8, err_msg=f"{name}: {key}")
                np.testing.assert_allclose(final_loss, reference_loss, rtol=1e-6, atol=1e-8)
                if iteration >= warmups:
                    times.append(elapsed)
                print(f"  {'warmup' if iteration < warmups else 'run'} {iteration + 1}: {elapsed:.6f}s, parameters verified", flush=True)
            result = {"backend": name, "status": "passed", "raw_seconds": times,
                      "median_seconds": statistics.median(times), "min_seconds": min(times),
                      "max_seconds": max(times), "examples_per_second": points * epochs / statistics.median(times),
                      "max_parameter_error": max_error, "final_batch_loss": final_loss}
        except Exception as exc:
            result = {"backend": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(json.dumps(result), flush=True)
    versions = {}
    for package in ("numpy", "numba", "numba-cuda", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "packages": versions,
                   "numba_threads": numba.get_num_threads(), "torch_threads": torch.get_num_threads(),
                   **source_fingerprint()}
    if "cuda" in backends:
        from numba import cuda
        try:
            device = cuda.get_current_device()
            device_name = device.name
            environment["cuda_device"] = device_name.decode() if isinstance(device_name, bytes) else device_name
        except Exception:
            environment["cuda_device"] = "unavailable"
    return {"schema_version": 1, "label": label, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"points": points, "hidden": hidden, "batch_size": batch_size,
                       "epochs": epochs, "runs": runs, "warmups": warmups, "threads": threads,
                       "seed": seed, "learning_rate": learning_rate, "dtype": "float64",
                       "linear_implementation": implementation, "dataset": "xor"},
            "methodology": "Identical initial weights, preloaded batches, batch order, mean BCE (+1e-6), and SGD; inputs do not request gradients. Construction, warmups, validation and reporting excluded; CUDA synchronized at timing boundaries. Final parameters and loss checked against eager PyTorch CPU on every run. Fixed steps, not time to convergence.",
            "environment": environment, "results": results}


def add_arguments(parser):
    for name, default in (("points", 10000), ("hidden", 64), ("batch-size", 1000),
                          ("epochs", 3), ("runs", 3), ("warmups", 1), ("threads", 4), ("seed", 1337)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "cuda", "torch"), default=["cpu", "torch"])
    parser.add_argument("--implementation", choices=("matmul", "broadcast"), default="matmul")
    parser.add_argument("--label", default="current")
    parser.add_argument("--output", type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    report = run_benchmark(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(any(row["status"] != "passed" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
