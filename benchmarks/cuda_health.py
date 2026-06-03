import argparse
import json
from functools import lru_cache
from typing import Any

import numpy as np
from numba import cuda


@cuda.jit
def _cuda_add_one_probe(values):
    index = cuda.grid(1)
    if index == 0:
        values[index] += 1.0


def _stringify(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


@lru_cache(maxsize=1)
def _cached_cuda_healthcheck() -> dict[str, Any]:
    health: dict[str, Any] = {
        "numba_cuda_available": False,
        "runtime_healthy": False,
        "device_name": None,
        "probe_result": None,
        "error": None,
    }

    try:
        health["numba_cuda_available"] = bool(cuda.is_available())
        if not health["numba_cuda_available"]:
            health["error"] = "numba.cuda.is_available() returned false"
            return health

        values = np.array([1.0], dtype=np.float64)
        device_values = cuda.to_device(values)
        _cuda_add_one_probe[1, 1](device_values)
        cuda.synchronize()
        result = device_values.copy_to_host()

        health["probe_result"] = float(result[0])
        health["runtime_healthy"] = abs(float(result[0]) - 2.0) < 1e-9
        if not health["runtime_healthy"]:
            health["error"] = f"CUDA probe returned {float(result[0])}, expected 2.0"

        try:
            health["device_name"] = _stringify(cuda.get_current_device().name)
        except Exception as exc:
            health["device_name"] = f"unknown ({type(exc).__name__}: {exc})"

    except Exception as exc:
        health["error"] = f"{type(exc).__name__}: {exc}"

    return health


def cuda_healthcheck() -> dict[str, Any]:
    return dict(_cached_cuda_healthcheck())


def cuda_runtime_healthy() -> bool:
    return bool(cuda_healthcheck()["runtime_healthy"])


def print_markdown(health: dict[str, Any]) -> None:
    print("# CUDA Runtime Health")
    print()
    print(f"- Numba CUDA available: `{health['numba_cuda_available']}`")
    print(f"- Runtime probe healthy: `{health['runtime_healthy']}`")
    print(f"- Device name: `{health['device_name']}`")
    print(f"- Probe result: `{health['probe_result']}`")
    if health["error"]:
        print(f"- Error: `{health['error']}`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown instead of JSON.",
    )
    args = parser.parse_args()

    health = cuda_healthcheck()
    if args.markdown:
        print_markdown(health)
    else:
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()
