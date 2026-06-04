import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import minitorch

from benchmarks.capture_environment import collect_environment
from benchmarks.run_fast_tensor import FastTensorBackend, GPUBackend

try:
    import torch
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    TORCH_IMPORT_ERROR = ""


@dataclass
class MatmulResult:
    size: int
    backend: str
    runs: int
    warmups: int
    median_seconds: float | None
    mean_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    raw_seconds: list[float]
    status: str
    notes: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    if torch is not None:
        torch.manual_seed(seed)


def make_matrix(size: int) -> list[list[float]]:
    return [
        [(((row * 131) + (col * 17)) % 97) / 97.0 for col in range(size)]
        for row in range(size)
    ]


def summarize(
    size: int,
    backend: str,
    runs: int,
    warmups: int,
    times: list[float],
    status: str = "ok",
    notes: str = "",
) -> MatmulResult:
    if not times:
        return MatmulResult(
            size=size,
            backend=backend,
            runs=runs,
            warmups=warmups,
            median_seconds=None,
            mean_seconds=None,
            min_seconds=None,
            max_seconds=None,
            raw_seconds=[],
            status=status,
            notes=notes,
        )

    return MatmulResult(
        size=size,
        backend=backend,
        runs=runs,
        warmups=warmups,
        median_seconds=statistics.median(times),
        mean_seconds=statistics.mean(times),
        min_seconds=min(times),
        max_seconds=max(times),
        raw_seconds=times,
        status=status,
        notes=notes,
    )


def sync_cuda() -> None:
    try:
        from numba import cuda

        cuda.synchronize()
    except Exception:
        pass


def benchmark_minitorch(size: int, backend_name: str, backend, args) -> MatmulResult:
    left = minitorch.tensor(make_matrix(size), backend=backend)
    right = minitorch.tensor(make_matrix(size), backend=backend)
    if backend_name == "MiniTorch CUDA":
        sync_cuda()

    times: list[float] = []

    def run_once() -> None:
        _ = left @ right
        if backend_name == "MiniTorch CUDA":
            sync_cuda()

    try:
        for _ in range(args.warmups):
            run_once()

        for _ in range(args.runs):
            start = time.perf_counter()
            run_once()
            times.append(time.perf_counter() - start)
    except Exception as exc:
        return summarize(
            size=size,
            backend=backend_name,
            runs=args.runs,
            warmups=args.warmups,
            times=times,
            status="failed",
            notes=f"{type(exc).__name__}: {exc}",
        )

    return summarize(
        size=size,
        backend=backend_name,
        runs=args.runs,
        warmups=args.warmups,
        times=times,
        notes="preloaded tensors; construction and host/device transfer excluded",
    )


def benchmark_torch(size: int, args) -> MatmulResult:
    if torch is None:
        return summarize(
            size=size,
            backend="PyTorch CPU",
            runs=args.runs,
            warmups=args.warmups,
            times=[],
            status="skipped",
            notes=f"PyTorch unavailable: {TORCH_IMPORT_ERROR}",
        )

    left = torch.tensor(make_matrix(size), dtype=torch.float64)
    right = torch.tensor(make_matrix(size), dtype=torch.float64)
    times: list[float] = []

    try:
        for _ in range(args.warmups):
            _ = left @ right

        for _ in range(args.runs):
            start = time.perf_counter()
            _ = left @ right
            times.append(time.perf_counter() - start)
    except Exception as exc:
        return summarize(
            size=size,
            backend="PyTorch CPU",
            runs=args.runs,
            warmups=args.warmups,
            times=times,
            status="failed",
            notes=f"{type(exc).__name__}: {exc}",
        )

    return summarize(
        size=size,
        backend="PyTorch CPU",
        runs=args.runs,
        warmups=args.warmups,
        times=times,
        notes="preloaded tensors; median excludes warmup",
    )


def markdown_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}s"


def table_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_markdown(results: list[MatmulResult], environment: dict) -> str:
    lines = [
        "# MiniTorch Matrix Multiply Scaling Benchmark",
        "",
        "This benchmark times preloaded square matrix multiplication across increasing sizes.",
        "Tensor construction and host/device transfer are excluded so the result focuses on backend compute behavior.",
        "",
        "## Environment",
        "",
        f"- Captured UTC: `{environment['captured_at_utc']}`",
        f"- Git commit: `{environment['git_commit']}`",
        "- Git status: `clean`"
        if not environment["git_status"]
        else "- Git status: dirty",
        f"- Python: `{environment['python']}`",
        f"- Platform: `{environment['platform']}`",
        f"- Processor: `{environment['processor']}`",
        "",
        "## Package Versions",
        "",
    ]

    for package, version in environment["packages"].items():
        lines.append(f"- {package}: `{version}`")

    cuda_health = environment.get("cuda_health")
    if cuda_health:
        lines.extend(
            [
                "",
                "## CUDA Runtime Health",
                "",
                f"- Numba CUDA available: `{cuda_health['numba_cuda_available']}`",
                f"- Runtime probe healthy: `{cuda_health['runtime_healthy']}`",
                f"- Device name: `{cuda_health['device_name']}`",
                f"- Probe result: `{cuda_health['probe_result']}`",
            ]
        )
        if cuda_health["error"]:
            lines.append(f"- Error: `{cuda_health['error']}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Size | Backend | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )

    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.size),
                    table_cell(result.backend),
                    str(result.runs),
                    str(result.warmups),
                    markdown_seconds(result.median_seconds),
                    markdown_seconds(result.mean_seconds),
                    markdown_seconds(result.min_seconds),
                    markdown_seconds(result.max_seconds),
                    table_cell(result.status),
                    table_cell(result.notes),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Raw Seconds", ""])
    for result in results:
        raw = ", ".join(f"{item:.6f}" for item in result.raw_seconds) or "N/A"
        lines.append(f"- {result.backend} size={result.size}: `{raw}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "GPU speedups should only be claimed if CUDA wins at larger sizes in the same clean environment.",
            "If CUDA remains slower, the likely bottlenecks are kernel-launch overhead, many small kernels, and non-fused educational kernels rather than raw GPU compute capacity.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_outputs(args, results: list[MatmulResult], environment: dict):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{args.output_name}.md"
    json_path = output_dir / f"{args.output_name}.json"

    markdown_path.write_text(format_markdown(results, environment), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "environment": environment,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path, json_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument(
        "--output-name",
        default="latest_matmul_scaling_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results: list[MatmulResult] = []
    for size in args.sizes:
        results.append(benchmark_minitorch(size, "MiniTorch fast CPU", FastTensorBackend, args))

        if args.include_cuda:
            if GPUBackend is None:
                results.append(
                    summarize(
                        size=size,
                        backend="MiniTorch CUDA",
                        runs=args.runs,
                        warmups=args.warmups,
                        times=[],
                        status="skipped",
                        notes="CUDA runtime health probe failed",
                    )
                )
            else:
                results.append(benchmark_minitorch(size, "MiniTorch CUDA", GPUBackend, args))

        if args.include_torch:
            results.append(benchmark_torch(size, args))

    environment = collect_environment()
    markdown_path, json_path = write_outputs(args, results, environment)

    print(format_markdown(results, environment))
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
