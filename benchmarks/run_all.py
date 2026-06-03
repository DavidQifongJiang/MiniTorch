import argparse
import contextlib
import io
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
from benchmarks.run_fast_tensor import FastTensorBackend, FastTrain, GPUBackend
from benchmarks.run_torch import TorchTrain


DATASETS = {
    "simple": "Simple",
    "diag": "Diag",
    "split": "Split",
    "xor": "Xor",
    "circle": "Circle",
    "spiral": "Spiral",
}


@dataclass
class BenchmarkResult:
    name: str
    backend: str
    config: str
    runs: int
    warmups: int
    median_seconds: float | None
    mean_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    raw_seconds: list[float]
    status: str
    notes: str


def quiet_log_fn(*args, **kwargs):
    return None


def set_seed(seed: int):
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_dataset(name: str, points: int):
    dataset_key = DATASETS[name]
    return minitorch.datasets[dataset_key](points)


def timed_call(fn):
    stream = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(stream):
        fn()
    return time.perf_counter() - start


def summarize(
    name: str,
    backend: str,
    config: str,
    runs: int,
    warmups: int,
    times: list[float],
    status: str = "ok",
    notes: str = "",
):
    if not times:
        return BenchmarkResult(
            name=name,
            backend=backend,
            config=config,
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

    return BenchmarkResult(
        name=name,
        backend=backend,
        config=config,
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


def run_training_benchmark(
    name: str,
    backend_label: str,
    trainer_factory,
    dataset_name: str,
    points: int,
    hidden: int,
    rate: float,
    epochs: int,
    batch_size: int,
    runs: int,
    warmups: int,
    seed: int,
):
    config = (
        f"dataset={dataset_name}, points={points}, hidden={hidden}, "
        f"rate={rate}, epochs={epochs}, batch_size={batch_size}"
    )
    times = []

    def run_once(iteration: int):
        set_seed(seed + iteration)
        data = make_dataset(dataset_name, points)
        trainer = trainer_factory(hidden, batch_size)
        trainer.train(data, rate, max_epochs=epochs, log_fn=quiet_log_fn)

    try:
        for i in range(warmups):
            timed_call(lambda i=i: run_once(i))

        for i in range(runs):
            elapsed = timed_call(lambda i=i: run_once(warmups + i))
            times.append(elapsed)
    except Exception as exc:
        return summarize(
            name=name,
            backend=backend_label,
            config=config,
            runs=runs,
            warmups=warmups,
            times=times,
            status="failed",
            notes=f"{type(exc).__name__}: {exc}",
        )

    return summarize(
        name=name,
        backend=backend_label,
        config=config,
        runs=runs,
        warmups=warmups,
        times=times,
        notes="median excludes warmup",
    )


def run_cuda_benchmark(args, dataset_name: str):
    if GPUBackend is None:
        return summarize(
            name="MLP training",
            backend="MiniTorch CUDA",
            config=training_config(args, dataset_name),
            runs=args.runs,
            warmups=args.warmups,
            times=[],
            status="skipped",
            notes="CUDA backend is not available in this Python environment",
        )

    return run_training_benchmark(
        name="MLP training",
        backend_label="MiniTorch CUDA",
        trainer_factory=lambda hidden, batch_size: FastTrain(
            hidden, backend=GPUBackend, batch_size=batch_size
        ),
        dataset_name=dataset_name,
        points=args.points,
        hidden=args.hidden,
        rate=args.rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        runs=args.runs,
        warmups=args.warmups,
        seed=args.seed,
    )


def training_config(args, dataset_name: str):
    return (
        f"dataset={dataset_name}, points={args.points}, hidden={args.hidden}, "
        f"rate={args.rate}, epochs={args.epochs}, batch_size={args.batch_size}"
    )


def markdown_seconds(value: float | None):
    if value is None:
        return "N/A"
    return f"{value:.4f}s"


def table_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def gpu_discussion(results: list[BenchmarkResult]):
    cuda_results = [result for result in results if "CUDA" in result.backend]
    lines = [
        "## GPU Discussion",
        "",
        "MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.",
        "",
    ]

    if not cuda_results:
        lines.append(
            "CUDA was not requested for this run. Use `--include-cuda` to add MiniTorch CUDA rows."
        )
        return lines

    for result in cuda_results:
        if result.status == "ok":
            lines.append(
                f"- `{result.config}` completed on MiniTorch CUDA with median {markdown_seconds(result.median_seconds)}."
            )
        else:
            lines.append(
                f"- `{result.config}` was `{result.status}` on MiniTorch CUDA: {result.notes}"
            )

    lines.append(
        "Do not report a GPU speedup unless the CUDA row is `ok` and the result file captures a clean git commit and stable environment."
    )
    return lines


def format_markdown(results: list[BenchmarkResult], environment: dict):
    lines = [
        "# MiniTorch Unified Benchmark Run",
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

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Benchmark | Backend | Config | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )

    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    table_cell(result.name),
                    table_cell(result.backend),
                    table_cell(result.config),
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
        raw = ", ".join(f"{item:.4f}" for item in result.raw_seconds) or "N/A"
        lines.append(f"- {result.backend} ({result.config}): `{raw}`")

    lines.extend([""] + gpu_discussion(results))

    return "\n".join(lines) + "\n"


def write_outputs(args, results, environment):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_name
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

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
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--torch-rate", type=float, default=0.5)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=None,
        help="Single dataset to benchmark. Use --datasets for multiple.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=None,
        help="One or more datasets to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--include-cuda",
        action="store_true",
        help="Include MiniTorch CUDA benchmark if available.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory for markdown and JSON benchmark output.",
    )
    parser.add_argument(
        "--output-name",
        default="latest_unified_benchmark",
        help="Output file stem for markdown and JSON results.",
    )
    return parser.parse_args()


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    if args.dataset:
        return [args.dataset]
    return ["xor"]


def main():
    args = parse_args()
    results = []

    for dataset_name in selected_datasets(args):
        results.append(
            run_training_benchmark(
                name="MLP training",
                backend_label="MiniTorch fast CPU",
                trainer_factory=lambda hidden, batch_size: FastTrain(
                    hidden, backend=FastTensorBackend, batch_size=batch_size
                ),
                dataset_name=dataset_name,
                points=args.points,
                hidden=args.hidden,
                rate=args.rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                runs=args.runs,
                warmups=args.warmups,
                seed=args.seed,
            )
        )

        if args.include_cuda:
            results.append(run_cuda_benchmark(args, dataset_name))

        results.append(
            run_training_benchmark(
                name="MLP training",
                backend_label="PyTorch CPU fair mini-batch",
                trainer_factory=lambda hidden, batch_size: TorchTrain(
                    hidden, batch_size=batch_size
                ),
                dataset_name=dataset_name,
                points=args.points,
                hidden=args.hidden,
                rate=args.torch_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                runs=args.runs,
                warmups=args.warmups,
                seed=args.seed,
            )
        )

    environment = collect_environment()
    markdown_path, json_path = write_outputs(args, results, environment)

    print(format_markdown(results, environment))
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
