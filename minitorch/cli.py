"""Installed command-line interface for training, evaluation, and benchmarks."""

import argparse
import json
from pathlib import Path

import numba


def main(argv=None):
    parser = argparse.ArgumentParser(prog="minitorch", description="Train and benchmark a small deep learning runtime.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train an MLP and save a portable checkpoint")
    training.add_argument("--dataset", choices=("xor", "simple", "split"), default="xor")
    for name, default in (("points", 1000), ("hidden", 16), ("epochs", 100), ("batch-size", 64),
                          ("validation-points", 1000), ("seed", 1337), ("threads", min(4, numba.config.NUMBA_NUM_THREADS))):
        training.add_argument(f"--{name}", type=int, default=default)
    training.add_argument("--learning-rate", type=float, default=0.1)
    training.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--checkpoint", type=Path, required=True)
    training.add_argument("--report", type=Path)
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved model on a new generated test set")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    evaluation.add_argument("--points", type=int, default=1000)
    evaluation.add_argument("--seed", type=int, help="Defaults to training seed + 2; separate from validation")
    evaluation.add_argument("--batch-size", type=int, default=256)
    evaluation.add_argument("--threads", type=int, default=min(4, numba.config.NUMBA_NUM_THREADS))
    from .benchmark import add_arguments
    add_arguments(commands.add_parser("benchmark", help="Run a matched PyTorch comparison (requires benchmark extra)"))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        threads = args.get("threads", 4)
        if threads < 1:
            raise ValueError("threads must be positive")
        numba.set_num_threads(threads)
        if command == "benchmark":
            from .benchmark import run_benchmark
            output = args.pop("output")
            result = run_benchmark(**args)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return int(any(row["status"] != "passed" for row in result["results"]))
        args.pop("threads")
        from .checkpoint import load_checkpoint, save_checkpoint
        from .models import make_dataset
        from .training import evaluate, train
        if command == "train":
            checkpoint, report_path = args.pop("checkpoint"), args.pop("report")
            model, report = train(**args, log=lambda row: print(json.dumps(row), flush=True))
            save_checkpoint(model, checkpoint, report)
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"checkpoint": str(checkpoint), "validation": report["history"][-1]["validation"]}))
        else:
            model, metadata = load_checkpoint(args["checkpoint"], args["backend"])
            seed = args["seed"] if args["seed"] is not None else metadata.get("seed", model.seed) + 2
            x, y = make_dataset(args["points"], seed, metadata.get("dataset", "xor"))
            print(json.dumps({"seed": seed, "dataset": metadata.get("dataset", "xor"), **evaluate(model, x, y, args["batch_size"])}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"minitorch: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
