import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cuda_health import cuda_healthcheck


def run_command(command):
    if shutil.which(command[0]) is None:
        return None

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"error: {exc}"

    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def module_version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"not available ({exc})"

    return getattr(module, "__version__", "installed")


def collect_environment():
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short"]),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "minitorch": module_version("minitorch"),
            "numpy": module_version("numpy"),
            "numba": module_version("numba"),
            "torch": module_version("torch"),
        },
        "cuda_health": cuda_healthcheck(),
        "nvidia_smi": run_command(["nvidia-smi"]),
    }


def print_markdown(environment):
    print("# Captured Benchmark Environment")
    print()
    print(f"- Captured UTC: `{environment['captured_at_utc']}`")
    print(f"- Git commit: `{environment['git_commit']}`")
    if environment["git_status"]:
        print("- Git status: dirty")
        print()
        print("```text")
        print(environment["git_status"])
        print("```")
        print()
    else:
        print("- Git status: `clean`")
    print(f"- Python: `{environment['python']}`")
    print(f"- Platform: `{environment['platform']}`")
    print(f"- Machine: `{environment['machine']}`")
    print(f"- Processor: `{environment['processor']}`")
    print()
    print("## Package Versions")
    print()
    for name, version in environment["packages"].items():
        print(f"- {name}: `{version}`")
    print()

    cuda_health = environment["cuda_health"]
    print("## CUDA Runtime Health")
    print()
    print(f"- Numba CUDA available: `{cuda_health['numba_cuda_available']}`")
    print(f"- Runtime probe healthy: `{cuda_health['runtime_healthy']}`")
    print(f"- Device name: `{cuda_health['device_name']}`")
    print(f"- Probe result: `{cuda_health['probe_result']}`")
    if cuda_health["error"]:
        print(f"- Error: `{cuda_health['error']}`")
    print()

    if environment["nvidia_smi"]:
        print("## NVIDIA SMI")
        print()
        print("```text")
        print(environment["nvidia_smi"])
        print("```")
    else:
        print("NVIDIA SMI was not available.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown instead of JSON.",
    )
    args = parser.parse_args()

    environment = collect_environment()
    if args.markdown:
        print_markdown(environment)
    else:
        print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
