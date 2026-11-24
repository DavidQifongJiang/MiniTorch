"""Export a historical engine plus the current matched benchmark harness.

No checkout or worktree changes are needed. Run the exported benchmark from its
directory so imports resolve to the historical engine.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"], cwd=root, text=True
    ).strip()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be an empty directory to avoid overwriting files")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "minitorch"], cwd=root, text=True
    ).splitlines()
    for name in paths:
        if name.endswith(".py"):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root))
    # Shared workload, initialization, validation and timing boundaries.
    for name in ("models.py", "benchmark.py"):
        (output / "minitorch" / name).write_bytes((root / "minitorch" / name).read_bytes())
    (output / "baseline.json").write_text(json.dumps({
        "engine_revision": revision, "harness_files": ["models.py", "benchmark.py"],
        "run_from": str(output), "implementation": "broadcast"
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {revision} to {output}")


if __name__ == "__main__":
    main()
