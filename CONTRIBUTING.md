# Developing MiniTorch

Create a Python 3.11/3.12 environment and install `python -m pip install -e ".[dev]"`.
Use `python -m pytest -q` for CPU correctness. Install `.[benchmark]` to enable
the PyTorch differential tests in `tests/test_reference.py`.

For real CUDA checks in PowerShell:

```powershell
$env:NUMBA_NUM_THREADS="4"
$env:MINITORCH_RUN_CUDA_TESTS="true"
python -m pytest -q tests/test_tensor_general.py tests/test_runtime.py tests/test_reference.py
```

On bash, prefix the command with `NUMBA_NUM_THREADS=4 MINITORCH_RUN_CUDA_TESTS=true`.
Keep these tests opt-in on CPU machines. Run performance experiments separately
from tests or other CPU/GPU work. Preserve raw JSON, record the command and
hardware, and validate numerical results before interpreting timing differences.

Build with `python -m build`. Install the wheel into a fresh environment and run
`minitorch --help`, a small `train`, and `evaluate` from outside the source tree.
The CI workflow performs this installed-package smoke check as well as tests.

Generated checkpoints and temporary exports belong in ignored `artifacts/`.
Publish selected benchmark records in `benchmarks/results/`, accompanied by
methodology and limitations. Do not replace historical results with new numbers
under the same old experiment name. See `docs/PERFORMANCE.md` for the current
matched workload and the command to reconstruct its historical baseline.
