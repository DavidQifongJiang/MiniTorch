# Matched performance experiment — 2026-11-12

This is a new experiment, separate from the historical June 2026 timings.
Both the original engine and optimized engine were run with the same benchmark
harness and validated against eager PyTorch CPU.

## Results

| Backend | Original engine median | Optimized engine median | Original / optimized |
| --- | ---: | ---: | ---: |
| MiniTorch CPU | 49.135708 s | 0.296903 s | 165.5× |
| MiniTorch CUDA | 30.018630 s | 0.808137 s | 37.1× |
| PyTorch CPU reference | 0.024469 s | 0.027972 s | Reference measured in each session |

On this workload the optimized MiniTorch CPU path is about **10.6× slower than
PyTorch CPU**, versus about 2,008× for the original engine in its session.
CUDA remains slower than MiniTorch CPU for this small network. This is a local,
fixed-step training comparison, not a general framework speedup or time-to-accuracy
result. PyTorch itself varied between sessions; all raw repetitions are preserved.

- [Original engine record](../benchmarks/results/performance_before_2026_11_12.json)
- [Optimized engine record](../benchmarks/results/performance_after_2026_11_12.json)

Every warmup and measured run passed final-parameter and loss validation. Maximum
absolute final-parameter error versus PyTorch was **1.11e-16** for both MiniTorch
backends. Final pre-update batch loss was 0.5964929954566615 on CPU and
0.5964929954566619 on CUDA/PyTorch. This loss is a consistency check, not held-out accuracy.

## Workload and measurement

- XOR, 10,000 points; MLP dimensions 2→64→64→1; ReLU/ReLU/sigmoid.
- Batch size 1,000; three epochs; 30 SGD steps; learning rate 0.05; seed 1337.
- Float64 for every backend; four Numba/PyTorch CPU threads.
- Identical initial weights, dataset, batch order, mean binary cross entropy
  with explicit epsilon 1e-6, and SGD update rule.
- Preloaded inputs. One complete warmup followed by three measured runs, each
  starting from the same initial parameters. Backend instances are reused so
  measured runs do not recompile fresh kernel closures.
- Timer includes forward, loss, backward, optimizer, Python dispatch, and output
  allocation. Construction, warmup/compilation, parameter validation, and logging
  are excluded. CUDA is synchronized before timing and at completion.
- Every final parameter is checked against PyTorch at `rtol=1e-6, atol=1e-8`;
  failed backends receive no timing result and cause a nonzero CLI exit status.
- Local Windows machine: i9-12900H / RTX 3070 Ti Laptop GPU. Python 3.11.15,
  NumPy 2.4.6, Numba 0.65.1, numba-cuda 0.30.2, PyTorch 2.12.0+cpu.
  Detailed environment and package-source SHA-256 fingerprints are in each JSON.

The original engine is committed revision
`836e9e2b26905b8765a4a75c4852d15defac8b45`, exported into an isolated directory
with the new `models.py` and `benchmark.py` harness. Its dense layers use broadcast
multiply/reduce, matching the original trainer's formulation. The optimized
engine uses direct matmul. Neither path delegates MiniTorch computation to PyTorch;
PyTorch is used for comparison and validation only.

Inputs do not request gradients in the harness. The original engine nevertheless
creates gradient history for false gradient flags; fixing that behavior is part
of the engine improvement being measured.

## Why the implementation changed

1. **Dense layers:** replace broadcast multiplication followed by reduction with
   `x @ weight`. A 1,000×64 by 64×64 layer previously materialized 4,096,000
   float64 products, or 32.768 MB, before reduction. The new formulation avoids
   that specific temporary. This is an analytical allocation count, not a measured
   reduction in peak process/GPU memory.
2. **CPU kernels:** avoid per-element temporary index buffers; add aligned/scalar
   fast paths; distribute matmul output cells across threads; choose serial kernels
   for small operations to avoid thread-launch overhead.
3. **CUDA storage:** preserve device arrays through operations and metadata-only
   views, allocate outputs on-device, and pass small shape/stride tuples by value.
4. **Gradient bookkeeping:** honor `requires_grad_(False)`, skip constant input
   accumulation, avoid zero-buffer allocation for the first gradient, and prevent
   optimizer/inference operations from recording graphs.
5. **Correctness:** synchronize shared-memory tile reuse, handle incomplete tiles,
   finish reductions beyond 1,024 elements, and stabilize softmax/log-softmax.

These changes were measured together. The table does not attribute a separate
speedup to each optimization. It also does not compare against PyTorch CUDA,
`torch.compile`, float32, or mixed precision. Small CUDA kernels, scalar transfers,
allocations, and Python graph traversal remain overheads. First-use compilation
still matters for short-lived programs and is intentionally excluded here.

## Reproduce

Install `.[benchmark]` (and `.[cuda]` for an appropriate NVIDIA environment).
From the repository root, with no competing training/tests running:

```bash
python -m minitorch benchmark --implementation matmul --label after --backends cpu cuda torch --output artifacts/after.json
```

To reconstruct the original engine without changing your checkout:

```bash
python benchmarks/prepare_baseline.py --revision 836e9e2b26905b8765a4a75c4852d15defac8b45 --output artifacts/baseline
cd artifacts/baseline
python -m minitorch.benchmark --implementation broadcast --label before --backends cpu cuda torch --output ../before.json
```

Use the same Python executable/environment for both commands. Omit `cuda` on CPU
machines. `--implementation broadcast` on the current engine supports a future
ablation of dense-layer formulation; it is not the full historical baseline.

The historical 25.40 s / 24.30 s / 0.083 s records used a different trainer and
precision/thread protocol. Do not combine those values with this experiment to
claim an improvement ratio.

## Matrix multiplication validation

The [fresh matrix scaling record](../benchmarks/results/validated_matmul_2026_11_12.md)
checks every output against NumPy einsum for sizes 32, 64, 128, 256, and 512,
including the larger CUDA cases that failed the earlier audit. All passed.
At 512×512, synchronized operation medians were 109.205 ms for MiniTorch CPU,
3.959 ms for MiniTorch CUDA, and 1.836 ms for PyTorch CPU. These are different
devices for the CUDA/PyTorch comparison; no PyTorch GPU claim is implied.

```bash
python benchmarks/run_matmul_scaling.py --include-cuda --include-torch --sizes 32 64 128 256 512 --runs 5 --warmups 1 --threads 4 --output-name validated_matmul_2026_11_12
```

Input construction, warmup, and validation are excluded. Python dispatch, output
allocation, and completion synchronization are included. Five measured runs follow
one warmup; thread counts are four. Sub-millisecond timings are noisy, so the
[JSON](../benchmarks/results/validated_matmul_2026_11_12.json) retains every repetition.

## Local verification

- CPU/default suite, Python 3.12.7 / Numba 0.60.0: **252 passed, 4 expected failures**.
- CPU + actual CUDA backend suites, Python 3.11.15 / Numba 0.65.1:
  **163 passed** (`test_tensor_general.py`, `test_runtime.py`, `test_reference.py`).
- Both runs used `NUMBA_NUM_THREADS=4`; CUDA tests additionally used
  `MINITORCH_RUN_CUDA_TESTS=true`. CUDA checks ran on the RTX 3070 Ti Laptop GPU.
- Source distribution and wheel built successfully. The wheel was installed into
  a separate virtual environment reusing existing dependencies, and import location
  was checked to ensure execution came from its `site-packages`, outside the source tree.
- Installed `train`, `evaluate`, and `benchmark` commands passed smoke checks.
  The default XOR example trained on 1,000 examples for 100 epochs with hidden
  size 16, reaching 98.7% on 1,000 validation examples (seed 1338). Its saved
  checkpoint reached 99.0% on 1,000 separately generated test examples (seed 1339)
  on both CPU and CUDA. This single-seed toy-data check demonstrates the pipeline;
  it is not a generalization claim about real-world datasets.
  [Saved training and evaluation record](../benchmarks/results/training_demo_2026_11_12.json).

## GitHub verification — 2026-11-24

The [GitHub Actions run](https://github.com/DavidQifongJiang/MiniTorch/actions/runs/45477282622)
passed all five jobs for commit `4d62a7623bee9b85db09e799b9507ef403ca8eb3`:

- Windows and Linux, each with Python 3.11 and 3.12: 250 passed, one skipped
  optional PyTorch test module, and four existing expected failures per job.
- All four platform jobs built the source distribution and wheel, installed the
  wheel, and exercised training/checkpoint evaluation outside the source directory.
- A separate PyTorch reference job passed both forward/backward/SGD comparison tests.

Hosted CI validates CPU execution and packaging. Actual CUDA validation and
performance measurements remain the local GPU results recorded above.
