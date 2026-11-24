# Runtime and experiment design

MiniTorch separates the public tensor/gradient API from storage and execution.
The staged curriculum implementation remains readable; the engineering extension
adds a measured training path and a reproducible package interface.

## Execution path

`MLP.forward` → `Tensor` operations → `Function.apply` → `TensorBackend` →
`FastOps` (CPU) or `CudaOps` (GPU). Backpropagation traverses the recorded graph
in reverse topological order and reduces broadcast gradients to each input shape.
SGD updates parameters without recording another graph, then installs new leaves.

- `tensor_data.py`: contiguous storage plus shape/stride metadata. Views and
  permutations share storage. CUDA arrays remain device arrays; `to_numpy()` is
  an explicit host transfer. A non-contiguous tensor is copied only when required.
- `tensor.py`, `tensor_functions.py`, `autodiff.py`: gradient flags, function
  history, broadcast-gradient reduction, and nested/context-local `no_grad()`.
- `fast_ops.py`: direct stride arithmetic instead of temporary index arrays;
  aligned elementwise fast paths; serial execution for small operations and
  parallel execution for larger work. Matmul distributes output cells across
  threads, including when there is only one matrix batch.
- `cuda_ops.py`: map/zip kernels, full-axis reductions, and shared-memory tiled
  matmul with barriers before loading another tile. Shape/stride tuples are
  passed by value; outputs are allocated on the device.
- `models.py`: deterministic three-layer MLP and reusable dense layers. The
  default dense operation is `x @ weight + bias`. Broadcast mode remains available
  for experiments. Backend instances are cached so repeated runs reuse compiled
  kernels.

## Experiment interface

- `training.py`: seeded synthetic data, separate validation data, deterministic
  batch ordering, complete evaluation, and machine-readable metrics.
- `checkpoint.py`: versioned NPZ weights with JSON metadata; atomic replacement;
  parameter-name, shape, dtype, and finite-value checks on load. It saves an MLP
  for inference or further manual training, not a complete optimizer/RNG resume state.
- `benchmark.py`: identical workload for MiniTorch/PyTorch, compilation warmups,
  CUDA synchronization, final-parameter/loss validation, raw repeated timings,
  environment information, and a SHA-256 fingerprint of package source files.
- `cli.py`: installed `train`, `evaluate`, and `benchmark` commands. A failed
  benchmark backend returns a nonzero exit status and no valid timing claim.

## Validation

Tests combine the original pytest/Hypothesis and finite-difference checks with
NumPy reference checks and optional PyTorch differential checks. Regression cases
include 512×512 CUDA matmul, non-multiple tile sizes, transposes, batch broadcasting,
reductions exceeding 1,024 elements, extreme softmax logits, dropout gradients,
gradient modes, checkpoint round trips, and a partial-batch CNN forward/backward.
The PyTorch tests compare predictions, every parameter gradient, and an SGD step.

GitHub Actions is configured to run CPU tests and wheel smoke tests on Windows
and Linux with Python 3.11/3.12, plus a separate PyTorch reference job. Real CUDA
tests are opt-in and need a GPU; hosted CPU CI does not certify GPU behavior.

## Deliberate scope

This is an educational float64 runtime. Matmul supports 2D and 3D tensors with
batch broadcasting. Convolution kernels are currently CPU-only. CUDA supports the
MLP tensor path, not the MNIST CNN. There is no distributed training, automatic
mixed precision, general graph compiler, or full PyTorch API compatibility.
Many small GPU kernels and Python graph traversal still cost time. MNIST model
regression tests use generated inputs; they establish execution correctness,
not an accuracy result on downloaded MNIST.

The kernel synchronization and storage approach follow the
[Numba CUDA matrix-multiplication example](https://nvidia.github.io/numba-cuda/user/examples.html)
and [device memory documentation](https://nvidia.github.io/numba-cuda/user/memory.html).
