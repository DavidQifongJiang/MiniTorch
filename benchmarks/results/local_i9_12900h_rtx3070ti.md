# Local Benchmark Results: i9-12900H + RTX 3070 Ti Laptop GPU

This file is the canonical local benchmark record for MiniTorch. Keep all numbers in this file tied to the same machine and dependency environment.

## Equipment

| Component | Value |
| --- | --- |
| Machine | ASUS ROG Strix G533ZW |
| CPU | 12th Gen Intel Core i9-12900H |
| CPU layout | 14 cores / 20 logical processors |
| RAM | about 25 GB |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| GPU memory | 8192 MiB |
| NVIDIA driver | 581.80 |
| CUDA reported by driver | 13.0 |
| OS | Windows 11 |
| Python | 3.12.7 |

## Benchmark Protocol

- Run from the MiniTorch repository root.
- Use the same terminal session for related benchmark runs.
- Warm up JIT-backed code before timing.
- Run each timed benchmark at least 5 times.
- Report median time.
- Keep raw output below the summary table.

## Environment Capture

Command:

```powershell
python benchmarks/capture_environment.py --markdown
```

Paste the captured output here before publishing benchmark numbers. Make sure the captured git commit matches the code used for the benchmark run.

## Summary

| Benchmark | Backend | Input / Config | Runs | Median Time | Notes |
| --- | --- | --- | ---: | ---: | --- |
| MLP training | MiniTorch fast CPU | dataset=simple, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 3.6311s | 1 warmup, fair mini-batch comparison |
| MLP training | PyTorch CPU fair mini-batch | dataset=simple, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 0.2016s | 1 warmup, fair mini-batch comparison |
| MLP training | MiniTorch fast CPU | dataset=split, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 3.6641s | 1 warmup, fair mini-batch comparison |
| MLP training | PyTorch CPU fair mini-batch | dataset=split, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 0.2040s | 1 warmup, fair mini-batch comparison |
| MLP training | MiniTorch fast CPU | dataset=xor, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 3.6415s | 1 warmup, fair mini-batch comparison |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 0.1964s | 1 warmup, fair mini-batch comparison |
| MLP training | MiniTorch CUDA | see CUDA validation section | 3 | see below | Validated in dedicated `minitorch-cuda` environment |
| Tensor kernel diagnostics | Numba fast CPU | map/zip/reduce/matmul diagnostics | TBD | TBD | Uses `parallel_check.py` |

## Latest Validated Run

| Field | Value |
| --- | --- |
| Date | 2026-06-03 |
| Git commit | `738f7cccb39a559526c40562df2dc1346c7d99b1` |
| Git status at capture | clean |
| Command | `python benchmarks/run_all.py --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --batch-size 10 --datasets simple split xor --output-name local_i9_12900h_rtx3070ti_fair_batch_run_2026_06_03` |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_fair_batch_run_2026_06_03.md`](local_i9_12900h_rtx3070ti_fair_batch_run_2026_06_03.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_fair_batch_run_2026_06_03.json`](local_i9_12900h_rtx3070ti_fair_batch_run_2026_06_03.json) |

## Latest CUDA Validation Run

CUDA validation uses a dedicated conda environment instead of base Anaconda:

```powershell
conda activate minitorch-cuda
python benchmarks/cuda_health.py --markdown
python benchmarks/run_all.py --include-cuda --runs 3 --warmups 1 --epochs 5 --points 100 --hidden 10 --batch-size 10 --datasets simple split xor --output-name local_i9_12900h_rtx3070ti_cuda_validation_multi_2026_06_03
```

| Field | Value |
| --- | --- |
| Date | 2026-06-03 |
| Git commit | `022831a940ecbf626090b26f995a8b0d5adf9da8` |
| Git status at capture | clean |
| Python | `3.11.15` |
| Numba | `0.65.1` with `numba-cuda` |
| PyTorch | `2.12.0+cpu` |
| CUDA runtime health | healthy; RTX 3070 Ti detected; probe result `2.0` |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_cuda_validation_multi_2026_06_03.md`](local_i9_12900h_rtx3070ti_cuda_validation_multi_2026_06_03.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_cuda_validation_multi_2026_06_03.json`](local_i9_12900h_rtx3070ti_cuda_validation_multi_2026_06_03.json) |

| Dataset | MiniTorch fast CPU median | MiniTorch CUDA median | PyTorch CPU median |
| --- | ---: | ---: | ---: |
| simple | 0.2882s | 6.0520s | 0.0174s |
| split | 0.2911s | 6.0108s | 0.0166s |
| xor | 0.2894s | 5.9098s | 0.0160s |

CUDA interpretation: the CUDA backend is now functionally validated, but it is slower than MiniTorch fast CPU on this small MLP workload. The result is expected for this implementation because training launches many small kernels and still pays host/device transfer overhead. Report this as a systems-learning result, not a GPU speedup result.

Historical CUDA note: base Anaconda previously failed a CUDA health probe with a Windows/Numba access violation. The dedicated `minitorch-cuda` environment fixes this by using the NVIDIA-maintained Numba CUDA target/bindings.

## Matrix Multiplication Scaling Run

Matrix multiplication scaling isolates backend compute behavior by timing preloaded square tensors. Tensor construction and host/device transfer are excluded.

```powershell
conda activate minitorch-cuda
python benchmarks/run_matmul_scaling.py --include-cuda --include-torch --runs 5 --warmups 1 --sizes 32 64 128 256 512 --output-name local_i9_12900h_rtx3070ti_matmul_scaling_2026_06_04
```

| Field | Value |
| --- | --- |
| Date | 2026-06-04 |
| Git commit | `253640d2b7b411fb9b41e973cf818726d751bf27` |
| Git status at capture | clean |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_matmul_scaling_2026_06_04.md`](local_i9_12900h_rtx3070ti_matmul_scaling_2026_06_04.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_matmul_scaling_2026_06_04.json`](local_i9_12900h_rtx3070ti_matmul_scaling_2026_06_04.json) |

| Matrix Size | MiniTorch fast CPU median | MiniTorch CUDA median | PyTorch CPU median | CUDA vs MiniTorch CPU |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.000294s | 0.003012s | 0.000003s | 0.10x |
| 64 | 0.001335s | 0.003107s | 0.000021s | 0.43x |
| 128 | 0.003955s | 0.005294s | 0.000021s | 0.75x |
| 256 | 0.021037s | 0.008230s | 0.000155s | 2.56x |
| 512 | 0.292770s | 0.026949s | 0.000960s | 10.86x |

Scaling interpretation: CUDA is slower on small matrices because launch overhead dominates. Once the matrix is large enough, the CUDA backend overtakes MiniTorch fast CPU, reaching about 10.9x speedup at 512x512. PyTorch CPU remains much faster because it uses mature native BLAS/kernel implementations.

Superseded note: the earlier `local_i9_12900h_rtx3070ti_run_2026_06_03` result used MiniTorch mini-batches but a PyTorch full-batch baseline. It is retained as raw history but should not be used as the main comparison.

## Commands

Fast CPU trainer:

```powershell
python benchmarks/run_fast_tensor.py --BACKEND cpu --DATASET xor --PTS 250 --HIDDEN 10 --RATE 0.05
```

CUDA trainer:

```powershell
python benchmarks/run_fast_tensor.py --BACKEND gpu --DATASET xor --PTS 250 --HIDDEN 10 --RATE 0.05
```

PyTorch baseline:

```powershell
python benchmarks/run_torch.py
```

Parallel diagnostics:

```powershell
python benchmarks/parallel_check.py
```

Unified benchmark runner:

```powershell
python benchmarks/run_all.py --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --batch-size 10 --datasets simple split xor
```

Unified benchmark runner with CUDA:

```powershell
python benchmarks/run_all.py --include-cuda --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --batch-size 10 --datasets simple split xor
```

## Raw Output

Add raw benchmark output here after running the commands.
