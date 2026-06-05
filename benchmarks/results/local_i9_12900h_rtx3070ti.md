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
- Run each small/standard benchmark at least 5 times. For long-running scaled CUDA
  experiments, run at least 2 measured trials, keep raw timings, and label the
  run as scaled rather than standard.
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
| MLP training | MiniTorch fast CPU | dataset=xor, points=10000, hidden=64, epochs=3, batch_size=1000, preloaded batches, no eval timing | 2 | 25.4046s | 1 warmup, scaled workload |
| MLP training | MiniTorch CUDA | dataset=xor, points=10000, hidden=64, epochs=3, batch_size=1000, preloaded batches, no eval timing | 2 | 24.3043s | 1 warmup, 1.05x vs MiniTorch fast CPU |
| MLP training | MiniTorch CUDA | dataset=xor, points=10000, hidden=64, epochs=3, batch_size=10000, preloaded batches, no eval timing | 2 | N/A | Full-batch boundary check failed with `AssertionError` |
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

## Scaled MLP Training Run

This run increases the training workload to test whether MiniTorch CUDA improves when each batch has more work.

```powershell
conda activate minitorch-cuda
python benchmarks/run_all.py --include-cuda --runs 2 --warmups 1 --epochs 3 --points 1000 --hidden 64 --batch-size 100 --datasets xor --output-name local_i9_12900h_rtx3070ti_mlp_scaled_xor_2026_06_04
```

| Field | Value |
| --- | --- |
| Date | 2026-06-04 |
| Git commit | `eab119914f30e7456805b549ab26052978dfa525` |
| Git status at capture | clean |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_mlp_scaled_xor_2026_06_04.md`](local_i9_12900h_rtx3070ti_mlp_scaled_xor_2026_06_04.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_mlp_scaled_xor_2026_06_04.json`](local_i9_12900h_rtx3070ti_mlp_scaled_xor_2026_06_04.json) |

| Dataset | Config | MiniTorch fast CPU median | MiniTorch CUDA median | PyTorch CPU median |
| --- | --- | ---: | ---: | ---: |
| xor | points=1000, hidden=64, epochs=3, batch_size=100 | 3.2728s | 6.3295s | 0.0171s |

Scaled MLP interpretation: increasing the workload narrows the gap, but CUDA is still slower for this training path. The likely cause is that the MLP workload still launches many separate kernels and repeatedly creates batch tensors, so overhead remains significant even when batches are larger.

## Large MLP Training with Preloaded Batches

This run uses benchmark-only controls to reduce repeated batch tensor creation
and remove periodic evaluation from timed training. It is meant to isolate the
training path more clearly than the small validation run.

```powershell
conda activate minitorch-cuda
python benchmarks/run_all.py --include-cuda --runs 2 --warmups 1 --epochs 3 --points 10000 --hidden 64 --batch-size 1000 --datasets xor --preload-batches --skip-eval --collect-timing --output-name local_i9_12900h_rtx3070ti_mlp_10k_preloaded_xor_2026_06_04
```

| Field | Value |
| --- | --- |
| Captured UTC | `2026-06-05T00:19:10.484655+00:00` |
| Git commit | `0bc3b687b09cabd57163c0b4454a6cbd110f2bb6` |
| Git status at capture | clean |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_mlp_10k_preloaded_xor_2026_06_04.md`](local_i9_12900h_rtx3070ti_mlp_10k_preloaded_xor_2026_06_04.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_mlp_10k_preloaded_xor_2026_06_04.json`](local_i9_12900h_rtx3070ti_mlp_10k_preloaded_xor_2026_06_04.json) |

| Dataset | Config | MiniTorch fast CPU median | MiniTorch CUDA median | PyTorch CPU median |
| --- | --- | ---: | ---: | ---: |
| xor | points=10000, hidden=64, epochs=3, batch_size=1000, preloaded batches | 25.4046s | 24.3043s | 0.0829s |

| Backend | Data prep | Forward | Loss + backward | Optimizer | Epoch total |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniTorch fast CPU | 0.0113s | 4.7949s | 20.5437s | 0.0466s | 25.3855s |
| MiniTorch CUDA | 0.0110s | 5.9638s | 17.8471s | 0.4606s | 24.2715s |
| PyTorch CPU fair mini-batch | 0.0582s | 0.0044s | 0.0120s | 0.0038s | 0.0202s |

Large MLP interpretation: preloading batches and increasing batch size finally
lets MiniTorch CUDA slightly beat MiniTorch fast CPU. The gain is small because
the training loop still launches many kernels and optimizer work is expensive
relative to PyTorch's mature native implementation. The useful resume claim is
not "MiniTorch beats PyTorch"; it is "I built and measured CPU/CUDA backends,
then identified where the custom backend wins, loses, and still needs work."

## Full-Batch Boundary Check

This run checks whether a single full batch improves GPU behavior. MiniTorch
CUDA currently fails this configuration with an assertion, so it is documented
as a backend limitation rather than a performance claim.

```powershell
conda activate minitorch-cuda
python benchmarks/run_all.py --include-cuda --runs 2 --warmups 1 --epochs 3 --points 10000 --hidden 64 --batch-size 10000 --datasets xor --preload-batches --skip-eval --collect-timing --output-name local_i9_12900h_rtx3070ti_mlp_10k_full_batch_xor_2026_06_04
```

| Field | Value |
| --- | --- |
| Captured UTC | `2026-06-05T00:25:06.474972+00:00` |
| Git commit | `b02ac9f7f135ea8182d8b34bc7fef2e3d359047b` |
| Git status at capture | clean |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_mlp_10k_full_batch_xor_2026_06_04.md`](local_i9_12900h_rtx3070ti_mlp_10k_full_batch_xor_2026_06_04.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_mlp_10k_full_batch_xor_2026_06_04.json`](local_i9_12900h_rtx3070ti_mlp_10k_full_batch_xor_2026_06_04.json) |

| Backend | Status | Median |
| --- | --- | ---: |
| MiniTorch fast CPU | ok | 24.8662s |
| MiniTorch CUDA | failed: `AssertionError` | N/A |
| PyTorch CPU fair mini-batch | ok | 0.0755s |

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
