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
| XOR MLP training | MiniTorch fast CPU | dataset=xor, points=250, hidden=10, rate=0.05, epochs=25 | 5 | 3.9934s | 1 warmup, median excludes warmup |
| XOR MLP training | PyTorch CPU | dataset=xor, points=250, hidden=10, rate=0.5, epochs=25 | 5 | 0.0170s | 1 warmup, median excludes warmup |
| XOR MLP training | MiniTorch CUDA | dataset=xor, points=250, hidden=10, rate=0.05, epochs=25 | N/A | N/A | Not published; CUDA smoke run failed in this Windows/Numba environment |
| Tensor kernel diagnostics | Numba fast CPU | map/zip/reduce/matmul diagnostics | TBD | TBD | Uses `parallel_check.py` |

## Latest Validated Run

| Field | Value |
| --- | --- |
| Date | 2026-06-03 |
| Git commit | `c24b6d3d5ff207b033d30927451b7f9740969d95` |
| Git status at capture | clean |
| Command | `python benchmarks/run_all.py --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --dataset xor --output-name local_i9_12900h_rtx3070ti_run_2026_06_03` |
| Detailed Markdown | [`local_i9_12900h_rtx3070ti_run_2026_06_03.md`](local_i9_12900h_rtx3070ti_run_2026_06_03.md) |
| Raw JSON | [`local_i9_12900h_rtx3070ti_run_2026_06_03.json`](local_i9_12900h_rtx3070ti_run_2026_06_03.json) |

CUDA note: a small smoke run with `--include-cuda` failed with a Windows/Numba access violation. CUDA numbers should not be reported until the CUDA backend is validated in a stable CUDA environment.

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
python benchmarks/run_all.py --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --dataset xor
```

Unified benchmark runner with CUDA:

```powershell
python benchmarks/run_all.py --include-cuda --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --dataset xor
```

## Raw Output

Add raw benchmark output here after running the commands.
