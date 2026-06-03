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
| XOR MLP training | MiniTorch fast CPU | PTS=250, HIDDEN=10, RATE=0.05 | TBD | TBD | Warmed up |
| XOR MLP training | MiniTorch CUDA | PTS=250, HIDDEN=10, RATE=0.05 | TBD | TBD | Warmed up |
| XOR MLP training | PyTorch CPU | PTS=250, HIDDEN=10, RATE=0.5 | TBD | TBD | Baseline |
| Tensor kernel diagnostics | Numba fast CPU | map/zip/reduce/matmul diagnostics | TBD | TBD | Uses `parallel_check.py` |

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

## Raw Output

Add raw benchmark output here after running the commands.
