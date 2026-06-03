# MiniTorch Benchmark Methodology

MiniTorch benchmarks should be reported from one standard environment. The purpose is to make performance claims reproducible enough for engineering discussion instead of treating benchmark numbers as one-off screenshots.

## Standard Benchmark Environment

Use this machine as the primary benchmark target unless a result explicitly says otherwise:

| Component | Standard Environment |
| --- | --- |
| Machine | ASUS ROG Strix G533ZW |
| CPU | 12th Gen Intel Core i9-12900H |
| CPU layout | 14 cores / 20 logical processors |
| Memory | about 25 GB RAM |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| GPU memory | 8192 MiB |
| NVIDIA driver | 581.80 |
| CUDA reported by driver | 13.0 |
| OS | Windows 11 |
| Python | 3.12.7 |

If benchmarks are run on another machine, create a separate result file under `benchmarks/results/` and put that machine in the file name.

## Benchmark Rules

1. Use the same machine for all numbers in a comparison table.
2. Record Python, OS, CPU, GPU, driver, CUDA, and dependency versions.
3. Run from a clean terminal with no training jobs, games, browsers, or IDE indexing tasks running in the background when possible.
4. Warm up JIT-backed code before measuring. Numba compilation time should not be mixed with steady-state runtime unless the result is explicitly labeled cold-start.
5. Run each benchmark at least 5 times.
6. Report median time as the headline number.
7. Keep raw command output in the result file or a linked artifact.
8. Separate CPU, fast CPU, CUDA, and PyTorch baseline numbers.
9. Use fixed input sizes, epochs, hidden dimensions, dataset, learning rate, and random seed where the benchmark supports it.
10. Do not compare numbers from different machines in the same speedup claim.

## Recommended Benchmark Categories

| Category | What It Shows | Example |
| --- | --- | --- |
| Naive tensor backend vs fast CPU backend | Impact of optimized tensor kernels | map, zip, reduce, matrix multiply |
| Fast CPU backend vs CUDA backend | Hardware acceleration story | MLP training or matrix multiply |
| MiniTorch vs PyTorch baseline | Framework overhead and implementation gap | XOR MLP training |
| Scaling behavior | How performance changes as tensor size grows | matrix multiply sizes 64, 128, 256, 512 |

## Standard Commands

Capture the environment:

```powershell
python benchmarks/capture_environment.py --markdown
```

Run backend parallel diagnostics:

```powershell
python benchmarks/parallel_check.py
```

Run the fast MiniTorch trainer:

```powershell
python benchmarks/run_fast_tensor.py --BACKEND cpu --DATASET xor --PTS 250 --HIDDEN 10 --RATE 0.05
```

Run the CUDA MiniTorch trainer, if CUDA is available:

```powershell
python benchmarks/run_fast_tensor.py --BACKEND gpu --DATASET xor --PTS 250 --HIDDEN 10 --RATE 0.05
```

Run the PyTorch baseline:

```powershell
python benchmarks/run_torch.py
```

Run the unified benchmark suite:

```powershell
python benchmarks/run_all.py --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --dataset xor
```

Include CUDA when the current Python environment has a working CUDA backend:

```powershell
python benchmarks/run_all.py --include-cuda --runs 5 --warmups 1 --epochs 25 --points 250 --hidden 10 --dataset xor
```

## Reporting Format

Use this table format in result files:

| Benchmark | Backend | Input / Config | Runs | Median Time | Notes |
| --- | --- | --- | ---: | ---: | --- |
| XOR MLP training | MiniTorch fast CPU | PTS=250, HIDDEN=10, RATE=0.05 | 5 | TBD | Warmed up |
| XOR MLP training | MiniTorch CUDA | PTS=250, HIDDEN=10, RATE=0.05 | 5 | TBD | Warmed up |
| XOR MLP training | PyTorch CPU | PTS=250, HIDDEN=10, RATE=0.5 | 5 | TBD | Baseline |

## Interpreting Results

Benchmark numbers are not the main claim of MiniTorch. The main claim is systems understanding: autodiff, tensor layout, backend abstraction, and kernel implementation. Benchmarks support that story by showing that backend choices have measurable performance impact.
