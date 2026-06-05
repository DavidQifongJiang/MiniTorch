# MiniTorch Unified Benchmark Run

## Environment

- Captured UTC: `2026-06-05T00:19:10.484655+00:00`
- Git commit: `0bc3b687b09cabd57163c0b4454a6cbd110f2bb6`
- Git status: `clean`
- Python: `3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:36:00) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`

## Package Versions

- minitorch: `installed`
- numpy: `2.4.6`
- numba: `0.65.1`
- torch: `2.12.0+cpu`

## CUDA Runtime Health

- Numba CUDA available: `True`
- Runtime probe healthy: `True`
- Device name: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
- Probe result: `2.0`

## Summary

| Benchmark | Backend | Config | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MLP training | MiniTorch fast CPU | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | 25.4046s | 25.4046s | 25.3776s | 25.4316s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | 24.3043s | 24.3043s | 24.2723s | 24.3364s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | 0.0829s | 0.0829s | 0.0312s | 0.1346s | ok | median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU (dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true): `25.3776, 25.4316`
- MiniTorch CUDA (dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true): `24.3364, 24.2723`
- PyTorch CPU fair mini-batch (dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true): `0.1346, 0.0312`

## Timing Breakdown

Median cumulative timing per measured training run. This is most useful with `--collect-timing`, `--preload-batches`, and `--skip-eval` when comparing where training time goes.

| Backend | Config | Data Prep | Forward | Loss + Backward | Optimizer | Evaluation | Epoch Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniTorch fast CPU | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 0.0113s | 4.7949s | 20.5437s | 0.0466s | N/A | 25.3855s |
| MiniTorch CUDA | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 0.0110s | 5.9638s | 17.8471s | 0.4606s | N/A | 24.2715s |
| PyTorch CPU fair mini-batch | dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true | 0.0582s | 0.0044s | 0.0120s | 0.0038s | N/A | 0.0202s |


## GPU Discussion

MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.
CUDA availability is checked with a real device-transfer and kernel-launch probe, not only `numba.cuda.is_available()`.

- `dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=1000, preload_batches=true, evaluate=false, collect_timing=true` completed on MiniTorch CUDA with median 24.3043s.
Do not report a GPU speedup unless the CUDA row is `ok` and the result file captures a clean git commit and stable environment.
