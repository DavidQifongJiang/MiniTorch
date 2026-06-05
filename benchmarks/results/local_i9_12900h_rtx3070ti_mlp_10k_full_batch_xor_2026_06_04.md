# MiniTorch Unified Benchmark Run

## Environment

- Captured UTC: `2026-06-05T00:25:06.474972+00:00`
- Git commit: `b02ac9f7f135ea8182d8b34bc7fef2e3d359047b`
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
| MLP training | MiniTorch fast CPU | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | 24.8662s | 24.8662s | 24.8547s | 24.8778s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | N/A | N/A | N/A | N/A | failed | AssertionError:  |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true | 2 | 1 | 0.0755s | 0.0755s | 0.0273s | 0.1238s | ok | median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU (dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true): `24.8547, 24.8778`
- MiniTorch CUDA (dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true): `N/A`
- PyTorch CPU fair mini-batch (dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true): `0.1238, 0.0273`

## Timing Breakdown

Median cumulative timing per measured training run. This is most useful with `--collect-timing`, `--preload-batches`, and `--skip-eval` when comparing where training time goes.

| Backend | Config | Data Prep | Forward | Loss + Backward | Optimizer | Evaluation | Epoch Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniTorch fast CPU | dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true | 0.0111s | 4.6395s | 20.1994s | 0.0046s | N/A | 24.8435s |
| PyTorch CPU fair mini-batch | dataset=xor, points=10000, hidden=64, rate=0.5, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true | 0.0597s | 0.0012s | 0.0094s | 0.0005s | N/A | 0.0112s |


## GPU Discussion

MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.
CUDA availability is checked with a real device-transfer and kernel-launch probe, not only `numba.cuda.is_available()`.

- `dataset=xor, points=10000, hidden=64, rate=0.05, epochs=3, batch_size=10000, preload_batches=true, evaluate=false, collect_timing=true` was `failed` on MiniTorch CUDA: AssertionError: 
Do not report a GPU speedup unless the CUDA row is `ok` and the result file captures a clean git commit and stable environment.
