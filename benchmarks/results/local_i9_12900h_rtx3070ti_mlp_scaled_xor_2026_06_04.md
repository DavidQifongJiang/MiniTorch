# MiniTorch Unified Benchmark Run

## Environment

- Captured UTC: `2026-06-04T23:55:22.629803+00:00`
- Git commit: `eab119914f30e7456805b549ab26052978dfa525`
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
| MLP training | MiniTorch fast CPU | dataset=xor, points=1000, hidden=64, rate=0.05, epochs=3, batch_size=100 | 2 | 1 | 3.2728s | 3.2728s | 3.2452s | 3.3004s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=xor, points=1000, hidden=64, rate=0.05, epochs=3, batch_size=100 | 2 | 1 | 6.3295s | 6.3295s | 6.1607s | 6.4982s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=1000, hidden=64, rate=0.5, epochs=3, batch_size=100 | 2 | 1 | 0.0171s | 0.0171s | 0.0164s | 0.0177s | ok | median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU (dataset=xor, points=1000, hidden=64, rate=0.05, epochs=3, batch_size=100): `3.3004, 3.2452`
- MiniTorch CUDA (dataset=xor, points=1000, hidden=64, rate=0.05, epochs=3, batch_size=100): `6.4982, 6.1607`
- PyTorch CPU fair mini-batch (dataset=xor, points=1000, hidden=64, rate=0.5, epochs=3, batch_size=100): `0.0177, 0.0164`

## GPU Discussion

MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.
CUDA availability is checked with a real device-transfer and kernel-launch probe, not only `numba.cuda.is_available()`.

- `dataset=xor, points=1000, hidden=64, rate=0.05, epochs=3, batch_size=100` completed on MiniTorch CUDA with median 6.3295s.
Do not report a GPU speedup unless the CUDA row is `ok` and the result file captures a clean git commit and stable environment.
