# MiniTorch Unified Benchmark Run

## Environment

- Captured UTC: `2026-06-03T17:15:34.372202+00:00`
- Git commit: `022831a940ecbf626090b26f995a8b0d5adf9da8`
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
| MLP training | MiniTorch fast CPU | dataset=simple, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 0.2882s | 0.2887s | 0.2869s | 0.2911s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=simple, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 6.0520s | 6.0433s | 5.9895s | 6.0885s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=simple, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10 | 3 | 1 | 0.0174s | 0.0175s | 0.0165s | 0.0187s | ok | median excludes warmup |
| MLP training | MiniTorch fast CPU | dataset=split, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 0.2911s | 0.2908s | 0.2899s | 0.2915s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=split, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 6.0108s | 6.0051s | 5.9408s | 6.0638s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=split, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10 | 3 | 1 | 0.0166s | 0.0162s | 0.0154s | 0.0167s | ok | median excludes warmup |
| MLP training | MiniTorch fast CPU | dataset=xor, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 0.2894s | 0.2899s | 0.2891s | 0.2912s | ok | median excludes warmup |
| MLP training | MiniTorch CUDA | dataset=xor, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10 | 3 | 1 | 5.9098s | 5.9163s | 5.9078s | 5.9313s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10 | 3 | 1 | 0.0160s | 0.0167s | 0.0159s | 0.0182s | ok | median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU (dataset=simple, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `0.2911, 0.2882, 0.2869`
- MiniTorch CUDA (dataset=simple, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `6.0885, 5.9895, 6.0520`
- PyTorch CPU fair mini-batch (dataset=simple, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10): `0.0187, 0.0165, 0.0174`
- MiniTorch fast CPU (dataset=split, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `0.2911, 0.2915, 0.2899`
- MiniTorch CUDA (dataset=split, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `6.0108, 5.9408, 6.0638`
- PyTorch CPU fair mini-batch (dataset=split, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10): `0.0154, 0.0166, 0.0167`
- MiniTorch fast CPU (dataset=xor, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `0.2894, 0.2912, 0.2891`
- MiniTorch CUDA (dataset=xor, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10): `5.9078, 5.9313, 5.9098`
- PyTorch CPU fair mini-batch (dataset=xor, points=100, hidden=10, rate=0.5, epochs=5, batch_size=10): `0.0159, 0.0160, 0.0182`

## GPU Discussion

MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.
CUDA availability is checked with a real device-transfer and kernel-launch probe, not only `numba.cuda.is_available()`.

- `dataset=simple, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10` completed on MiniTorch CUDA with median 6.0520s.
- `dataset=split, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10` completed on MiniTorch CUDA with median 6.0108s.
- `dataset=xor, points=100, hidden=10, rate=0.05, epochs=5, batch_size=10` completed on MiniTorch CUDA with median 5.9098s.
Do not report a GPU speedup unless the CUDA row is `ok` and the result file captures a clean git commit and stable environment.
