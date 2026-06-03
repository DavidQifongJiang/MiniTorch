# MiniTorch Unified Benchmark Run

## Environment

- Captured UTC: `2026-06-03T03:43:21.575129+00:00`
- Git commit: `738f7cccb39a559526c40562df2dc1346c7d99b1`
- Git status: `clean`
- Python: `3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`

## Package Versions

- minitorch: `installed`
- numpy: `2.0.2`
- numba: `0.61.0`
- torch: `2.5.1+cu121`

## Summary

| Benchmark | Backend | Config | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MLP training | MiniTorch fast CPU | dataset=simple, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 1 | 3.6311s | 3.6468s | 3.5808s | 3.7138s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=simple, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 1 | 0.2016s | 0.2027s | 0.1967s | 0.2102s | ok | median excludes warmup |
| MLP training | MiniTorch fast CPU | dataset=split, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 1 | 3.6641s | 3.6841s | 3.5994s | 3.8298s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=split, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 1 | 0.2040s | 0.2062s | 0.1984s | 0.2168s | ok | median excludes warmup |
| MLP training | MiniTorch fast CPU | dataset=xor, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10 | 5 | 1 | 3.6415s | 3.6652s | 3.5924s | 3.7807s | ok | median excludes warmup |
| MLP training | PyTorch CPU fair mini-batch | dataset=xor, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10 | 5 | 1 | 0.1964s | 0.1975s | 0.1936s | 0.2059s | ok | median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU (dataset=simple, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10): `3.6777, 3.7138, 3.6311, 3.6306, 3.5808`
- PyTorch CPU fair mini-batch (dataset=simple, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10): `0.2069, 0.2102, 0.2016, 0.1967, 0.1981`
- MiniTorch fast CPU (dataset=split, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10): `3.5994, 3.8298, 3.6641, 3.6891, 3.6381`
- PyTorch CPU fair mini-batch (dataset=split, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10): `0.2040, 0.1984, 0.2033, 0.2168, 0.2087`
- MiniTorch fast CPU (dataset=xor, points=250, hidden=10, rate=0.05, epochs=25, batch_size=10): `3.6860, 3.6254, 3.5924, 3.7807, 3.6415`
- PyTorch CPU fair mini-batch (dataset=xor, points=250, hidden=10, rate=0.5, epochs=25, batch_size=10): `0.1972, 0.1964, 0.1936, 0.1942, 0.2059`

## GPU Discussion

MiniTorch's CUDA backend is implemented with Numba CUDA kernels. GPU numbers are only publishable when the CUDA row completes successfully in the same benchmark environment.

CUDA was not requested for this run. Use `--include-cuda` to add MiniTorch CUDA rows.
