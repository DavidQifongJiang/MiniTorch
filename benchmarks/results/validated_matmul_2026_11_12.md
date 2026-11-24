# MiniTorch Matrix Multiply Scaling Benchmark

This benchmark times preloaded square matrix multiplication across increasing sizes.
Input construction and validation are outside timing; dispatch, output allocation, and CUDA synchronization are included.
Every output is checked against NumPy einsum; a failed validation produces no timing claim.

## Environment

- Captured UTC: `2026-11-12T18:08:56.420015+00:00`
- Git commit: `836e9e2b26905b8765a4a75c4852d15defac8b45`
- Git status: dirty
- Python: `3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:36:00) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26300-SP0`
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

| Size | Backend | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 32 | MiniTorch fast CPU | 5 | 1 | 0.000206s | 0.000201s | 0.000156s | 0.000259s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 32 | MiniTorch CUDA | 5 | 1 | 0.000554s | 0.000594s | 0.000514s | 0.000783s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 32 | PyTorch CPU | 5 | 1 | 0.000020s | 0.000031s | 0.000010s | 0.000082s | ok | preloaded tensors; median excludes warmup; every output checked outside timing |
| 64 | MiniTorch fast CPU | 5 | 1 | 0.000577s | 0.000665s | 0.000368s | 0.001116s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 64 | MiniTorch CUDA | 5 | 1 | 0.000701s | 0.000726s | 0.000553s | 0.000864s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 64 | PyTorch CPU | 5 | 1 | 0.000090s | 0.000082s | 0.000029s | 0.000158s | ok | preloaded tensors; median excludes warmup; every output checked outside timing |
| 128 | MiniTorch fast CPU | 5 | 1 | 0.001283s | 0.001282s | 0.001068s | 0.001694s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 128 | MiniTorch CUDA | 5 | 1 | 0.000639s | 0.000696s | 0.000550s | 0.001059s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 128 | PyTorch CPU | 5 | 1 | 0.000107s | 0.000109s | 0.000092s | 0.000123s | ok | preloaded tensors; median excludes warmup; every output checked outside timing |
| 256 | MiniTorch fast CPU | 5 | 1 | 0.008193s | 0.008323s | 0.007893s | 0.008942s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 256 | MiniTorch CUDA | 5 | 1 | 0.001257s | 0.001449s | 0.001157s | 0.002016s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 256 | PyTorch CPU | 5 | 1 | 0.000430s | 0.000454s | 0.000314s | 0.000693s | ok | preloaded tensors; median excludes warmup; every output checked outside timing |
| 512 | MiniTorch fast CPU | 5 | 1 | 0.109205s | 0.110011s | 0.105527s | 0.116079s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 512 | MiniTorch CUDA | 5 | 1 | 0.003959s | 0.003940s | 0.003820s | 0.004036s | ok | preloaded inputs; synchronized operation includes dispatch/allocation; every output checked outside timing |
| 512 | PyTorch CPU | 5 | 1 | 0.001836s | 0.002076s | 0.001641s | 0.002570s | ok | preloaded tensors; median excludes warmup; every output checked outside timing |

## Raw Seconds

- MiniTorch fast CPU size=32: `0.000259, 0.000206, 0.000214, 0.000169, 0.000156`
- MiniTorch CUDA size=32: `0.000783, 0.000589, 0.000532, 0.000554, 0.000514`
- PyTorch CPU size=32: `0.000082, 0.000030, 0.000020, 0.000013, 0.000010`
- MiniTorch fast CPU size=64: `0.000809, 0.001116, 0.000577, 0.000453, 0.000368`
- MiniTorch CUDA size=64: `0.000553, 0.000850, 0.000864, 0.000661, 0.000701`
- PyTorch CPU size=64: `0.000090, 0.000158, 0.000095, 0.000036, 0.000029`
- MiniTorch fast CPU size=128: `0.001694, 0.001283, 0.001072, 0.001068, 0.001293`
- MiniTorch CUDA size=128: `0.000639, 0.000590, 0.000550, 0.000641, 0.001059`
- PyTorch CPU size=128: `0.000123, 0.000123, 0.000092, 0.000107, 0.000099`
- MiniTorch fast CPU size=256: `0.008942, 0.008193, 0.007893, 0.008575, 0.008013`
- MiniTorch CUDA size=256: `0.001257, 0.001157, 0.002016, 0.001565, 0.001251`
- PyTorch CPU size=256: `0.000314, 0.000693, 0.000430, 0.000390, 0.000443`
- MiniTorch fast CPU size=512: `0.109204, 0.116079, 0.109205, 0.110037, 0.105527`
- MiniTorch CUDA size=512: `0.003891, 0.003820, 0.004036, 0.003996, 0.003959`
- PyTorch CPU size=512: `0.001836, 0.001822, 0.001641, 0.002511, 0.002570`

## Interpretation

GPU speedups should only be claimed if CUDA wins at larger sizes in the same clean environment.
If CUDA remains slower, the likely bottlenecks are kernel-launch overhead, many small kernels, and non-fused educational kernels rather than raw GPU compute capacity.
