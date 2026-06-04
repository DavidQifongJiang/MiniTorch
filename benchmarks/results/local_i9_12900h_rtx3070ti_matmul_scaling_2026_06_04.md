# MiniTorch Matrix Multiply Scaling Benchmark

This benchmark times preloaded square matrix multiplication across increasing sizes.
Tensor construction and host/device transfer are excluded so the result focuses on backend compute behavior.

## Environment

- Captured UTC: `2026-06-04T23:47:37.984317+00:00`
- Git commit: `253640d2b7b411fb9b41e973cf818726d751bf27`
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

| Size | Backend | Runs | Warmups | Median | Mean | Min | Max | Status | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 32 | MiniTorch fast CPU | 5 | 1 | 0.000294s | 0.000326s | 0.000290s | 0.000424s | ok | preloaded tensors; construction and host/device transfer excluded |
| 32 | MiniTorch CUDA | 5 | 1 | 0.003012s | 0.002908s | 0.002680s | 0.003121s | ok | preloaded tensors; construction and host/device transfer excluded |
| 32 | PyTorch CPU | 5 | 1 | 0.000003s | 0.000004s | 0.000003s | 0.000008s | ok | preloaded tensors; median excludes warmup |
| 64 | MiniTorch fast CPU | 5 | 1 | 0.001335s | 0.001317s | 0.001155s | 0.001402s | ok | preloaded tensors; construction and host/device transfer excluded |
| 64 | MiniTorch CUDA | 5 | 1 | 0.003107s | 0.003335s | 0.003068s | 0.003869s | ok | preloaded tensors; construction and host/device transfer excluded |
| 64 | PyTorch CPU | 5 | 1 | 0.000021s | 0.000024s | 0.000011s | 0.000045s | ok | preloaded tensors; median excludes warmup |
| 128 | MiniTorch fast CPU | 5 | 1 | 0.003955s | 0.003822s | 0.003585s | 0.004014s | ok | preloaded tensors; construction and host/device transfer excluded |
| 128 | MiniTorch CUDA | 5 | 1 | 0.005294s | 0.005238s | 0.004610s | 0.005839s | ok | preloaded tensors; construction and host/device transfer excluded |
| 128 | PyTorch CPU | 5 | 1 | 0.000021s | 0.000022s | 0.000021s | 0.000026s | ok | preloaded tensors; median excludes warmup |
| 256 | MiniTorch fast CPU | 5 | 1 | 0.021037s | 0.021748s | 0.020103s | 0.024146s | ok | preloaded tensors; construction and host/device transfer excluded |
| 256 | MiniTorch CUDA | 5 | 1 | 0.008230s | 0.008177s | 0.008001s | 0.008250s | ok | preloaded tensors; construction and host/device transfer excluded |
| 256 | PyTorch CPU | 5 | 1 | 0.000155s | 0.000181s | 0.000152s | 0.000282s | ok | preloaded tensors; median excludes warmup |
| 512 | MiniTorch fast CPU | 5 | 1 | 0.292770s | 0.289720s | 0.272833s | 0.304702s | ok | preloaded tensors; construction and host/device transfer excluded |
| 512 | MiniTorch CUDA | 5 | 1 | 0.026949s | 0.027150s | 0.026624s | 0.027852s | ok | preloaded tensors; construction and host/device transfer excluded |
| 512 | PyTorch CPU | 5 | 1 | 0.000960s | 0.001037s | 0.000920s | 0.001387s | ok | preloaded tensors; median excludes warmup |

## Raw Seconds

- MiniTorch fast CPU size=32: `0.000424, 0.000326, 0.000293, 0.000294, 0.000290`
- MiniTorch CUDA size=32: `0.003017, 0.003012, 0.002712, 0.002680, 0.003121`
- PyTorch CPU size=32: `0.000008, 0.000004, 0.000003, 0.000003, 0.000003`
- MiniTorch fast CPU size=64: `0.001155, 0.001381, 0.001402, 0.001315, 0.001335`
- MiniTorch CUDA size=64: `0.003869, 0.003068, 0.003091, 0.003540, 0.003107`
- PyTorch CPU size=64: `0.000045, 0.000025, 0.000017, 0.000021, 0.000011`
- MiniTorch fast CPU size=128: `0.004014, 0.003598, 0.003959, 0.003585, 0.003955`
- MiniTorch CUDA size=128: `0.004610, 0.005083, 0.005294, 0.005361, 0.005839`
- PyTorch CPU size=128: `0.000026, 0.000021, 0.000021, 0.000021, 0.000022`
- MiniTorch fast CPU size=256: `0.021037, 0.020995, 0.024146, 0.022460, 0.020103`
- MiniTorch CUDA size=256: `0.008001, 0.008247, 0.008230, 0.008156, 0.008250`
- PyTorch CPU size=256: `0.000154, 0.000282, 0.000164, 0.000155, 0.000152`
- MiniTorch fast CPU size=512: `0.298399, 0.304702, 0.272833, 0.292770, 0.279893`
- MiniTorch CUDA size=512: `0.027661, 0.026949, 0.026624, 0.027852, 0.026664`
- PyTorch CPU size=512: `0.001387, 0.000960, 0.000947, 0.000920, 0.000972`

## Interpretation

GPU speedups should only be claimed if CUDA wins at larger sizes in the same clean environment.
If CUDA remains slower, the likely bottlenecks are kernel-launch overhead, many small kernels, and non-fused educational kernels rather than raw GPU compute capacity.
