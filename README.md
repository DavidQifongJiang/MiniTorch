# MiniTorch

MiniTorch is a PyTorch-style deep learning framework built from scratch in Python to reimplement the core layers of modern ML systems: automatic differentiation, multidimensional tensors, optimized CPU kernels, CUDA acceleration, and neural-network training.

Instead of treating frameworks like PyTorch as a black box, MiniTorch rebuilds the stack from first principles:

- mathematical operators and functional primitives
- scalar reverse-mode autodiff
- tensor abstractions with broadcasting and shape transforms
- optimized CPU tensor kernels with Numba
- CUDA kernels for map / zip / reduce / matrix multiply
- neural-network primitives and end-to-end training demos

The goal of this project is both correctness and systems understanding: not just training a model, but understanding how gradients, tensor layouts, parallel kernels, and backend design work together.

---

## What MiniTorch Supports

### Core framework features
- Reverse-mode automatic differentiation
- Computation graph construction and backpropagation
- Scalar and tensor abstractions
- Broadcasting-aware gradient propagation
- Reshape, permute, reduction, and elementwise tensor ops
- Batched matrix multiplication
- Backend-aware execution for CPU and GPU

### Performance features
- Numba-optimized CPU tensor kernels
- Parallelized map / zip / reduce kernels
- CUDA kernels with shared-memory optimizations
- Batched matrix multiplication on GPU

### Neural-network features
- ReLU, sigmoid, log, exp
- Softmax and log-softmax
- Max reduction and max pooling
- Average pooling
- Dropout
- End-to-end MLP training on nonlinear datasets

---

## Why This Project Matters

Most ML projects show how to use frameworks.

MiniTorch shows how a framework works internally.

This project demonstrates:
- understanding of reverse-mode autodiff instead of only calling `.backward()`
- understanding of tensor broadcasting and multidimensional gradient propagation
- understanding of backend design and kernel-level optimization
- ability to connect theory, correctness checks, and performance engineering

That makes MiniTorch especially relevant for:
- Machine Learning Engineer roles
- ML Systems / Infrastructure roles
- Deep Learning Systems roles
- Performance-oriented software engineering roles

---

## Architecture Overview

MiniTorch is organized as a progression from mathematical foundations to end-to-end training:

### Module 0 — Foundations
Implements the core mathematical operators and higher-order functional primitives used across the framework.

Examples:
- `add`, `mul`, `neg`, `sigmoid`, `relu`, `log`, `exp`
- `map`, `zipWith`, `reduce`

### Module 1 — Scalar Autodiff
Builds scalar reverse-mode autodiff from scratch.

Key ideas:
- scalar computation graphs
- operator overloading
- topological ordering
- chain-rule-based backpropagation
- finite-difference derivative checking

### Module 2 — Tensor Engine
Generalizes autodiff from scalars to multidimensional tensors.

Key ideas:
- tensor storage, shape, and stride handling
- broadcasting-aware operations
- reshape / permute / reduction support
- backend-agnostic tensor interface
- gradient propagation across tensor operations

### Module 3 — Performance Backends
Adds optimized execution backends.

CPU backend:
- Numba JIT compilation
- parallelized map / zip / reduce
- optimized batched matrix multiplication

GPU backend:
- CUDA kernels for tensor map / zip / reduce
- shared-memory matrix multiplication
- GPU execution for tensor operations and training workloads

### Module 4 — Neural-Network Primitives
Builds higher-level ML functionality on top of the tensor engine.

Includes:
- pooling
- softmax / log-softmax
- dropout
- max reduction
- small MLP training demos on nonlinear datasets

---

## Benchmarks

MiniTorch includes both correctness-oriented and performance-oriented validation.

### Current measured results
- Up to **~8× CPU speedup** on large tensors (for example 1024×1024) when comparing naive and optimized tensor backends
- Reduced end-to-end training time from **~3.2s to ~1.4s per epoch** after integrating the GPU backend (**~2.3× speedup**)
- Verified training on multiple nonlinear classification tasks including XOR

### Benchmarking notes
To make these results fully reproducible, this repository should include:
- benchmark scripts
- hardware information
- tensor sizes
- backend settings
- run instructions

A future README update should include a full benchmark table like this:

| Workload | Backend A | Backend B | Result |
|---|---:|---:|---:|
| Elementwise tensor ops (1024×1024) | naive CPU | optimized CPU | ~8× faster |
| MLP training epoch | CPU | GPU | ~2.3× faster |
| Nonlinear classification | autodiff framework | training stable | converges |

---

## Example Capabilities

MiniTorch supports end-to-end model execution rather than isolated operator demos.

Examples include:
- training multilayer perceptrons on nonlinear toy datasets
- validating gradient correctness through derivative checking
- running tensor operations on CPU and GPU backends
- applying softmax, dropout, and pooling built on the same tensor engine

---

## Project Structure

This repository currently reflects the framework as a staged build-up across modules.

Suggested interpretation:

- `mod0-*` — operator foundations
- `mod1-*` — scalar autodiff
- `mod2-*` — tensor abstraction
- `mod3-*` — optimized CPU and CUDA backends
- `mod4-*` — neural-network primitives and training demos

A future public-facing cleanup should consolidate the final implementation into a cleaner structure such as:

```text
minitorch/
examples/
benchmarks/
tests/
docs/
archive/
