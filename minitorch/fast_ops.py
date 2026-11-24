"""Numba CPU kernels with small-workload dispatch and stride-aware indexing."""
from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from numba import njit as _njit, prange

from .tensor_data import shape_broadcast
from .tensor_ops import TensorOps, prepare_matmul

Fn = TypeVar("Fn")
PARALLEL_THRESHOLD = 32768


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Compile a kernel without Python object operations."""
    return _njit(inline="always", **kwargs)(fn)


def _compile_kernel(fn, kind="elementwise"):
    """Avoid thread-launch overhead for small operations; expose both variants."""
    serial = _njit(fn)
    parallel = _njit(parallel=True)(fn)

    def dispatch(*args):
        outputs = len(args[0])
        work = outputs
        if kind == "matmul":
            work *= args[4][-1]
        elif kind == "reduce":
            work *= args[4][args[6]]
        kernel = parallel if outputs >= 64 and work >= PARALLEL_THRESHOLD else serial
        return kernel(*args)

    dispatch.serial = serial
    dispatch.parallel = parallel
    dispatch.parallel_diagnostics = parallel.parallel_diagnostics
    return dispatch


@_njit(inline="always")
def _position(ordinal, shape, strides):
    ordinal = np.int64(ordinal)
    position = 0
    for dim in range(len(shape) - 1, -1, -1):
        coordinate = ordinal % shape[dim]
        ordinal //= shape[dim]
        position += coordinate * strides[dim]
    return position


@_njit(inline="always")
def _broadcast_position(ordinal, out_shape, shape, strides):
    ordinal = np.int64(ordinal)
    position = 0
    offset = len(out_shape) - len(shape)
    for dim in range(len(out_shape) - 1, -1, -1):
        coordinate = ordinal % out_shape[dim]
        ordinal //= out_shape[dim]
        source_dim = dim - offset
        if source_dim >= 0 and shape[source_dim] != 1:
            position += coordinate * strides[source_dim]
    return position


class FastOps(TensorOps):
    @staticmethod
    def map(fn):
        kernel = tensor_map(njit(fn))

        def ret(a, out=None):
            if out is None:
                out = a.empty(a.shape)
            kernel(*out.tuple(), *a.tuple())
            return out
        return ret

    @staticmethod
    def zip(fn):
        kernel = tensor_zip(njit(fn))

        def ret(a, b):
            out = a.empty(tuple(shape_broadcast(a.shape, b.shape)))
            kernel(*out.tuple(), *a.tuple(), *b.tuple())
            return out
        return ret

    @staticmethod
    def reduce(fn, start=0.0):
        kernel = tensor_reduce(njit(fn))

        def ret(a, dim):
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.empty(tuple(out_shape))
            out._tensor._storage[:] = start
            kernel(*out.tuple(), *a.tuple(), dim)
            return out
        return ret

    @staticmethod
    def matrix_multiply(a, b):
        a, b, shape, both_2d = prepare_matmul(a, b)
        out = a.empty(shape)
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
        return out.view(shape[1], shape[2]) if both_2d else out


def tensor_map(fn):
    """Unary map, including permuted output and broadcast input layouts."""
    def kernel(out, out_shape, out_strides, values, shape, strides):
        if np.array_equal(out_shape, shape) and np.array_equal(out_strides, strides):
            for i in prange(len(out)):
                out[i] = fn(values[i])
        else:
            for i in prange(len(out)):
                out[_position(i, out_shape, out_strides)] = fn(values[_broadcast_position(i, out_shape, shape, strides)])
    return _compile_kernel(kernel)


def tensor_zip(fn):
    """Binary map with an aligned-storage fast path and general broadcasting."""
    def kernel(out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
        aligned_a = np.array_equal(out_shape, a_shape) and np.array_equal(out_strides, a_strides)
        aligned_b = np.array_equal(out_shape, b_shape) and np.array_equal(out_strides, b_strides)
        if aligned_a and aligned_b:
            for i in prange(len(out)):
                out[i] = fn(a[i], b[i])
        elif aligned_a and len(b) == 1:
            for i in prange(len(out)):
                out[i] = fn(a[i], b[0])
        elif aligned_b and len(a) == 1:
            for i in prange(len(out)):
                out[i] = fn(a[0], b[i])
        else:
            for i in prange(len(out)):
                ai = _broadcast_position(i, out_shape, a_shape, a_strides)
                bi = _broadcast_position(i, out_shape, b_shape, b_strides)
                out[_position(i, out_shape, out_strides)] = fn(a[ai], b[bi])
    return _compile_kernel(kernel)


def tensor_reduce(fn):
    """Reduce one complete dimension without allocating index buffers."""
    def kernel(out, out_shape, out_strides, a, a_shape, a_strides, dim):
        for i in prange(len(out)):
            out_pos = _position(i, out_shape, out_strides)
            pos = _position(i, out_shape, a_strides)
            value = out[out_pos]
            for _ in range(a_shape[dim]):
                value = fn(value, a[pos])
                pos += a_strides[dim]
            out[out_pos] = value
    return _compile_kernel(kernel, "reduce")


def _tensor_matrix_multiply(out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
    """Parallelize independent output cells, including a single matrix batch."""
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    rows, columns = out_shape[1], out_shape[2]
    for ordinal in prange(len(out)):
        # Explicit signed conversion avoids uint64/int64 promotion in prange.
        index = np.int64(ordinal)
        batch = index // (rows * columns)
        row = (index // columns) % rows
        column = index % columns
        ai = batch * a_batch_stride + row * a_strides[1]
        bi = batch * b_batch_stride + column * b_strides[2]
        value = 0.0
        for _ in range(a_shape[2]):
            value += a[ai] * b[bi]
            ai += a_strides[2]
            bi += b_strides[1]
        out[batch * out_strides[0] + row * out_strides[1] + column * out_strides[2]] = value


tensor_matrix_multiply = _compile_kernel(_tensor_matrix_multiply, "matmul")
