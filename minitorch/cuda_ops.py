# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from minitorch.tensor import Tensor
from minitorch.tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from minitorch.tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # Add `Any` for kwarg
    """Apply JIT compilation to the provided function for device-specific operations.

    Args:
    ----
        fn (Callable[..., Any]): The function to JIT compile.
        **kwargs (Any): Additional keyword arguments for the JIT compilation.

    Returns:
    -------
        Callable[..., Any]: The JIT-compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # Add `Fn` type


def jit(fn, **kwargs) -> FakeCUDAKernel:  # type: ignore[ANN003]
    """Apply JIT compilation to the provided function.

    Args:
    ----
        fn (Callable[..., Any]): The function to JIT compile.
        **kwargs (Any): Additional keyword arguments for the JIT compilation.

    Returns:
    -------
        FakeCUDAKernel: The JIT-compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Make these always be a 3 dimensional multiply"""
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_ele = a_storage[index_to_position(a_index, a_strides)]
            b_ele = b_storage[index_to_position(b_index, b_strides)]
            out[index_to_position(out_index, out_strides)] = fn(a_ele, b_ele)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and an output array of size $n // \text{blockDIM}$,
    this kernel sums up each `blockDim` values into a corresponding output cell.

    Example:
    -------
        Input:  $[a_1, a_2, ..., a_{100}]$

        Output: $[a_1 + ... + a_{31}, a_{32} + ... + a_{64}, ..., ]$

    Note:
    ----
        Each block must perform the summation using shared memory.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): The length of the input tensor.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i < size:
        cache[pos] = float(a[i])
        cuda.syncthreads()
    else:
        cache[pos] = 0.0

    step = 1
    while step < BLOCK_DIM:
        if pos % (2 * step) == 0 and pos + step < BLOCK_DIM:
            cache[pos] += cache[pos + step]
        cuda.syncthreads()
        step *= 2

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the sum of elements in a 1D tensor using a practice CUDA kernel.

    This function calculates the sum of all elements in the input tensor `a` by
    utilizing the CUDA kernel `jit_sum_practice`. The summation is performed in
    parallel, using a specified number of threads per block and blocks per grid.
    The result is stored in a tensor `out` and returned to the caller.

    Args:
    ----
        a (Tensor): A 1D tensor whose elements need to be summed.

    Returns:
    -------
        TensorData: A tensor containing the computed sum in its first element.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        cache[pos] = reduce_value
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            dim = a_shape[reduce_dim]
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < dim:
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()
                idx = 0
                while 2**idx < BLOCK_DIM:
                    if pos % ((2**idx) * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + (2**idx)])
                        cuda.syncthreads()
                    idx += 1
            if pos == 0:
                out[index_to_position(out_index, out_strides)] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    if i >= 0 and i < size and j >= 0 and j < size:
        a_shared[local_i, local_j] = a[i * size + j]
        b_shared[local_i, local_j] = b[i * size + j]
        cuda.syncthreads()
        acc = 0.0
        for k in range(size):
            acc += a_shared[local_i, k] * b_shared[k, local_j]
        out[i * size + j] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication of two tensors using a practice CUDA kernel.

    This function computes the product of two square matrices `a` and `b`
    using the CUDA kernel `jit_mm_practice`. It initializes the output tensor,
    moves it to the CUDA device, and launches the kernel with the specified
    grid and block dimensions.

    Args:
    ----
        a (Tensor): The first input tensor of shape (size, size).
        b (Tensor): The second input tensor of shape (size, size).

    Returns:
    -------
        TensorData: The resulting tensor of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # TODO: Implement for Task 3.4.
    # 1. Retrieve matrix dimensions for subsequent computations
    # Matrices 'a' and 'b' have shapes (batch_a, Rows, K_dim) and (batch_b, K_dim, Columns) respectively
    # The output is computed as: out[n, row, col] = sum_over_idx a[n_a, row, idx] * b[n_b, idx, col]

    Rows, Columns, K_dim = out_shape[1], out_shape[2], a_shape[-1]
    accumulator = 0.0

    # 2. For out[row, col], shared memory blocks must contain segments of a[row, ...] and b[..., col]
    # Loop to copy data into shared memory blocks
    for offset in range(0, K_dim, BLOCK_DIM):
        a_idx = offset + pj
        if i < Rows and a_idx < K_dim:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + a_idx * a_strides[2]
            ]
        # Compute row index for `b` in the current segment.
        b_idx = offset + pi

        # Check bounds for valid indices in `b`.
        if b_idx < K_dim and j < Columns:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + b_idx * b_strides[1] + j * b_strides[2]
            ]
        # Synchronize threads after copying data
        cuda.syncthreads()
        # 3. After each block copy from 'a' and 'b', compute partial dot products and accumulate them
        for idx in range(BLOCK_DIM):
            if offset + idx < K_dim:
                accumulator += a_shared[pi, idx] * b_shared[idx, pj]

    if i < Rows and j < Columns:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = (
            accumulator
        )


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
