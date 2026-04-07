from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Apply the Numba JIT (Just-In-Time) compilation to the provided function for performance optimization.

    Args:
    ----
        fn (Fn): The function to be compiled.
        **kwargs (Any): Additional options to configure the JIT behavior.

    Returns:
    -------
        Fn: The compiled function optimized for execution.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.
    for i in prange(out_size):
        out_index: Index = np.zeros(3, np.int32)
        in_index: Index = np.zeros(3, np.int32)
        weights_index: Index = np.zeros(3, np.int32)

        to_index(i, out_shape, out_index)
        cur_batch, cur_out_channels, cur_width = out_index
        val = 0.0
        for c1 in range(in_channels):
            for c2 in range(kw):
                if not reverse:
                    if (cur_width + c2) >= width:
                        val += 0.0
                    else:
                        weights_index[0] = cur_out_channels
                        weights_index[1] = c1
                        weights_index[2] = c2
                        in_index[0] = cur_batch
                        in_index[1] = c1
                        in_index[2] = cur_width + c2
                        val += (
                            input[index_to_position(in_index, s1)]
                            * weight[index_to_position(weights_index, s2)]
                        )
                else:
                    if (cur_width - c2) < 0:
                        val += 0.0
                    else:
                        weights_index[0] = cur_out_channels
                        weights_index[1] = c1
                        weights_index[2] = c2
                        in_index[0] = cur_batch
                        in_index[1] = c1
                        in_index[2] = cur_width - c2
                        val += (
                            input[index_to_position(in_index, s1)]
                            * weight[index_to_position(weights_index, s2)]
                        )

        out[index_to_position(out_index, out_strides)] = val


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for the convolution operation.

        Args:
        ----
        ctx (Context): The context object that stores information from the forward pass, including saved tensors.
        grad_output (Tensor): The gradient of the output tensor with respect to the loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]: A tuple containing:
            - The gradient of the input tensor.
            - The gradient of the weight tensor.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    for out_i in prange(out_size):
        out_index = np.zeros(4, dtype=np.int32)
        to_index(out_i, out_shape, out_index)
        current_batch, current_out_channels, current_height, current_width = out_index
        val = 0
        for current_in_channels in prange(in_channels):
            for current_kh in range(kh):
                for current_kw in range(kw):
                    i = current_kh
                    j = current_kw
                    if reverse:
                        j = kw - current_kw - 1
                        i = kh - current_kh - 1
                    w = weight[
                        s20 * current_out_channels
                        + s21 * current_in_channels
                        + s22 * i
                        + s23 * j
                    ]
                    inc = 0
                    if reverse:
                        if current_height - i >= 0 and current_width - j >= 0:
                            inc = input[
                                current_batch * s10
                                + current_in_channels * s11
                                + (current_height - i) * s12
                                + (current_width - j) * s13
                            ]
                    else:
                        if i + current_height < height and j + current_width < width:
                            inc = input[
                                current_batch * s10
                                + current_in_channels * s11
                                + (i + current_height) * s12
                                + (j + current_width) * s13
                            ]
                    val += w * inc
        out[
            current_batch * out_strides[0]
            + current_out_channels * out_strides[1]
            + current_height * out_strides[2]
            + current_width * out_strides[3]
        ] = val


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for the convolution operation.

        Args:
        ----
        ctx (Context): The context object that stores information from the forward pass, including saved tensors.
        grad_output (Tensor): The gradient of the output tensor with respect to the loss.

        Returns:
        -------
        Tuple[Tensor, Tensor]: A tuple containing:
            - The gradient of the input tensor.
            - The gradient of the weight tensor.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
