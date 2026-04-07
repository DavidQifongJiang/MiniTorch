from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod

import minitorch

from minitorch import operators
from minitorch.autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use `class`here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    @abstractmethod
    def forward(cls, ctx: Context, *inputs: float) -> float:
        """Forward pass."""
        pass  # Must be implemented by subclasses

    @classmethod
    @abstractmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass."""
        pass  # Must be implemented by subclasses

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to the given Scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function f(x, y) = x + y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns the result of a + b."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns the gradient of addition for both inputs."""
        return d_output, d_output


class Sub(ScalarFunction):
    """Sub function f(x, y) = x - y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns the result of a - b."""
        return a - b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns the gradient of Subraction for both inputs."""
        return d_output, -d_output


class Log(ScalarFunction):
    """Logarithm function f(x) = log(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns log(a)."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of log with respect to the input."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns the result of a * b."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns the gradients of multiplication for both inputs."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns 1 / a."""
        ctx.save_for_backward(a)
        return float(1 / a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of 1 / a."""
        (a,) = ctx.saved_values
        return -d_output / (a**2)


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns -a."""
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of -a."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + e^{-x})"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns the sigmoid of a."""
        result = 1 / (1 + operators.exp(-a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of sigmoid."""
        (sigmoid_value,) = ctx.saved_values
        return d_output * sigmoid_value * (1 - sigmoid_value)


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns max(0, a)."""
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of ReLU."""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0


class Exp(ScalarFunction):
    """Exponential function f(x) = e^x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns e^a."""
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return float(result)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the gradient of e^a."""
        (result,) = ctx.saved_values
        return d_output * result


class LT(ScalarFunction):
    """Less than function f(x, y) = x < y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns whether a < b."""
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns zero gradients for a comparison."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function f(x, y) = x == y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns whether a == b."""
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns zero gradients for an equality comparison."""
        return 0.0, 0.0
