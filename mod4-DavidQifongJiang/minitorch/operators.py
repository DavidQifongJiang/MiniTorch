"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
def mul(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


def id(x: float) -> float:
    """Return the input value unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Makes it negative"""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    # ASSIGN0.1
    return (x - y < 1e-2) and (y - x < 1e-2)
    # END ASSIGN0.1


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg"""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return d if x > 0 else 0


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions


# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
    fn: Function from one value to one value.

    Returns:
    -------
    A function that takes a list, applies `fn` to each element, and returns a new list

    """

    # ASSIGN0.3
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map
    # END ASSIGN0.3


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
    fn: combine two values

    Returns:
    -------
    Function that takes two equally sized lists `ls1` and `ls2`, produce a new list
    by applying fn(x, y) on each pair of elements.

    """

    # ASSIGN0.3
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith
    # END ASSIGN0.3


# - reduce
def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
    fn: combine two values
    start: start value $x_0$

    Returns:
    -------
    Function that takes a list `ls` of elements
    $x_1 \dots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2, fn(x_1, x_0)))`

    """

    # ASSIGN0.3
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce
    # END ASSIGN0.3


# Use these to implement
# - negList : negate a list
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)


# - addLists : add two lists together
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two iterables."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    # ASSIGN0.3
    return reduce(mul, 1.0)(ls)
    # END ASSIGN0.3


# TODO: Implement for Task 0.3.
