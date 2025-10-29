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


def is_close(x: float, y: float) -> bool:
    """Returns the larger of two numbers"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(0, x)


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
def map(f: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Apply function `f` to each element of the iterable."""
    return (f(x) for x in iterable)


# - zipWith
def zipWith(
    f: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Combine elements from two iterables using function `f`."""
    return (f(x, y) for x, y in zip(lst1, lst2))


# - reduce
def reduce(
    f: Callable[[float, float], float], iterable: Iterable[float], initial: float
) -> float:
    """Reduce the iterable to a single value using function `f` and an initial value."""
    result = initial
    for x in iterable:
        result = f(result, x)
    return result


# Use these to implement
# - negList : negate a list
def negList(iterable: Iterable[float]) -> Iterable[float]:
    """Negate all elements in an iterable."""
    return map(lambda x: -x, iterable)


# - addLists : add two lists together
def addLists(iterable1: Iterable[float], iterable2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two iterables."""
    return zipWith(lambda x, y: x + y, iterable1, iterable2)


# - sum: sum lists
def sum(iterable: Iterable[float]) -> float:
    """Sum all elements in an iterable."""
    return reduce(lambda x, y: x + y, iterable, 0)


# - prod: take the product of lists
def prod(iterable: Iterable[float]) -> float:
    """Calculate the product of all elements in an iterable."""
    return reduce(lambda x, y: x * y, iterable, 1)


# TODO: Implement for Task 0.3.
