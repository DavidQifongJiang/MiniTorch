from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Convert vals to a list so we can modify it
    vals_forward = list(vals)
    vals_backward = list(vals)

    # Modify the argument of interest by adding and subtracting epsilon
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon

    # Compute the central difference approximation
    derivative = (f(*vals_forward) - f(*vals_backward)) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative value `x` for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns `True` if this variable is a leaf (input) in the graph."""
        ...

    def is_constant(self) -> bool:
        """Returns `True` if this variable is constant (not differentiable)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Applies the chain rule to compute gradients for parent variables."""
        if not self.parents:
            return []
        # Otherwise, return a list of tuples with each parent and the corresponding derivative
        return [(parent, d_output) for parent in self.parents]


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Perform a depth-first search to return the variables in topological order.

    Args:
    ----
        variable (Variable): The starting variable (typically the output node in a computation graph).

    Returns:
    -------
        Iterable[Variable]: The variables sorted in topological order.

    """
    visited = set()
    order = []

    def dfs(var: Variable) -> None:
        """Performs depth-first search on a variable."""
        if id(var) in visited or var.is_constant():
            return

        visited.add(id(var))  # Mark the variable as visited

        # Visit each parent (dependency) of the current variable
        for parent in var.parents:
            dfs(parent)

        # Add the variable to the order after processing its parents
        order.append(var)

    # Start DFS from the given variable
    dfs(variable)

    # Return the variables in reversed order (so that dependencies come before the node)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any = 1.0) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for all leaf nodes.

    Args:
    ----
        variable: The right-most variable in the computation graph.
        deriv: The initial derivative (typically 1.0 for the output variable).

    """
    # Step 1: Get the topological order of the computation graph
    ordered_vars = topological_sort(variable)

    # Step 2: Initialize a dictionary to store derivatives for each variable using id() for hashing
    derivatives = {id(variable): deriv}

    # Step 3: Traverse the variables in topological order
    for var in ordered_vars:
        var_id = id(var)
        if var.is_leaf():
            var.accumulate_derivative(derivatives.get(var_id, 0))
        else:
            for parent, parent_deriv in var.chain_rule(derivatives.get(var_id, 0)):
                parent_id = id(parent)
                if parent_id in derivatives:
                    derivatives[parent_id] += parent_deriv
                else:
                    derivatives[parent_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors used during the forward pass."""
        return self.saved_values
