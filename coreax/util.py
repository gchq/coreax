# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functionality to perform simple, generic tasks and operations.

The functions within this module are simple solutions to various problems or
requirements that are sufficiently generic to be useful across multiple areas of the
codebase. Examples of this include computation of squared distances, definition of
class factories and checks for numerical precision.
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jaxopt import OSQP

import coreax.coresubset as cc
import coreax.metrics as cm
import coreax.refine as cr
import coreax.weights as cw

#: Kernel evaluation function.
KernelFunction = Callable[[ArrayLike, ArrayLike], Array]

#: Pairwise kernel evaluation function if gradients and bandwidth are defined.
KernelFunctionWithGrads = Callable[
    [ArrayLike, ArrayLike, ArrayLike, ArrayLike, int, float], Array
]


class NotCalculatedError(Exception):
    """Raise when trying to use a variable that has not been calculated yet."""


def apply_negative_precision_threshold(
    x: ArrayLike, precision_threshold: float = 1e-8
) -> float:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Scalar value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :return: ``x``, rounded to 0.0 if it is between ``-precision_threshold`` and 0.0
    """
    # Cast to float. Will raise TypeError if array is not zero-dimensional.
    x = float(x)

    if precision_threshold < 0.0:
        raise ValueError(
            f"precision_threshold must be positive; value {precision_threshold} given."
        )

    if -precision_threshold < x < 0.0:
        return 0.0

    return x


@jit
def squared_distance(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate the squared distance between two vectors.

    :param x: First vector argument
    :param y: Second vector argument
    :return: Dot product of ``x - y`` and ``x - y``, the square distance between ``x``
        and ``y``
    """
    return jnp.dot(x - y, x - y)


@jit
def squared_distance_pairwise(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise square distance between two arrays.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise squared distances between ``x_array`` and ``y_array`` as an
        :math:`n \times m` array
    """
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    fn = vmap(
        vmap(squared_distance, in_axes=(None, 0), out_axes=0),
        in_axes=(0, None),
        out_axes=0,
    )
    return fn(x, y)


@jit
def difference(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate vector difference for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :return: Vector difference ``x - y``
    """
    return x - y


@jit
def pairwise_difference(x_array: ArrayLike, y_array: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise difference between two arrays of vectors.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise differences between ``x_array`` and ``y_array`` as an
        :math:`n \times m \times d` array
    """
    fn = vmap(
        vmap(difference, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1
    )
    return fn(x_array, y_array)


def solve_qp(kernel_mm: ArrayLike, kernel_matrix_row_sum_mean: ArrayLike) -> Array:
    r"""
    Solve quadratic programs with :mod:`jaxopt`.

    Solves simplex weight problems of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    :param kernel_mm: :math:`m \times m` coreset Gram matrix
    :param kernel_matrix_row_sum_mean: :math`m \times 1` array of Gram matrix means
    :return: Optimised solution for the quadratic program
    """
    # Setup optimisation problem - all variable names are consistent with the OSQP
    # terminology. Begin with the objective parameters
    q_array = jnp.array(kernel_mm)
    c = -jnp.array(kernel_matrix_row_sum_mean)

    # Define the equality constraint parameters
    num_points = q_array.shape[0]
    a_array = jnp.ones((1, num_points))
    b = jnp.array([1.0])

    # Define the inequality constraint parameters
    g_array = jnp.eye(num_points) * -1.0
    h = jnp.zeros(num_points)

    # Define solver object and run solver
    qp = OSQP()
    sol = qp.run(
        params_obj=(q_array, c), params_eq=(a_array, b), params_ineq=(g_array, h)
    ).params
    return sol.primal


def call_with_excess_kwargs(call_obj: Callable, *args, **kwargs) -> Any:
    """
    Call an object when invalid parameters have been provided as keyword arguments.

    Keyword arguments with invalid names are ignored.

    Positional arguments before keyword arguments will be passed without filtering. If
    too many positional arguments are passed, ``call_obj`` may raise an exception. If a
    keyword argument is also covered by a positional argument, it will be ignored.

    :param call_obj: Object to call
    :return: ``call_obj`` called with arguments with valid names
    """
    arguments = inspect.signature(call_obj).parameters
    # Construct list of argument names without positional arguments
    kw_names = tuple(arguments)[len(args) :]
    # Construct dictionary of keyword arguments to pass
    kw_args = {kw: kwargs[kw] for kw in kw_names if kw in kwargs}
    return call_obj(*args, **kw_args)


class ClassFactory:
    """
    Factory to return classes that can be looked up by name.

    Returned classes are uninstantiated objects.

    To use this factory, create an instance in the appropriate module and call
    :meth:`register` for each class to be registered. The instance may then be imported
    to wherever requires output from the factory.
    """

    def __init__(self, class_type: type):
        """
        Initialise factory.

        :param class_type: Type of class that factory produces
        """
        self.class_type = class_type
        self.lookup_table: dict[str, class_type] = {}

    def register(self, name: str, class_obj: type) -> None:
        """
        Register a class name.

        :param name: Name to identify class
        :param class_obj: Uninstantiated class to register
        :raises ValueError: If an object has already been registered with ``name``
        :raises TypeError: If ``obj`` does not match :attr:`~ClassFactory.obj_type`
        """
        if name in self.lookup_table:
            raise ValueError(f"{name} already used for {self.lookup_table[name]}.")
        if not isinstance(class_obj, type):
            raise TypeError("class_obj must be an uninstantiated class object.")
        if not issubclass(class_obj, self.class_type):
            raise TypeError(
                f"Class type {type(class_obj)} does not match type for class factory: "
                f"{self.class_type}."
            )
        self.lookup_table[name] = class_obj

    def get(self, name: str | type) -> type:
        """
        Get class by name or return input if an object of the required type is passed.

        Passing an object already of the required type permits dependency injection. The
        type is determined by :attr:`~ClassFactory.obj_type`.

        :param name: Registered name of class to fetch, or uninstantiated class object
            of a matching type
        :raises TypeError: If a non-string that does not match the factory type is
            passed, i.e. cannot be used for dependency injection
        :raises KeyError: If the string name is not recognised
        :return: Uninstantiated class object
        """
        # Check for dependency injection
        if isinstance(name, type):
            if issubclass(name, self.class_type):
                return name
            raise TypeError(f"{name} is not a subclass of {self.class_type}.")

        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string or a subclass of {self.class_type}."
            )

        # Get class by name
        class_obj = self.lookup_table.get(name)
        if class_obj is None:
            raise KeyError(f"Class name {name} not recognised.")
        return class_obj


def create_instance_from_factory(
    factory_obj: ClassFactory,
    class_type: str
    | type[cc.Coreset]
    | type[cm.Metric]
    | type[cr.Refine]
    | type[cw.WeightsOptimiser],
    **kwargs,
) -> cc.Coreset | cm.Metric | cr.Refine | cw.WeightsOptimiser:
    """
    Create a refine object for use with the fit method.

    :param class_type: The name of a class to use, or the uninstantiated class
        directly as a dependency injection
    :return: Class instance of the requested type
    """
    class_obj = factory_obj.get(class_type)

    # Initialise, accounting for different classes having different numbers of
    # parameters
    return call_with_excess_kwargs(
        class_obj,
        **kwargs,
    )
