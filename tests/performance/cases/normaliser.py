"""A simple test that serves to normalise other performance test results."""

from jax import Array
from jax import numpy as jnp
from jax import random as jr

from coreax.util import JITCompilableFunction


def _setup_data() -> Array:
    """Set up a large random matrix to be operated on."""
    key = jr.key(1_234)
    matrix = jr.normal(key, (2500, 2500))
    return matrix


def _operate(matrix: Array) -> Array:
    """Perform some arbitrary operations on a matrix."""
    squared = jnp.square(matrix)
    halved_transpose = matrix.T / 2
    result = squared @ halved_transpose
    return result.trace()


def setup_normaliser() -> JITCompilableFunction:
    """Set up a test against which other performance tests will be normalised."""
    matrix = _setup_data()
    return JITCompilableFunction(
        fn=_operate,
        fn_args=(matrix,),
        fn_kwargs=None,
        jit_kwargs=None,
        name="NORMALISER",
    )
