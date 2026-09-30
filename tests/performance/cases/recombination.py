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

"""Performance cases for Caratheodory and tree recombination."""

import functools
import itertools

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array

from coreax import Data
from coreax.solvers import CaratheodoryRecombination, TreeRecombination
from coreax.util import JITCompilableFunction


@functools.cache
def _setup_problem(dimension: int, degree: int) -> tuple[Data, Array]:
    """Create a product cubature and polynomial test-function powers."""
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    product_nodes = np.asarray(list(itertools.product(nodes, repeat=dimension)))
    product_weights = np.asarray(list(itertools.product(weights, repeat=dimension)))
    data = Data(jnp.asarray(product_nodes), jnp.prod(product_weights, axis=1))

    powers = [
        power
        for power in itertools.product(range(degree + 1), repeat=dimension)
        if 0 < sum(power) <= degree
    ]
    return data, jnp.asarray(powers, dtype=jnp.int32)


def _evaluate_monomials(powers: Array, point: Array) -> Array:
    """Evaluate all configured monomials at one point."""
    return jnp.prod(jnp.power(point[jnp.newaxis, :], powers), axis=1)


def _test_functions(powers: Array):
    """Create a JAX-compatible callable for the monomial test functions."""
    return jtu.Partial(_evaluate_monomials, powers)


def _caratheodory_setup(dimension: int, degree: int) -> JITCompilableFunction:
    data, powers = _setup_problem(dimension, degree)
    solver = CaratheodoryRecombination(
        test_functions=_test_functions(powers), mode="implicit-explicit"
    )
    return JITCompilableFunction(
        fn=solver.reduce,
        fn_args=(data,),
        fn_kwargs=None,
        jit_kwargs=None,
        name=f"recombination_caratheodory_d{dimension}_degree{degree}",
    )


def _tree_setup(
    dimension: int, degree: int, tree_reduction_factor: int
) -> JITCompilableFunction:
    data, powers = _setup_problem(dimension, degree)
    solver = TreeRecombination(
        test_functions=_test_functions(powers),
        mode="implicit-explicit",
        tree_reduction_factor=tree_reduction_factor,
    )
    return JITCompilableFunction(
        fn=solver.reduce,
        fn_args=(data,),
        name=(
            f"recombination_tree_d{dimension}_degree{degree}"
            f"_factor{tree_reduction_factor}"
        ),
    )


def setup_recombination() -> list[JITCompilableFunction]:
    """
    Set up recombination scaling cases suggested by the algorithm review.

    The degree-three and degree-four cases increase the number of polynomial test
    functions from 19 to 34 in three dimensions. Tree recombination is also run with
    reduction factors two and four to expose the implementation's second scaling axis.
    """
    return [
        _caratheodory_setup(3, 3),
        _caratheodory_setup(3, 4),
        _tree_setup(3, 3, 2),
        _tree_setup(3, 4, 2),
        _tree_setup(3, 4, 4),
    ]
