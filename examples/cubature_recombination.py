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
End-to-end example for using recombination solvers.

In this example we demonstrate how the recombination solvers in Coreax can be used to
find non-product Cubature formulae, given product Cubature formulae over the `n`-cube.
These formulae can be used to exactly (*up to finite-precision arithmetic*) integrate
all `n`-variate polynomials of, at most, some given degree `m`, over the `n`-cube.
"""

import functools as ft
import itertools
from collections.abc import Iterable
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float, Int

from coreax import Coresubset, Data
from coreax.solvers import TreeRecombination


# First we define some functions for generating a baseline product Cubature and the
# `n`-variate monomial test-functions which the product Cubature and the recombined
# cubature (when it is computed later on) should integrate exactly over the `n`-cube.
def leggauss_product_cubature(dimension: int, degree: int) -> tuple[Array, Array]:
    """
    Construct a Legendre-Gauss product cubature over the `n`-cube.

    :param dimension: Dimension `n` of the `n`-cube (product space/domain)
    :param degree: The algebraic degree `m` of all multi-variate polynomials that the
        product cubature exactly integrates, over the `n`-cube (product space/domain)
    :return: Nodes and weights for the product cubature formula.
    """
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    prod_nodes = np.fromiter(
        itertools.product(nodes, repeat=dimension),
        dtype=np.dtype((np.float32, dimension)),
    )
    prod_weights_un_multiplied = np.fromiter(
        itertools.product(weights, repeat=dimension),
        dtype=np.dtype((np.float32, dimension)),
    )
    return jnp.asarray(prod_nodes), jnp.prod(prod_weights_un_multiplied, axis=-1)


def monomial_power_generator(
    dimension: int, max_degree: int, *, mode: Literal["all", "even", "odd"] = "all"
) -> Iterable[tuple[int, ...]]:
    """
    Return a generator for all combinations of `n`-variate monomial powers.

    :param dimension: Number of unique variates `n`; dimension of the domain over which
        the monomials are defined
    :param max_degree: Maximal degree of any monomial power `m`; equal to the sum of
        powers for each variate in a given monomial E.G :math:`xy**2` has degree three
        :math:`xy` has degree two, :math:`x` has degree one, etc...
    :param mode: If to return 'all', 'even', or 'odd' `n`-variate monomial powers; a
        monomial is even if and only if its powers sum to a non-zero even value. E.G.
        :math:`xy` is even, :math:`xy**2` is odd, and :math:`1 == x^0 y^0` is odd.
    :return: Generator for all multi-variate monomial powers of the specified dimension
        and maximum degree.
    """
    monomial_degree_generator = itertools.product(
        range(max_degree + 1),
        repeat=dimension,
    )
    exact_monomial_degree_generator = itertools.filterfalse(
        lambda x: sum(x) > max_degree,
        monomial_degree_generator,
    )
    if mode == "all":
        return exact_monomial_degree_generator
    if mode == "even":
        return itertools.filterfalse(
            lambda x: sum(x) == 0 or sum(x) % 2 != 0,
            exact_monomial_degree_generator,
        )
    if mode == "odd":
        return itertools.filterfalse(
            lambda x: sum(x) % 2 == 0, exact_monomial_degree_generator
        )
    raise ValueError("Invalid mode; must be one of ['all', 'even', 'odd'].")


def test_functions_from_monomial_powers(
    monomial_powers: Int[Array, "k n"],
) -> Callable[[Float[Array, " n"]], Float[Array, " k"]]:
    """
    Construct test functions given a set of `n`-variate monomial powers.

    :param monomial_powers: The integer powers for `k` monomials in `n` variates
    :return: A callable that evaluates a passed `n`-vector for the `k` monomials
        defined by `monomial_powers`; can be used as the `test_functions` for a
        :class:`~coreax.solvers.RecombinationSolver`.
    """
    _, n = monomial_powers.shape
    maximum_degree = jnp.max(jnp.sum(monomial_powers, axis=-1))
    coefficients = jnp.zeros((maximum_degree + 1, n))
    column_index = jnp.arange(n)

    @jax.vmap
    def polyval_coefficients(_power: Int[Array, "k n"]) -> Int[Array, "k m+1 n"]:
        """
        Create a polyval coefficient matrix given an `n`-variate monomial power.

        :param _power: Powers for `k` monomials in `n` variates
        :return: `k` polyval coefficient arrays where, for a given array, the [i,j]
            index gives the `j`-th variate raised to the `m+1-i`-th power, where `m+1`
            is the maximal degree of any of the `k` monomials plus one (the plus one is
            required to account for the zero-degree coefficients required by polyval).
        """
        return coefficients.at[maximum_degree - _power, column_index].set(1)

    @ft.partial(jax.vmap, in_axes=(0, None))
    def monomial_test_function(
        _coefficients: Int[Array, "k m+1 n"], x: Float[Array, " n"]
    ) -> Float[Array, " k"]:
        """
        Evaluate an `n`-variate monomial.

        :param _coefficients: `k` polyval coefficient arrays of maximum degree `m`; the
            plus one in the type annotation is required to account for the zero-degree
            coefficients required by polyval
        :param x: An `n`-variate array at which to evaluate the monomial
        :return: The `k` monomials, defined by `_coefficients`, evaluated at `x`.
        """
        return jnp.prod(jnp.polyval(_coefficients, x))

    return jtu.Partial(monomial_test_function, polyval_coefficients(monomial_powers))


def main(dimension: int = 3, max_degree: int = 4) -> Coresubset:
    r"""
    Run the 'recombination' example for finding non-product cubature formulae.

    Generates the non-product (recombined) cubature formulae over the `n`-cube, for all
    `n`-variate polynomials of maximal degree `m`, given a product Legendre-Gauss
    cubature formulae over the same domain with the same degree `m`.

    .. note::
        Cubature formulae are simply quadrature formulae for `n`-dimensional domains.
        However, it should be noted that many authors do not make this distinction and
        simply refer to all formulae as ``quadratures``.

    An `N`-point cubature formulae of degree `m`, with respect to some positive Borel
    measure :math:`\mu` on :math:`X \subseteq \mathbb{R}^n`, is any discrete measure
    :math:`\mathbb{Q}^N := \sum_{i=1}^N \lambda_i \delta_{x_i}`, with positive weights
    :math:`\lambda \in \mathbb{R}_+` and nodes :math:`x_i \in \text{supp}(X)`, for which
    the following equation

    .. math::
        \int_X f(x) d\mu(x) = \int_X f(x) d\mathbb{Q}^N(x):=\sum_{i=1}^N\lambda_i f(x_i)

    holds for all `n`-variate polynomials of degree `m`, :math:`f \in \mathcal{P}_m(X)`.

    For the purposes of computational efficiency, the number of points `N` in any given
    cubature formula should be minimal. Hence, recombination is used in this example to
    find a non-product formula with :math:`N_\text{recombination} \le N` points, where
    `N` is the number of points in the Legendre-Gauss product formula over the `n`-cube.

    :param dimension: Number of unique variates `n`; dimension of the domain over which
        the monomials are defined
    :param max_degree: Maximal degree of any monomial power `m`; equal to the sum of
        powers for each variate in a given monomial E.G :math:`xy**2` has degree three
        :math:`xy` has degree two, :math:`x` has degree one, etc...
    :return: The coresubset (non-product) cubature.
    """
    # Using the above helper functions, we can generate a product Cubature and its
    # associated test-functions for the given dimension `n` and degree `m`.
    product_nodes, product_weights = leggauss_product_cubature(dimension, max_degree)
    product_cubature = Data(product_nodes, product_weights)
    print(f"Product Cubature:\n\t node_count: {len(product_nodes)}")
    monomial_powers = np.fromiter(
        monomial_power_generator(dimension, max_degree),
        dtype=np.dtype((np.int32, dimension)),
    )
    test_functions = test_functions_from_monomial_powers(jnp.asarray(monomial_powers))
    test_functions_shape = jax.eval_shape(test_functions, jnp.zeros((dimension,)))
    print(f"Test Functions:\n\t count: {len(test_functions_shape)}")

    # The recombination algorithm in Coreax can now be applied to the above generated
    # product Cubature, as follows:
    solver = TreeRecombination(test_functions=test_functions, mode="explicit")
    coresubset_cubature, _ = solver.reduce(product_cubature)
    coresubset_nodes, coresubset_weights = jtu.tree_leaves(coresubset_cubature.points)
    print(f"Recombined Cubature:\n\t node_count: {len(coresubset_nodes)}")

    # The product Cubature and the recombined (Coresubset) Cubature should evaluate to
    # the same integrals for all test-functions in `test_functions`, up to some
    # normalizing constant that can be easily determined given both formulae.
    vmap_test_functions = jax.vmap(test_functions)
    pushed_forward_product_cubature = vmap_test_functions(product_nodes)
    pushed_forward_coresubset_cubature = vmap_test_functions(coresubset_nodes)

    # By using `jnp.average` we are implicitly normalizing the product cubature; the
    # coresubset cubature is already normalized.
    product_cubature_integral = jnp.average(
        pushed_forward_product_cubature, axis=0, weights=product_weights
    )
    coresubset_cubature_integral = jnp.average(
        pushed_forward_coresubset_cubature, axis=0, weights=coresubset_weights
    )

    # Check equality up to a reasonable relative and absolute tolerance, given the use
    # of finite-precision arithmetic.
    print(f"Expected integrals: {product_cubature_integral}")
    print(f"Cubature integrals: {coresubset_cubature_integral}")
    in_tolerance = jnp.isclose(
        product_cubature_integral, coresubset_cubature_integral, rtol=1e-5, atol=1e-6
    )
    print(f"All within tolerance: {all(in_tolerance)}")
    if not all(in_tolerance):
        raise RuntimeError("Recombination failed for an unexpected reason.")

    return coresubset_cubature


if __name__ == "__main__":
    main()
