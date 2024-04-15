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
Tests for kernel implementations.

The tests within this file verify that the implementations of kernels used throughout
the codebase produce the expected results on simple examples.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from jax import Array, tree_util
from jax import numpy as jnp
from jax.typing import ArrayLike
from scipy.stats import norm as scipy_norm

import coreax.approximation
import coreax.kernel
import coreax.util


class KernelNoDivergenceMethod:
    """
    Example kernel with no method to compute divergence of inputs.

    This kernel is used to verify handling of invalid inputs to Stein kernels.
    """

    def __init__(self, a: float):
        """Initialise the KernelNoDivergenceMethod class."""
        self.a = a

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        children = ()
        aux_data = {"a": self.a}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable JIT decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate kernel on two inputs ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        return self.a * (x + y)


# Register the example kernel
tree_util.register_pytree_node(
    KernelNoDivergenceMethod,
    KernelNoDivergenceMethod.tree_flatten,
    KernelNoDivergenceMethod.tree_unflatten,
)


class KernelNoTreeFlatten:
    """
    Example kernel with no method to flatten a pytree.

    This kernel is used to verify handling of invalid inputs to Stein kernels.
    """

    def __init__(self, a: float):
        """Initialise the KernelNoTreeFlatten class."""
        self.a = a

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable JIT decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate kernel on two inputs ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        return self.a * (x + y)


class KernelNoTreeUnflatten:
    """
    Example kernel with no method to unflatten a pytree.

    This kernel is used to verify handling of invalid inputs to Stein kernels.
    """

    def __init__(self, a: float):
        """Initialise the KernelNoTreeUnflatten class."""
        self.a = a

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        # The score function is assumed to not change here - but it might if the kernel
        # changes - but this does not work when kernel is specified in children
        children = ()
        aux_data = {"a": self.a}
        return children, aux_data

    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate kernel on two inputs ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        return self.a * (x + y)


class TestKernelABC(unittest.TestCase):
    """
    Tests related to the Kernel abstract base class in ``kernel.py``.
    """

    def setUp(self):
        """
        Generate data for shared use across unit tests.
        """
        self.default_length_scale = 1 / np.sqrt(2)
        self.default_x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        self.default_zero_kernel_row_sum = jnp.zeros(len(self.default_x))

        # Define the simplest, real kernel object for testing purposes
        self.default_kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=self.default_length_scale
        )

    def test_approximator_valid(self) -> None:
        """
        Test usage of approximation object within the Kernel class.
        """
        # Disable pylint warning for abstract-class-instantiated as we are patching
        # these whilst testing creation of the parent class
        # pylint: disable=abstract-class-instantiated
        # Patch the abstract methods of the Kernel ABC, so it can be created
        p = patch.multiple(coreax.kernel.Kernel, __abstractmethods__=set())
        p.start()

        # Create the kernel and some example data
        kernel = coreax.kernel.Kernel()
        x = jnp.zeros(3)

        # Define a mocked approximator
        approximator = MagicMock(spec=coreax.approximation.RandomApproximator)
        approximator_approximate_method = MagicMock()
        approximator.approximate = approximator_approximate_method
        # pylint: enable=abstract-class-instantiated

        # Call the approximation method and check that approximation object is called as
        # expected
        kernel.approximate_kernel_matrix_row_sum_mean(x=x, approximator=approximator)
        approximator_approximate_method.assert_called_once_with(x)

    def test_update_kernel_matrix_row_sum_zero_max_size(self) -> None:
        """
        Test how the method update_kernel_matrix_row_sum handles a zero max_size.
        """
        # Compute the kernel matrix row sum - a max size of 0 should mean nothing gets
        # updated. This is expected behaviour, as max sizes of zero would get caught and
        # addressed in the wrapper that calls this inside of the kernel class. However,
        # this unusual choice should raise a warning to the user.
        with self.assertWarnsRegex(UserWarning, "max_size is not positive"):
            output = self.default_kernel.update_kernel_matrix_row_sum(
                self.default_x,
                self.default_zero_kernel_row_sum,
                0,
                0,
                self.default_kernel.compute,
                max_size=0,
            )

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, self.default_zero_kernel_row_sum)

    def test_update_kernel_matrix_row_sum_negative_max_size(self) -> None:
        """
        Test how the method update_kernel_matrix_row_sum handles a negative max_size.
        """
        # Define parameters for test
        max_size = 3

        # Define expected output for pairwise distances - in this case we are looking
        # from the start index (0) to start index + max_size in both axis (rows
        # and columns) and then adding the pairwise distances up to this point. Any
        # pairs of points beyond this subset of indices should not be computed. However,
        # note that when we pass max_size to the method, it's negative. Python
        # convention therefore states 'compute from start index up to abs(max_size)
        # elements from the end of the array'.
        expected_output = np.zeros([5, 5])
        for x_1_idx, x_1 in enumerate(self.default_x[0 : (5 - max_size)]):
            for x_2_idx, x_2 in enumerate(self.default_x[0 : (5 - max_size)]):
                expected_output[x_1_idx, x_2_idx] = self.default_kernel.compute(
                    x_1, x_2
                )[0, 0]
        expected_output = expected_output.sum(axis=1)

        # Compute the kernel matrix row sum - a negative max size should fill up to
        # max_size elements from the end of the array. This is expected behaviour, as
        # max sizes of zero would get caught and addressed in the wrapper that calls
        # this inside of the kernel class. However, this unusual choice should raise a
        # warning to the user.
        with self.assertWarnsRegex(UserWarning, "max_size is not positive"):
            output = self.default_kernel.update_kernel_matrix_row_sum(
                self.default_x,
                self.default_zero_kernel_row_sum,
                0,
                0,
                self.default_kernel.compute,
                max_size=-max_size,
            )

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_update_kernel_matrix_row_sum_float_max_size(self) -> None:
        """
        Test how the method update_kernel_matrix_row_sum handles a float max_size.
        """
        # Compute the kernel matrix row sum - a float max size should raise an error
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.update_kernel_matrix_row_sum(
                self.default_x,
                self.default_zero_kernel_row_sum,
                0,
                0,
                self.default_kernel.compute,
                max_size=1.0,
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "max_size must be an integer",
        )

    def test_update_kernel_matrix_row_sum_float_index(self) -> None:
        """
        Test how the method update_kernel_matrix_row_sum handles a float array index.
        """
        # Compute the kernel matrix row sum with the class - a float index should raise
        # an error
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.update_kernel_matrix_row_sum(
                self.default_x,
                self.default_zero_kernel_row_sum,
                0.0,
                0,
                self.default_kernel.compute,
                max_size=2,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "index i must be an integer",
        )

        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.update_kernel_matrix_row_sum(
                self.default_x,
                self.default_zero_kernel_row_sum,
                0,
                0.0,
                self.default_kernel.compute,
                max_size=2,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "index j must be an integer",
        )

    def test_update_kernel_matrix_row_sum_i_not_equal_j(self) -> None:
        """
        Test how the method update_kernel_matrix_row_sum when input indices differ.
        """
        # Define parameters for data
        max_size = 3

        # Define expected output for pairwise distances - in this case we are looking
        # from the start index (0) to start index + max_size (0 + 3) in both axis (rows
        # and columns) and then adding the pairwise distances up to this point. Any
        # pairs of points beyond this subset of indices should not be computed
        kernel_evaluations = np.zeros([5, 5])
        for x_1_idx, x_1 in enumerate(self.default_x[0:max_size]):
            for x_2_idx, x_2 in enumerate(self.default_x[1 : 1 + max_size]):
                kernel_evaluations[x_1_idx, 1 + x_2_idx] = self.default_kernel.compute(
                    x_1, x_2
                )[0, 0]

        # The expected output should just be the sum of the sums over each axis, as
        # we've only filled in the relevant bits in the above loop (note x_2 starts at
        # index 1 not 0, and j is set to 1 in the method call below)
        expected_output = kernel_evaluations.sum(axis=0) + kernel_evaluations.sum(
            axis=1
        )

        # Compute the kernel matrix row sum with the class
        output = self.default_kernel.update_kernel_matrix_row_sum(
            self.default_x,
            self.default_zero_kernel_row_sum,
            0,
            1,
            self.default_kernel.compute,
            max_size=max_size,
        )

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_kernel_matrix_row_sum_zero_max_size(self) -> None:
        """
        Test kernel matrix row sum method when given a zero value of max_size.
        """
        # Compute the kernel matrix row sum with a max size of 0, which would make
        # computations impossible, so we expect an error to be raised
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum(
                x=self.default_x, max_size=0
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "max_size must be a positive integer",
        )

    def test_calculate_kernel_matrix_row_sum_negative_max_size(self) -> None:
        """
        Test kernel matrix row sum method when given a negative value of max_size.
        """
        # Compute the kernel matrix row sum with a negative max size. To avoid nonsense
        # answers, this should raise an error
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum(
                x=self.default_x, max_size=-2
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "max_size must be a positive integer",
        )

    def test_calculate_kernel_matrix_row_sum_float_max_size(self) -> None:
        """
        Test kernel matrix row sum method when given a float value of max_size.
        """
        # Compute the kernel matrix row sum with a negative max size. To avoid nonsense
        # answers, this should raise an error
        with self.assertRaises(TypeError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum(
                x=self.default_x, max_size=2.0
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
        )

    def test_calculate_kernel_matrix_row_sum_mean_zero_max_size(self) -> None:
        """
        Test kernel matrix row sum mean method when given a zero value of max_size.
        """
        # Compute the kernel matrix row sum mean with a max size of 0, which would make
        # computations impossible, so we expect an error to be raised
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum_mean(
                x=self.default_x, max_size=0
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "max_size must be a positive integer",
        )

    def test_calculate_kernel_matrix_row_sum_mean_negative_max_size(self) -> None:
        """
        Test kernel matrix row sum mean method when given a negative value of max_size.
        """
        # Compute the kernel matrix row sum mean with a float max size. To avoid
        # nonsense answers, this should raise an error
        with self.assertRaises(ValueError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum_mean(
                x=self.default_x, max_size=-2
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "max_size must be a positive integer",
        )

    def test_calculate_kernel_matrix_row_sum_mean_float_max_size(self) -> None:
        """
        Test kernel matrix row sum mean method when given a float value of max_size.
        """
        # Compute the kernel matrix row sum mean with a float max size. To avoid
        # nonsense answers, this should raise an error
        with self.assertRaises(TypeError) as error_raised:
            self.default_kernel.calculate_kernel_matrix_row_sum_mean(
                x=self.default_x, max_size=2.0
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
        )


class TestSquaredExponentialKernel(unittest.TestCase):
    """
    Tests related to the SquaredExponentialKernel defined in ``kernel.py``.
    """

    def test_squared_exponential_kernel_unexpected_length_scale(self) -> None:
        r"""
        Test SquaredExponentialKernel computations with unexpected length_scale inputs.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/(2 * \text{length_scale}^2))`.
        Whilst a negative choice of length_scale would be unusual, the kernel can still
        be evaluated using it. Since the length_scale gets squared in the computation,
        the negative values should give the same results as the positive values. Very
        small values of length_scale will result in the exponential of a very large
        number, giving results approximately equal to 0. Very large values of
        length_scale will result in an exponential of values very near zero, yielding
        1.0.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of ``x`` and ``y``, we have:

        .. math::

            ||x - y||^2 &= (0.5 - 2.0)^2
                        &= 2.25

        If we take the length_scale to be 1.0, we get:
            k(x, y) &= \exp(- 2.25 / 2.0)
                    &= 0.324652467
        """
        # Create the kernel with a positive length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of -1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=1.0)
        self.assertEqual(kernel.compute(0.5, 2.0), 0.324652467)

        # Create the kernel with a negative length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=-1.0)
        self.assertEqual(kernel.compute(0.5, 2.0), 0.324652467)

        # Create the kernel with a large negative length_scale, which should just
        # yield an exponential to the power of almost zero and hence a result of 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=-10000.0)
        self.assertEqual(kernel.compute(0.5, 2.0), 1.0)

        # Create the kernel with a small negative length_scale, which should just
        # yield an exponential to the power of a very large negative number and hence
        # a result of 0.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=-0.0000001)
        self.assertEqual(kernel.compute(0.5, 2.0), 0.0)

    def test_squared_exponential_kernel_unexpected_output_scale(self) -> None:
        """
        Test SquaredExponentialKernel computations with unexpected output_scale inputs.

        This example uses the same length_scale demonstrated in
        test_squared_exponential_kernel_unexpected_length_scale. Although a negative
        output_scale would be unusual, there should be no issue evaluating the kernel
        with this.
        """
        kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=1.0, output_scale=1.0
        )
        self.assertEqual(kernel.compute(0.5, 2.0), 0.324652467)

        kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=1.0, output_scale=-1.0
        )
        self.assertEqual(kernel.compute(0.5, 2.0), -0.324652467)

    def test_squared_exponential_kernel_compute_two_floats(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of ``x`` and ``y``, we have:

        .. math::

            ||x - y||^2 &= (0.5 - 2.0)^2
                        &= 2.25

        If we take the length_scale to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x, y) &= \exp(- 2.25 / \pi)
                    &= 0.48860678

        If the length_scale is instead taken to be :math:`\sqrt{\pi}`, we get:
            k(x, y) &= \exp(- 2.25 / (2.0\pi))
                    &= 0.6990041
        """
        # Define data and length_scale
        x = 0.5
        y = 2.0
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have floats as inputs, so treat these single data-points in space
        expected_distance = 0.48860678

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

        # Alter the length_scale, and check the JIT decorator catches the update
        kernel.length_scale = np.sqrt(np.float32(np.pi))

        # Set expected output with this new length_scale
        expected_distance = 0.6990041

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y, with the new, altered length_scale
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_squared_exponential_kernel_compute_two_vectors(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.

        For the two input vectors

        .. math::

            x = [0, 1, 2, 3]

            y = [1, 2, 3, 4]

        For our choices of ``x`` and ``y``, we have:

        .. math::

            ||x - y||^2 &= (0 - 1)^2 + (1 - 2)^2 + (2 - 3)^2 + (3 - 4)^2
                        &= 4

        If we take the ``length_scale`` to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x, y) &= \exp(- 4 / \pi)
                    &= 0.279923327
        """
        # Define data and length_scale
        x = 1.0 * np.arange(4)
        y = x + 1.0
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have 1-dimensional arrays, so treat these as single data-points in space
        expected_distance = 0.279923327

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_squared_exponential_kernel_compute_two_arrays(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations on arrays.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.

        For the two input vectors
        .. math::

            x = [ [0, 1, 2, 3], [5, 6, 7, 8] ]

            y = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        For our choices of ``x`` and ``y``, we have distances of:

        .. math::

            ||x - y||^2 = [4, 0]

        If we take the ``length_scale`` to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x[0], y[0]) &= \exp(- 4 / \pi)
                          &= 0.279923327
            k(x[0], y[1]) &= \exp(- 100 / \pi)
                          &= 1.4996075 \times 10^{-14}
            k(x[1], y[0]) &= \exp(- 64 / \pi)
                          &= 1.4211038 \times 10^{-9}
            k(x[1], y[1]) &= \exp(- 0 / \pi)
                          &= 1.0

        """
        # Define data and length_scale
        x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
        y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected Gram matrix
        expected_distances = np.array(
            [[0.279923327, 1.4996075e-14], [1.4211038e-09, 1.0]]
        )

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the Gram matrix between ``x`` and ``y``
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        np.testing.assert_array_almost_equal(output, expected_distances, decimal=5)

    def test_squared_exponential_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class SquaredExponentialKernel gradient computations.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.
        The gradient of this with respect to ``x`` is:

        .. math:
            - \frac{(x - y)}{\text{length_scale}^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{
                2 \text{length_scale}^2}
            }
        """
        # Define some data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to x
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    -(x[x_idx, :] - y[y_idx, :])
                    / length_scale**2
                    * np.exp(
                        -(np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_squared_exponential_kernel_grad_x_elementwise(self) -> None:
        """
        Test SquaredExponentialKernel element-wise gradient computations w.r.t. ``x``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (
            -(x - y)
            / length_scale**2
            * np.exp(-(np.abs(x - y) ** 2) / (2 * length_scale**2))
        )

        # Compute output using Kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        output = kernel.grad_x_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_scaled_squared_exponential_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class SquaredExponentialKernel gradient computations; with scaling.

        The scaled SquaredExponential kernel is defined as
        :math:`k(x,y) = s\exp (-||x-y||^2/2 * \text{length_scale}^2)`.
        The gradient of this with respect to ``x`` is:

        .. math:
            - s\frac{(x - y)}{\text{length_scale}^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{
                2 \text{length_scale}^2}
            }
        """
        # Define some data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to x
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    -1.0
                    * output_scale
                    * (x[x_idx, :] - y[y_idx, :])
                    / length_scale**2
                    * np.exp(
                        -(np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=length_scale, output_scale=output_scale
        )

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_squared_exponential_kernel_gradients_wrt_y(self) -> None:
        r"""
        Test the class SquaredExponentialKernel gradient computations.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.
        The gradient of this with respect to ``y`` is:

        .. math:
            \frac{(x - y)}{\text{length_scale}^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{
                2 \text{length_scale}^2}
            }
        """
        # Define some data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to y
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    (x[x_idx, :] - y[y_idx, :])
                    / length_scale**2
                    * np.exp(
                        -(np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Evaluate the gradient
        output = kernel.grad_y(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_squared_exponential_kernel_grad_y_elementwise(self) -> None:
        """
        Test SquaredExponentialKernel element-wise gradient computations w.r.t. ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (
            (x - y)
            / length_scale**2
            * np.exp(-(np.abs(x - y) ** 2) / (2 * length_scale**2))
        )

        # Compute output using Kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        output = kernel.grad_y_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_pairwise_kernel_evaluation(self) -> None:
        r"""
        Test the definition of pairwise kernel evaluation functions.

        Pairwise distances mean, given two input arrays, we should return a matrix
        where the values correspond to the distance, as defined by the kernel, between
        each point in the first array and each point in the second array.

        The SquaredExponential kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/2 * \text{length_scale}^2)`.
        If we have two input arrays:

        .. math:
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = [10.0, 3.0, 0.0]

        then we expect an output matrix with 5 rows and 3 columns. Entry [0, 0] in that
        matrix is the kernel distance between points 0.0 and 10.0, entry [0, 1] is the
        kernel distance between 0.0 and 3.0 and so on.
        """
        # Define parameters for data
        length_scale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        y = np.array([10.0, 3.0, 0.0]).reshape(-1, 1)

        # Define the kernel object
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define expected output for pairwise distances
        expected_output = np.zeros([5, 3])
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = kernel.compute(x_, y_)[0, 0]

        # Compute the pairwise distances between the data using the kernel
        output = kernel.compute(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_update_kernel_matrix_row_sum(self) -> None:
        """
        Test updates to the kernel matrix row sum.

        In this test, we consider a subset of points (determined by the parameter
        max_size) and then compute the sum of pairwise distances between them. Any
        points outside this subset should not alter the existing kernel row sum values.
        """
        # Define parameters for data
        length_scale = 1 / np.sqrt(2)
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        max_size = 3

        # Pre-specify an empty kernel matrix row sum to update as we go
        kernel_row_sum = jnp.zeros(len(x))

        # Define the kernel object
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define expected output for pairwise distances - in this case we are looking
        # from the start index (0) to start index + max_size (0 + 3) in both axis (rows
        # and columns) and then adding the pairwise distances up to this point. Any
        # pairs of points beyond this subset of indices should not be computed
        expected_output = np.zeros([5, 5])
        for x_1_idx, x_1 in enumerate(x[0:max_size]):
            for x_2_idx, x_2 in enumerate(x[0:max_size]):
                expected_output[x_1_idx, x_2_idx] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.sum(axis=1)

        # Compute the kernel matrix row sum with the class
        output = kernel.update_kernel_matrix_row_sum(
            x, kernel_row_sum, 0, 0, kernel.compute, max_size=max_size
        )

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_kernel_matrix_row_sum(self) -> None:
        """
        Test computation of the kernel matrix row sum.

        We compute the distance between all pairs of points, and then sum these
        distances, giving a single value for each data-point.
        """
        # Define parameters for data
        length_scale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = x.reshape(-1, 1)

        # Define the kernel object
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define expected output for pairwise distances
        expected_output = np.zeros([5, 5])
        for x_1_idx, x_1 in enumerate(x):
            for x_2_idx, x_2 in enumerate(x):
                expected_output[x_1_idx, x_2_idx] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.sum(axis=1)

        # Compute the kernel matrix row sum with the class
        output = kernel.calculate_kernel_matrix_row_sum(x)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_kernel_matrix_row_sum_mean(self) -> None:
        """
        Test computation of the mean of the kernel matrix row sum.

        We compute the distance between all pairs of points, and then take the mean of
        these distances, giving a single value for each data-point.
        """
        # Define parameters for data
        length_scale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = x.reshape(-1, 1)

        # Define the kernel object
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define expected output for pairwise distances, then take the mean of them
        expected_output = np.zeros([5, 5])
        for x_1_idx, x_1 in enumerate(x):
            for x_2_idx, x_2 in enumerate(x):
                expected_output[x_1_idx, x_2_idx] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.mean(axis=1)

        # Compute the kernel matrix row sum mean with the class
        output = kernel.calculate_kernel_matrix_row_sum_mean(x)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_compute_normalised(self) -> None:
        """
        Test computation of normalised SquaredExponential kernel.

        A normalised SquaredExponential kernel is also known as a Gaussian kernel. We
        generate data and compare to a standard implementation of the Gaussian PDF.
        """
        # Setup some data
        length_scale = np.e
        num_points = 10
        x = np.arange(num_points)
        y = x + 1.0

        # Compute expected output using standard implementation of the Gaussian PDF
        expected_output = np.zeros((num_points, num_points))
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = scipy_norm(y_, length_scale).pdf(x_)

        # Compute the normalised PDF output using the kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=length_scale,
            output_scale=1 / (np.sqrt(2 * np.pi) * length_scale),
        )
        output = kernel.compute(x.reshape(-1, 1), y.reshape(-1, 1))

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_squared_exponential_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                expected_output[x_idx, y_idx] = (
                    2 * np.exp(-dot_product) * (dimension - 2 * dot_product)
                )
        # Compute output using Kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_squared_exponential_div_x_grad_y_elementwise(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of Jacobian w.r.t. ``y`` element-wise.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = 2 * np.exp(-((x - y) ** 2)) * (1 - 2 * (x - y) ** 2)

        # Compute output using Kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y_elementwise(x, y)

        self.assertAlmostEqual(output, expected_output, places=6)

    def test_scaled_squared_exponential_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``; scaled.
        """
        # Setup data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = output_scale * np.exp(
                    -dot_product / (2.0 * length_scale**2)
                )
                expected_output[x_idx, y_idx] = (
                    kernel_scaled
                    / length_scale**2
                    * (dimension - dot_product / length_scale**2)
                )
        # Compute output using Kernel class
        kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=length_scale, output_scale=output_scale
        )
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


class TestLaplacianKernel(unittest.TestCase):
    """
    Tests related to the LaplacianKernel defined in ``kernel.py``.
    """

    def test_laplacian_kernel_unexpected_length_scale(self) -> None:
        r"""
        Test LaplacianKernel computations with unexpected length_scale inputs.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/(2 * \text{length_scale}^2))`.
        Whilst a negative choice of length_scale would be unusual, the kernel can still
        be evaluated using it. Since the length_scale gets squared in the computation,
        the negative values should give the same results as the positive values. Very
        small values of length_scale will result in the exponential of a very large
        number, giving results approximately equal to 0. Very large values of
        length_scale will result in an exponential of values very near zero, yielding
        1.0.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of ``x`` and ``y``, we have:

        .. math::

            ||x - y|| &= |0.5 - 2.0|
                        &= 1.5

        If we take the ``length_scale`` to be 1.0 we get:
            k(x, y) &= \exp(- 1.5 / 2.0)
                    &= 0.472366553

        """
        # Create the kernel with a positive length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of -1.0
        kernel = coreax.kernel.LaplacianKernel(length_scale=1.0)
        self.assertAlmostEqual(kernel.compute(0.5, 2.0), 0.472366553, 6)

        # Create the kernel with a negative length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of 1.0
        kernel = coreax.kernel.LaplacianKernel(length_scale=-1.0)
        self.assertAlmostEqual(kernel.compute(0.5, 2.0), 0.472366553, 6)

        # Create the kernel with a large negative length_scale, which should just
        # yield an exponential to the power of almost zero and hence a result of 1.0
        kernel = coreax.kernel.LaplacianKernel(length_scale=-10000.0)
        self.assertEqual(kernel.compute(0.5, 2.0), 1.0)

        # Create the kernel with a small negative length_scale, which should just
        # yield an exponential to the power of a very large negative number and hence
        # a result of 0.0
        kernel = coreax.kernel.LaplacianKernel(length_scale=-0.0000001)
        self.assertEqual(kernel.compute(0.5, 2.0), 0.0)

    def test_laplacian_kernel_unexpected_output_scale(self) -> None:
        """
        Test LaplacianKernel computations with unexpected output_scale inputs.

        This example uses the same length_scale demonstrated in
        test_laplacian_kernel_unexpected_length_scale. Although a negative output_scale
        would be unusual, there should be no issue evaluating the kernel with this.
        """
        kernel = coreax.kernel.LaplacianKernel(length_scale=1.0, output_scale=1.0)
        self.assertAlmostEqual(kernel.compute(0.5, 2.0), 0.472366553, 6)

        kernel = coreax.kernel.LaplacianKernel(length_scale=1.0, output_scale=-1.0)
        self.assertAlmostEqual(kernel.compute(0.5, 2.0), -0.472366553, 6)

    def test_laplacian_kernel_compute_two_floats(self) -> None:
        r"""
        Test the class LaplacianKernel distance computations.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/2 * \text{length_scale}^2)`.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of ``x`` and ``y``, we have:

        .. math::

            ||x - y|| &= |0.5 - 2.0|
                        &= 1.5

        If we take the ``length_scale`` to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x, y) &= \exp(- 1.5 / \pi)
                    &= 0.62035410351

        If the ``length_scale`` is instead taken to be :math:`\sqrt{\pi}`, we get:
            k(x, y) &= \exp(- 1.5 / (2.0\pi))
                    &= 0.78762561126
        """
        # Define data and length_scale
        x = 0.5
        y = 2.0
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have floats as inputs, so treat these single data-points in space
        expected_distance = 0.62035410351

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

        # Alter the length_scale, and check the JIT decorator catches the update
        kernel.length_scale = np.sqrt(np.float32(np.pi))

        # Set expected output with this new length_scale
        expected_distance = 0.78762561126

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y, with the new, altered length_scale
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_laplacian_kernel_compute_two_vectors(self) -> None:
        r"""
        Test the class LaplacianKernel distance computations.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/2 * \text{length_scale}^2)`.

        For the two input vectors
        .. math::

            x = [0, 1, 2, 3]

            y = [1, 2, 3, 4]

        For our choices of ``x`` and ``y``, we have:

        .. math::

            \lVert x - y \rVert_1 &= |0 - 1| + |1 - 2| + |2 - 3| + |3 - 4|
                        &= 4

        If we take the ``length_scale`` to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x, y) &= \exp(- 4 / \pi)
                    &= 0.279923327
        """
        # Define data and length_scale
        x = 1.0 * np.arange(4)
        y = x + 1.0
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have 1-dimensional arrays, so treat these as single data-points in space
        expected_distance = 0.279923327

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_laplacian_kernel_compute_two_arrays(self) -> None:
        r"""
        Test the class LaplacianKernel distance computations on arrays.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/2 * \text{length_scale}^2)`.

        For the two input vectors

        .. math::

            x = [ [0, 1, 2, 3], [5, 6, 7, 8] ]

            y = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        For our choices of ``x`` and ``y``, we have distances of:

        .. math::

            ||x - y||_1 = [[4, 20], [16, 0]]

        If we take the ``length_scale`` to be :math:`\sqrt{\pi / 2.0}` we get:
            k(x[0], y[0]) &= \exp(- 4 / \pi)
                          &= 0.279923327
            k(x[0], y[1]) &= \exp(- 20 / \pi)
                          &= 0.00171868172
            k(x[1], y[0]) &= \exp(- 16 / \pi)
                          &= 0.00613983027
            k(x[1], y[1]) &= \exp(- 0 / \pi)
                          &= 1.0

        """
        # Define data and length_scale
        x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
        y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected Gram matrix
        expected_distances = np.array(
            [[0.279923327, 0.00171868172], [0.00613983027, 1.0]]
        )

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)

        # Evaluate the kernel - which computes the Gram matrix between ``x`` and ``y``
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        np.testing.assert_array_almost_equal(output, expected_distances, decimal=5)

    def test_laplacian_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class LaplacianKernel gradient computations.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-\Vert x-y \rVert_1/2 * \text{length_scale}^2)`. The
        gradient  of this with respect to ``x`` is:

        .. math:
            - \operatorname{sgn}{(x - y)}{2length\_scale^{2}}e^{-\frac{\lVert x - y
              \rVert_1}{2 \text{length_scale}^2}}
        """
        # Define some data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to x
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    -np.sign(x[x_idx, :] - y[y_idx, :])
                    / (2 * length_scale**2)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_laplacian_kernel_grad_x_elementwise(self) -> None:
        """
        Test LaplacianKernel element-wise gradient computations w.r.t. ``x``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (
            -np.sign(x - y)
            / (2 * length_scale**2)
            * np.exp(-np.abs(x - y) / (2 * length_scale**2))
        )

        # Compute output using Kernel class
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)
        output = kernel.grad_x_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_scaled_laplacian_kernel_gradients_wrt_x(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test the class LaplacianKernel gradient computations; with scaling.

        The scaled Laplacian kernel is defined as
        :math:`k(x,y) = \text{output_scale}\exp (-\lVert x-y \rVert_1/2 * \text{length_scale}^2)`.
        The gradient of this with respect to ``x`` is:

        .. math:

            - \text{output_scale}\operatorname{sgn}{(x - y)}{2length\_scale^{2}}e^{-\frac{\lVert x - y\rVert_1}{2 \text{length_scale}^2}}
        """  # noqa: E501
        # pylint: enable=line-too-long
        # Define some data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to x
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    -1.0
                    * output_scale
                    * np.sign(x[x_idx, :] - y[y_idx, :])
                    / (2 * length_scale**2)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(
            length_scale=length_scale, output_scale=output_scale
        )

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_laplacian_kernel_gradients_wrt_y(self) -> None:
        r"""
        Test the class LaplacianKernel gradient computations.

        The Laplacian kernel is defined as
        :math:`k(x,y) = \exp (-\lVert x-y \rVert_1/2 * \text{length_scale}^2)`. The
        gradient of this with respect to ``y`` is:

        .. math:
            \operatorname{sgn}{(x - y)}{2length\_scale^{2}}e^{-\frac{\lVert x - y
              \rVert_1}{2 \text{length_scale}^2}}
        """
        # Define some data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to y
        true_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                true_gradients[x_idx, y_idx] = (
                    np.sign(x[x_idx, :] - y[y_idx, :])
                    / (2 * length_scale**2)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * length_scale**2)
                    )
                )

        # Create the kernel
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)

        # Evaluate the gradient
        output = kernel.grad_y(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_laplacian_kernel_grad_y_elementwise(self) -> None:
        """
        Test LaplacianKernel element-wise gradient computations w.r.t. ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (
            np.sign(x - y)
            / (2 * length_scale**2)
            * np.exp(-np.abs(x - y) / (2 * length_scale**2))
        )

        # Compute output using Kernel class
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)
        output = kernel.grad_y_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_laplacian_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = (
                    -dimension
                    / (4 * length_scale**4)
                    * np.exp(
                        -jnp.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * length_scale**2)
                    )
                )
        # Compute output using Kernel class
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_laplacian_div_x_grad_y_elementwise(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of Jacobian w.r.t. ``y`` element-wise.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (
            -1 / (4 * length_scale**4) * np.exp(-np.abs(x - y) / (2 * length_scale**2))
        )

        # Compute output using Kernel class
        kernel = coreax.kernel.LaplacianKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y_elementwise(x, y)

        self.assertEqual(output, expected_output)

    def test_scaled_laplacian_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``; scaled.
        """
        # Setup data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = (
                    -1.0
                    * output_scale
                    * dimension
                    / (4 * length_scale**4)
                    * np.exp(
                        -jnp.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * length_scale**2)
                    )
                )
        # Compute output using Kernel class
        kernel = coreax.kernel.LaplacianKernel(
            length_scale=length_scale, output_scale=output_scale
        )
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


class TestPCIMQKernel(unittest.TestCase):
    """
    Tests related to the PCIMQKernel defined in ``kernel.py``.
    """

    def test_pcimq_kernel_unexpected_length_scale(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test PCIMQKernel computations with unexpected length_scale inputs.

        The PCIMQ kernel is defined as
        :math:`k(x,y) = \frac{1.0}{1.0 / \sqrt(1.0 + ((x - y) / \text{length_scale}) ** 2 / 2.0)}`.
        Whilst a negative choice of length_scale would be unusual, the kernel can still
        be evaluated using it. Since the length_scale gets squared in the computation,
        the negative values should give the same results as the positive values. Very
        small values of length_scale will result in the exponential of a very large
        number, giving results approximately equal to 0. Very large values of
        length_scale will result in an exponential of values very near zero, yielding
        1.0.
        """  # noqa: E501
        # pylint: enable=line-too-long

        # Define input data
        length_scale = np.e
        num_points = 10
        x = np.arange(num_points).reshape(-1, 1)
        y = x + 1.0

        # Compute expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = 1.0 / np.sqrt(
                    1.0 + ((x_[0] - y_[0]) / length_scale) ** 2 / 2.0
                )

        # Create the kernel with a positive length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of -1.0
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        # Check output matches expected
        np.testing.assert_array_almost_equal(
            kernel.compute(x, y), expected_output, decimal=3
        )

        # Create the kernel with a negative length_scale - this should give the same
        # answer when evaluating the kernel with a length_scale of 1.0
        kernel = coreax.kernel.PCIMQKernel(length_scale=-1.0 * length_scale)
        np.testing.assert_array_almost_equal(
            kernel.compute(x, y), expected_output, decimal=3
        )

        # Create the kernel with a large negative length_scale, which should just
        # yield a result of almost 1
        kernel = coreax.kernel.PCIMQKernel(length_scale=-10000.0)
        self.assertEqual(kernel.compute(0.5, 2.0), 1.0)

        # Create the kernel with a small negative length_scale, which should just
        # yield a result of almost 0
        kernel = coreax.kernel.PCIMQKernel(length_scale=-0.00000000001)
        self.assertAlmostEqual(kernel.compute(0.5, 2.0), 0.0)

    def test_pcimq_kernel_unexpected_output_scale(self) -> None:
        """
        Test PCIMQKernel computations with unexpected output_scale inputs.

        This example uses the same length_scale demonstrated in
        test_pcimq_kernel_unexpected_length_scale. Although a negative output_scale
        would be unusual, there should be no issue evaluating the kernel with this.
        """
        # Define input data
        length_scale = np.e
        num_points = 10
        x = np.arange(num_points).reshape(-1, 1)
        y = x + 1.0

        # Compute expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = 1.0 / np.sqrt(
                    1.0 + ((x_[0] - y_[0]) / length_scale) ** 2 / 2.0
                )

        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale, output_scale=1.0)
        np.testing.assert_array_almost_equal(
            kernel.compute(x, y), expected_output, decimal=3
        )

        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale, output_scale=-1.0)
        np.testing.assert_array_almost_equal(
            kernel.compute(x, y), -1.0 * expected_output, decimal=3
        )

    def test_pcimq_kernel_compute(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test the class PCIMQKernel distance computations.

        The PCIMQ kernel is defined as
        :math:`k(x,y) = \frac{1.0}{1.0 / \sqrt(1.0 + ((x - y) / \text{length_scale}) ** 2 / 2.0)}`.
        """  # noqa: E501
        # pylint: enable=line-too-long
        # Define input data
        length_scale = np.e
        num_points = 10
        x = np.arange(num_points).reshape(-1, 1)
        y = x + 1.0

        # Compute expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = 1.0 / np.sqrt(
                    1.0 + ((x_[0] - y_[0]) / length_scale) ** 2 / 2.0
                )

        # Compute distance using the kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.compute(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class PCIMQ gradient computations with respect to ``x``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = -(x[x_idx, :] - y[y_idx, :]) / (
                    1 + np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2
                ) ** (3 / 2)

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.grad_x(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_kernel_grad_x_elementwise(self) -> None:
        """
        Test the PCIMQ kernel element-wise gradient computations w.r.t. ``x``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = -(x - y) / (1 + np.abs(x - y) ** 2) ** (3 / 2)

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.grad_x_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_pcimq_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class PCIMQ gradient computations with respect to ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = (x[x_idx, :] - y[y_idx, :]) / (
                    1 + np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2
                ) ** (3 / 2)

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_kernel_grad_y_elementwise(self) -> None:
        """
        Test the class PCIMQ element-wise gradient computations with respect to ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = (x - y) / (1 + np.abs(x - y) ** 2) ** (3 / 2)

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.grad_y_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_scaled_pcimq_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class PCIMQ gradient computations with respect to ``y``; with scaling.
        """
        # Setup data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = (
                    output_scale
                    / (2 * length_scale**2)
                    * (x[x_idx, :] - y[y_idx, :])
                    / (
                        1
                        + (np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * length_scale**2)
                    )
                    ** (3 / 2)
                )

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(
            length_scale=length_scale, output_scale=output_scale
        )
        output = kernel.grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                denominator = (1 + dot_product) ** (3 / 2)
                expected_output[x_idx, y_idx] = (
                    dimension / denominator - 3 * dot_product / denominator ** (5 / 3)
                )

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_div_x_grad_y_elementwise(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of Jacobian w.r.t. ``y`` element-wise.
        """
        # Setup data
        length_scale = 1 / np.sqrt(2)
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        denominator = (1 + (x - y) ** 2) ** (3 / 2)
        expected_output = 1 / denominator - 3 * (x - y) ** 2 / denominator ** (5 / 3)

        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(length_scale=length_scale)
        output = kernel.divergence_x_grad_y_elementwise(x, y)

        self.assertAlmostEqual(output, expected_output, places=6)

    def test_scaled_pcimq_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``; scaled.
        """
        # Setup data
        length_scale = 1 / np.pi
        output_scale = np.e
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = output_scale / (
                    (1 + dot_product / (2 * length_scale**2)) ** (1 / 2)
                )
                expected_output[x_idx, y_idx] = (
                    output_scale
                    / (2 * length_scale**2)
                    * (
                        dimension * (kernel_scaled / output_scale) ** 3
                        - 3
                        * dot_product
                        * (kernel_scaled / output_scale) ** 5
                        / (2 * length_scale**2)
                    )
                )
        # Compute output using Kernel class
        kernel = coreax.kernel.PCIMQKernel(
            length_scale=length_scale, output_scale=output_scale
        )
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


class TestLinearKernel(unittest.TestCase):
    """
    Tests related to the LinearKernel defined in ``kernel.py``.
    """

    def test_linear_kernel_compute(self) -> None:
        r"""
        Test the class LinearKernel distance computations.

        The Linear kernel is defined as
        :math:`k(x,y) = x^Ty.
        """
        # Define input data
        num_points = 10
        x = np.arange(num_points).reshape(-1, 1)
        y = x + 1.0

        # Compute expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx, x_ in enumerate(x):
            for y_idx, y_ in enumerate(y):
                expected_output[x_idx, y_idx] = x_.item() * y_.item()

        # Compute distance using the kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.compute(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_linear_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class Linear gradient computations with respect to ``x``.
        """
        # Setup data
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = y[y_idx]

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.grad_x(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_linear_kernel_grad_x_elementwise(self) -> None:
        """
        Test the Linear kernel element-wise gradient computations w.r.t. ``x``.
        """
        # Setup data
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = y

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.grad_x_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_linear_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class Linear gradient computations with respect to ``y``.
        """
        # Setup data
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = x[x_idx]

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_linear_kernel_grad_y_elementwise(self) -> None:
        """
        Test the Linear kernel element-wise gradient computations w.r.t. ``y``.
        """
        # Setup data
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 1))
        y = generator.random((1, 1))

        # Define expected output
        expected_output = x

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.grad_y_elementwise(x, y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_linear_div_x_grad_y(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of kernel Jacobian w.r.t. ``y``.
        """
        # Setup data
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_output[x_idx, y_idx] = dimension

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_linear_div_x_grad_y_elementwise(self) -> None:
        """
        Test the divergence w.r.t. ``x`` of Jacobian w.r.t. ``y`` element-wise.
        """
        # Setup data
        random_data_generation_key = 1_989

        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((1, 2))
        y = generator.random((1, 2))

        # Define expected output
        expected_output = 2

        # Compute output using Kernel class
        kernel = coreax.kernel.LinearKernel()
        output = kernel.divergence_x_grad_y_elementwise(x, y)

        self.assertAlmostEqual(output, expected_output, places=6)


class TestSteinKernel(unittest.TestCase):
    """
    Tests related to the SteinKernel defined in ``kernel.py``.
    """

    def test_stein_kernel_computation(self) -> None:
        r"""
        Test computation of the SteinKernel.

        Due to the complexity of the Stein kernel, we check the size of the output
        matches the expected size, not the numerical values in the output array itself.
        """
        # Setup some data
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        length_scale = 1 / np.sqrt(2)

        def score_function(x_: ArrayLike) -> Array:
            """
            Compute a simple, example score function for testing purposes.

            :param x_: Point or points at which we wish to evaluate the score function
            :return: Evaluation of the score function at ``x_``
            """
            return -x_

        # Setup data
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)

        x = generator.random((num_points_x, dimension))
        y = generator.random((num_points_y, dimension))

        # Set expected output sizes
        expected_size = (10, 5)

        # Compute output using Kernel class
        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(length_scale=length_scale),
            score_function=score_function,
        )
        output = kernel.compute(x, y)

        # Check output sizes match the expected
        self.assertEqual(output.shape, expected_size)

    def test_stein_kernel_element_computation(self) -> None:
        r"""
        Test computation of a single element of the SteinKernel.
        """
        # Setup some data
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        length_scale = 1 / np.sqrt(2)
        beta = 0.5

        def score_function(x_: ArrayLike) -> Array:
            """
            Compute a simple, example score function for testing purposes.

            :param x_: Point or points at which we wish to evaluate the score function
            :return: Evaluation of the score function at ``x_``
            """
            return -x_

        def k_x_y(x_input, y_input):
            r"""
            Compute Stein kernel.

            Throughout this docstring, x_input and y_input are simply referred to as x
            and y.

            The base kernel is :math:`(1 + \lvert \mathbf{x} - \mathbf{y}
            \rvert^2)^{-1/2}`. :math:`\mathbb{P}` is :math:`\mathcal{N}(0, \mathbf{I})`
            with :math:`\nabla \log f_X(\mathbf{x}) = -\mathbf{x}`.

            In the code: n, m and r refer to shared denominators in the Stein kernel
            equation (rather than divergence, x_, y_ and z in the main code function).

            :math:`k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = n + m + r`.

            :math:`n := -\frac{3 \lvert \mathbf{x} - \mathbf{y} \rvert^2}{(1 + \lvert
            \mathbf{x} - \mathbf{y} \rvert^2)^{5/2}}`.

            :math:`m := 2\beta\left[ \frac{d + [\mathbf{y} -
            \mathbf{x}]^\intercal[\mathbf{x} - \mathbf{y}]}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{3/2}} \right]`.

            :math:`r := \frac{\mathbf{x}^\intercal \mathbf{y}}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{1/2}}`.

            :param x_input: a d-dimensional vector
            :param y_input: a d-dimensional vector
            :return: kernel evaluated at x, y
            """
            norm_sq = np.linalg.norm(x_input - y_input) ** 2
            n = -3 * norm_sq / (1 + norm_sq) ** 2.5
            m = (
                2
                * beta
                * (
                    dimension
                    + np.dot(
                        score_function(x_input) - score_function(y_input),
                        x_input - y_input,
                    )
                )
                / (1 + norm_sq) ** 1.5
            )
            r = (
                np.dot(score_function(x_input), score_function(y_input))
                / (1 + norm_sq) ** 0.5
            )
            return n + m + r

        # Setup data
        generator = np.random.default_rng(1_989)
        x = generator.random((num_points_x, dimension))
        y = generator.random((num_points_y, dimension))

        # Compute output using Kernel class
        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(length_scale=length_scale),
            score_function=score_function,
        )

        # Compute the output step-by-step with the element method
        expected_output = np.zeros([x.shape[0], y.shape[0]])
        output = kernel.compute(x, y)
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                # Compute via our hand-coded kernel evaluation
                expected_output[x_idx, y_idx] = k_x_y(x[x_idx, :], y[y_idx, :])

        # Check output matches the expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_stein_kernel_invalid_base_kernel_missing_divergence(self) -> None:
        r"""
        Test how the SteinKernel handles an invalid base kernel being passed.

        The base kernel here is missing a method to compute the divergence of the data.
        """
        # Setup some data
        num_points_x = 10
        num_points_y = 5
        dimension = 2

        def score_function(x_: ArrayLike) -> Array:
            """
            Compute a simple, example score function for testing purposes.

            :param x_: Point or points at which we wish to evaluate the score function
            :return: Evaluation of the score function at ``x_``
            """
            return -x_

        # Setup data
        generator = np.random.default_rng(1_989)
        x = generator.random((num_points_x, dimension))
        y = generator.random((num_points_y, dimension))

        # Compute output using Kernel class - since the base kernel does not have a
        # divergence method, it should raise an attribute error.
        kernel = coreax.kernel.SteinKernel(
            base_kernel=KernelNoDivergenceMethod(a=1.0),
            score_function=score_function,
        )
        with self.assertRaisesRegex(
            AttributeError, "object has no attribute 'divergence_x_grad_y_elementwise'"
        ):
            kernel.compute(x, y)

    def test_stein_kernel_invalid_base_kernel_no_pytree(self) -> None:
        r"""
        Test how the SteinKernel handles an invalid base kernel being passed.

        The base kernel here is missing a method to flatten a pytree, and then a method
        to unflatten a pytree.
        """

        def score_function(x_: ArrayLike) -> Array:
            """
            Compute a simple, example score function for testing purposes.

            :param x_: Point or points at which we wish to evaluate the score function
            :return: Evaluation of the score function at ``x_``
            """
            return -x_

        # Create the Kernel class - since the base kernel does not have tree_flatten
        # method, the code should raise an attribute error and inform the user.
        with self.assertRaises(AttributeError) as error_raised:
            coreax.kernel.SteinKernel(
                base_kernel=KernelNoTreeFlatten(a=1.0),
                score_function=score_function,
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "base_kernel must have the method tree_flatten implemented",
        )

        # Create the Kernel class - since the base kernel does not have tree_unflatten
        # method, the code should raise an attribute error and inform the user.
        with self.assertRaises(AttributeError) as error_raised:
            coreax.kernel.SteinKernel(
                base_kernel=KernelNoTreeUnflatten(a=1.0),
                score_function=score_function,
            )

        self.assertEqual(
            error_raised.exception.args[0],
            "base_kernel must have the method tree_unflatten implemented",
        )


if __name__ == "__main__":
    unittest.main()
