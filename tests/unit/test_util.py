# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest

import jax.numpy as jnp
import numpy as np
from scipy.stats import ortho_group

from coreax.util import (
    ClassFactory,
    call_with_excess_kwargs,
    pdiff,
    sq_dist,
    sq_dist_pairwise,
)


class Test(unittest.TestCase):
    def test_sq_dist(self) -> None:
        """
        Test square distance under float32.
        """
        x, y = ortho_group.rvs(dim=2)
        d = jnp.linalg.norm(x - y) ** 2
        td = sq_dist(x, y)
        self.assertAlmostEqual(td, d, places=3)
        td = sq_dist(x, x)
        self.assertAlmostEqual(td, 0.0, places=3)
        td = sq_dist(y, y)
        self.assertAlmostEqual(td, 0.0, places=3)

    def test_sq_dist_pairwise(self) -> None:
        """
        Test vmap version of sq distance.
        """
        # create an orthonormal matrix
        d = 3
        m = ortho_group.rvs(dim=d)
        tinner = sq_dist_pairwise(m, m)
        # Use original numpy because Jax arrays are immutable
        ans = np.ones((d, d)) * 2.0
        np.fill_diagonal(ans, 0.0)
        # Frobenius norm
        td = jnp.linalg.norm(tinner - ans)
        self.assertEqual(td.ndim, 0)
        self.assertAlmostEqual(float(td), 0.0, places=3)

    def test_pdiff(self) -> None:
        """
        Test the function pdiff.

        This test ensures efficient computation of pairwise differences.
        """
        m = 10
        n = 10
        d = 3
        x_array = np.random.random((n, d))
        y_array = np.random.random((m, d))
        z_array = np.array([[x - y for y in y_array] for x in x_array])
        tst = pdiff(x_array, y_array)
        self.assertAlmostEqual(float(jnp.linalg.norm(tst - z_array)), 0.0, places=3)


class TestCallWithExcessKwargs(unittest.TestCase):
    # Define a test function
    @staticmethod
    def trial_function(a: int = 1, b: int = 2, c: int = 3):
        return a, b, c

    class TrialClass:
        def __init__(self, a: int = 1, b: int = 2, c: int = 3):
            self.a = a
            self.b = b
            self.c = c

    def test_function_empty(self):
        """Test that no arguments are passed to trial function."""
        self.assertEqual(call_with_excess_kwargs(self.trial_function), (1, 2, 3))

    def test_function_mixed(self):
        """
        Test that positional and keyword arguments are passed with excess ignored.

        For trial function.
        """
        self.assertEqual(
            call_with_excess_kwargs(self.trial_function, 4, c=5, d=6), (4, 2, 5)
        )

    def test_class_mixed(self):
        """
        Test that positional and keyword arguments are passed with excess ignored.

        For trial class.
        """
        actual = call_with_excess_kwargs(self.TrialClass, 4, c=5, d=6)
        self.assertEqual(actual.a, 4)
        self.assertEqual(actual.b, 2)
        self.assertEqual(actual.c, 5)

    def test_too_many_positional(self):
        """
        Test that all positional arguments are passed to function if too many are given.

        A :exc:`TypeError` will be raised if too many positional arguments.
        """
        self.assertRaises(TypeError, self.trial_function, 4, 5, 6, 7)


class TestClassFactory(unittest.TestCase):
    """Test ClassFactory."""

    # Define some output classes for the factory
    class BaseClass:
        pass

    class AClass(BaseClass):
        pass

    class BClass(BaseClass):
        pass

    class CClass:
        """Class of another type."""

        pass

    def test_factory(self):
        """
        Test factory by trying all operations.
        """
        factory = ClassFactory(self.BaseClass)

        # Register some objects
        factory.register("a", self.AClass)
        # Test name clash
        self.assertRaises(ValueError, factory.register, "a", self.BClass)
        factory.register("b", self.BClass)
        # Check for instantiated class
        self.assertRaises(TypeError, factory.register, "c", self.BClass())
        # Check for wrong base class
        self.assertRaises(TypeError, factory.register, "d", self.CClass)
        # Check something strange
        self.assertRaises(TypeError, factory.register, "e", 2)

        # Test retrieval
        # Dependency injection
        self.assertEqual(factory.get(self.AClass), self.AClass)
        # Wrong type
        self.assertRaises(TypeError, factory.get, self.CClass)
        # Already instantiated
        self.assertRaises(TypeError, factory.get, self.AClass())
        # By name
        self.assertEqual(factory.get("a"), self.AClass)
        self.assertEqual(factory.get("b"), self.BClass)
        # Invalid name
        self.assertRaises(KeyError, factory.get, "f")
        # Check something strange
        self.assertRaises(TypeError, factory.get, 2)


if __name__ == "__main__":
    unittest.main()
