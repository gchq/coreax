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

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

import coreax.data as cd
import coreax.metrics as cm
import coreax.reduction as cr
import coreax.weights as cw


class MockCreatedInstance:
    @staticmethod
    def solve(*args):
        return tuple(*args)


class TestCoreset(unittest.TestCase):
    """
    Tests related to :class:`Coreset`.
    """

    def setUp(self) -> None:
        self.original_data = MagicMock()
        self.original_data.pre_coreset_array = jnp.array([0, 0.3, 0.75, 1])
        self.weights = "MMD"
        self.kernel = MagicMock()
        self.test_class = cr.Coreset(self.original_data, self.weights, self.kernel)
        self.coreset = self.original_data[jnp.array([0, 1, 3])]
        self.test_class.coreset = self.coreset

    def test_solve_weights(self) -> None:
        """
        Test calls within ``solve_weights``.
        """
        with patch("coreax.util.create_instance_from_factory") as mock_create_instance:
            mock_create_instance.return_value = MockCreatedInstance()
            actual_out = self.test_class.solve_weights()
            mock_create_instance.assert_called_once_with(
                cw.weights_factory, self.weights, self.kernel
            )
            # MockCreatedInstance returns tuple of input arguments
            self.assertEqual(
                actual_out, (self.original_data.pre_coreset_array, self.coreset)
            )

    def test_compute_metric(self) -> None:
        """
        Test the compute_metric method of :class:`coreax.reduction.DataReduction`.
        """
        with patch("coreax.util.create_instance_from_factory") as mock_create_instance:
            mock_create_instance.return_value = MockCreatedInstance()
            actual_out = self.test_class.compute_metric("metric_a")
            mock_create_instance.assert_called_once_with(
                cm.metric_factory, "metric_a", kernel=self.kernel
            )
            # MockCreatedInstance returns tuple of input arguments
            self.assertEqual(
                actual_out, (self.original_data.pre_coreset_array, self.coreset)
            )


class TestSizeReduce(unittest.TestCase):
    """Test :class:`SizeReduce`."""

    def test_random_sample(self):
        """Test reduction with :class:`RandomSample`."""
        orig_data = cd.ArrayData.load([[i, 2 * i] for i in range(20)])
        strategy = cr.SizeReduce("random", num_points=10)
        coreset = strategy.reduce(orig_data)
        # Check shape of output
        self.assertEqual(coreset.coreset.format().shape, [10, 2])


if __name__ == "__main__":
    unittest.main()
