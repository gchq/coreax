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
from unittest.mock import patch, Mock

import jax.numpy as jnp
from jax import Array

import coreax.coreset as cc
import coreax.reduction as cr
import coreax.util as cu
import coreax.weights as cw


class TestDataReduction(unittest.TestCase):
    """
    Tests related to kernel.py functions.
    """

    def setUp(self) -> None:
        self.original_data = jnp.array([0, 0.3, 0.75, 1])
        self.weight = 'MMD'
        self.kernel = jnp.array([0, 0.5, 0.6, 0.5, 0])
        self.test_class = cr.DataReduction(self.original_data, self.weight, self.kernel)
        # make reduced data distinguishable from original data without calling an actual reduction
        self.reduced_data = self.original_data[jnp.array([0, 1, 3])]
        self.test_class.reduced_data = reduced_data

    def test_solve_weights(self) -> None:
        """
        Test the solve_weights method of :class:`coreax.reduction.DataReduction`.
        """


        with patch('coreax.reduction.DataReduction._create_instance_from_factory') \
            as mock_create_instance:

            mock_create_instance.return_value = MockCreatedInstance()

            actual_out = self.test_class.solve_weights()

            mock_create_instance.assert_called_once_with(cw.weights_factory, self.weight)

            self.assertEqual(actual_out, (self.original_data, self.reduced_data, self.kernel))

    def test_fit(self) -> None:
        """
        Test the fit method of :class:`coreax.reduction.DataReduction`.
        """

        with patch('coreax.reduction.DataReduction._create_instance_from_factory') \
                as mock_create_instance:
            mock_create_instance.return_value = MockCreatedInstance()

            actual_out = self.test_class.fit('coreset_a')

            mock_create_instance.assert_called_once_with(cc.coreset_factory, 'coreset_a')

            self.assertEqual(actual_out, (self.original_data, self.kernel))

    def test_refine(self) -> None:
        """
        Test the refine method of :class:`coreax.reduction.DataReduction`.
        """

        with patch('coreax.reduction.DataReduction._create_instance_from_factory') \
                as mock_create_instance:
            mock_create_instance.return_value = MockCreatedInstance()

            actual_out = self.test_class.refine('refine_a')

            mock_create_instance.assert_called_once_with(cc.refine_factory, 'refine_a', kernel=self.kernel)

            self.assertEqual(actual_out, (self.original_data, self.kernel, kernel_mean))

    def test_compute_metric(self) -> None:
        """
        Test the compute_metric method of :class:`coreax.reduction.DataReduction`.
        """

        with patch('coreax.reduction.DataReduction._create_instance_from_factory') \
                as mock_create_instance:
            mock_create_instance.return_value = MockCreatedInstance()

            actual_out = self.test_class.compute_metric('metric_a')

            mock_create_instance.assert_called_once_with(cc.refine_factory, 'metric_a', kernel=self.kernel, weight=self.weight)

            self.assertEqual(actual_out, (self.original_data, self.reduced_data))

    def test_create_instance_from_factory(self) -> None:
        """
        Test the _create_instance_from_factory method of :class:`coreax.reduction.DataReduction`.
        """

        with patch('coreax.util.call_with_excess_kwargs') as mock_call:

            actual_out = cu.create_instance_from_factory(MockClassFactory, 'class_a', a='a', b='b')

            mock_call.assert_called_once_with('class_a', a='a', b='b')


# Mocked output of reduction.DataReduction._create_instance_from_factory
class MockCreatedInstance:
    @staticmethod
    def solve(*args):
        return tuple(*args)

# Mocked ClassFactory
class MockClassFactory:
    @staticmethod
    def get(*args):
        return tuple(*args)
