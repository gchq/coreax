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
Integration test for analytic example of greedy kernel points.
"""

import numpy as np

from examples.greedy_kernel_points_analytic import (
    main as greedy_kernel_points_analytic_main,
)

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


def test_greedy_kernel_points_analytic() -> None:
    """Test end-to-end code run on greedy kernel points analytic example."""
    expect_indices = np.array([2, 1])
    expect_data = np.array([[2, 1], [0, 1]])
    expect_supervision = np.array([5, 1])

    actual = greedy_kernel_points_analytic_main()

    np.testing.assert_array_equal(np.squeeze(actual.indices.data), expect_indices)
    np.testing.assert_array_equal(actual.points.data, expect_data)
    np.testing.assert_array_equal(
        np.squeeze(actual.points.supervision), expect_supervision
    )


# pylint: enable=duplicate-code
