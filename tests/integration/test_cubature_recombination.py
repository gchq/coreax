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
Integration test for generating non-product cubature using recombination.
"""

import unittest
from unittest.mock import patch

from examples.cubature_recombination import main as cubature_recombination_main


class TestCubatureRecombination(unittest.TestCase):
    """
    Test end-to-end code run with a cubature recombination example.
    """

    def test_cubature_recombination(self) -> None:
        """
        Test cubature_recombination.py example.

        An end-to-end test to check cubature_recombination.py runs without error.
        """
        with patch("builtins.print"):
            # Run the example; required assertions are made directly in the example.
            _ = cubature_recombination_main(dimension=3, max_degree=4)
