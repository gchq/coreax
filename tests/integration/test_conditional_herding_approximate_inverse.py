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
Integration test for ConditionalKernelHerding example with approximate inversion.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from examples.conditional_herding_approximate_inverse import (
    main as conditional_herding_approximate_inverse_main,
)

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


class TestConditionalKernelHerdingApproximateInverse(unittest.TestCase):
    """
    Test end-to-end code run with a tabular data example with approximate inverses.
    """

    def test_conditional_herding_approximate_inverse(self) -> None:
        """
        Test conditional_herding_approximate_inverse.py example.

        An end-to-end test to check conditional_herding_approximate_inverse.py runs
        without error.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run approximate ConditionalKernelHerding example
            out_path = Path(tmp_dir) / "conditional_herding_approximate_inverse.png"
            conditional_herding_approximate_inverse_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assertTrue(Path(out_path).resolve().is_file())


# pylint: enable=duplicate-code
