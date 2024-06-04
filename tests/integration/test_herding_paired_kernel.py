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
Integration test for basic herding example.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from examples.herding_paired_kernel import main as herding_paired_kernel_main

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


class TestHerdingPairedKernel(unittest.TestCase):
    """
    Test end-to-end code run using a PairedKernel.
    """

    def test_herding_basic(self) -> None:
        """
        Test herding_paired_kernel.py example.

        An end-to-end test to check herding_paired_kernel.py runs without error.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_paired_kernel.png"
            herding_paired_kernel_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assertTrue(Path(out_path).resolve().is_file())


# pylint: enable=duplicate-code
