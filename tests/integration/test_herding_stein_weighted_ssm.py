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
Integration test for weighted herding with a Stein kernel example.

This test determines the score function for the Stein kernel using sliced score
matching.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from examples.herding_stein_weighted_ssm import main as herding_stein_weighted_ssm_main

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


class TestHerdingSteinWeightedSSM(unittest.TestCase):
    """
    Test end-to-end code run with a tabular data example.

    A Stein kernel, with score function determined via sliced score matching is used,
    and the result is a weighted coreset.
    """

    def test_herding_stein_weighted(self) -> None:
        """
        Test herding_stein_weighted_ssm.py example.

        An end-to-end test to check herding_stein_weighted_ssm.py runs without error,
        generates output, and has coreset MMD better than MMD from random sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_stein_weighted.png"
            mmd_coreset, mmd_random = herding_stein_weighted_ssm_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assertTrue(Path(out_path).resolve().is_file())

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )


# pylint: enable=duplicate-code


if __name__ == "__main__":
    unittest.main()
