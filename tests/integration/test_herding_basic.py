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

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import coreax.util
from examples.herding_basic import main as herding_basic_main


class TestHerdingBasic(unittest.TestCase):
    """
    Test end-to-end code run with a basic tabular data example.
    """

    def test_herding_basic(self) -> None:
        """
        Test herding_basic.py example.

        An end-to-end test to check herding_basic.py runs without error, generates
        output, and has coreset MMD better than MMD from random sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_basic.png"
            mmd_coreset, mmd_random = herding_basic_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            coreax.util.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )
