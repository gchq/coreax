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
Integration test for pounce (video) example.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.pounce import main as pounce_main

# pylint: disable=duplicate-code


class TestPounce(unittest.TestCase):
    """
    Test end-to-end code run with a video example.
    """

    def test_pounce(self) -> None:
        """
        Test pounce.py example.

        An end-to-end test to check pounce.py runs without error, generates  output, and
        has coreset MMD better than MMD from random sampling.
        """
        in_path = Path(os.path.dirname(__file__)) / Path(
            "../../examples/data/pounce/pounce.gif"
        )
        with patch("builtins.print"), tempfile.TemporaryDirectory() as tmp_dir:
            # Run pounce.py
            mmd_coreset, mmd_random = pounce_main(
                in_path=in_path, out_path=Path(tmp_dir)
            )

            self.assertTrue(
                Path(tmp_dir / Path("pounce_coreset.gif")).resolve().is_file()
            )
            self.assertTrue(
                Path(tmp_dir / Path("pounce_frames.png")).resolve().is_file()
            )

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )


# pylint: enable=duplicate-code


if __name__ == "__main__":
    unittest.main()
