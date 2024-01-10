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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import coreax.util
from examples.david_map_reduce_weighted import main as david_map_reduce_weighted_main


class TestDavid(unittest.TestCase):
    """
    Test end-to-end code run with an image example.
    """

    def test_david_map_reduce_weighted(self) -> None:
        """
        Test david_map_reduce_weighted.py example.

        An end-to-end test to check david_map_reduce_weighted.py runs without error,
        generates output, and has coreset MMD better than MMD from random sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print") as mock_print,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run david_map_reduce_weighted.py
            in_path = Path(os.path.dirname(__file__)) / Path(
                "../../examples/data/david_orig.png"
            )
            out_path = Path(tmp_dir) / "david_coreset.png"
            mmd_coreset, mmd_random = david_map_reduce_weighted_main(
                in_path=in_path, out_path=out_path
            )

            self.assertEqual(
                call("Image dimensions: (215, 180)"),
                mock_print.call_args_list[0],
                msg="Unexpected print statement. Likely due to unexpected image size",
            )

            mock_show.assert_called_once()

            coreax.util.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )
