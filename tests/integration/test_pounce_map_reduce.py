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
from unittest.mock import patch

import coreax.util
from examples.pounce_map_reduce import main as pounce_map_reduce_main


class TestPounceMapReduce(unittest.TestCase):
    """
    Test end-to-end code run with a video example.

    Memory requirements are reduced with a map reduce algorithm.
    """

    def test_pounce_map_reduce(self) -> None:
        """
        Test pounce_map_reduce.py example.

        An end-to-end test to check pounce_map_reduce.py runs without error, generates
        output, and has coreset MMD better than MMD from random sampling.
        """
        in_path = Path(os.path.dirname(__file__)) / Path(
            "../../examples/data/pounce/pounce.gif"
        )
        with patch("builtins.print"), tempfile.TemporaryDirectory() as tmp_dir:
            # Run pounce_map_reduce.py
            mmd_coreset, mmd_random = pounce_map_reduce_main(
                in_path=in_path, out_path=Path(tmp_dir)
            )

            coreax.util.assert_is_file(tmp_dir / Path("pounce_map_reduce_coreset.gif"))
            coreax.util.assert_is_file(tmp_dir / Path("pounce_map_reduce_frames.png"))

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )
