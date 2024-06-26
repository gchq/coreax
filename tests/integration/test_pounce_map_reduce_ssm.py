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
Integration test for pounce (video) example using map reduce.

This test determines the score function for the Stein kernel using sliced score
matching.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.pounce_map_reduce_ssm import main as pounce_map_reduce_ssm_main

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


class TestPounceMapReduce(unittest.TestCase):
    """
    Test end-to-end code run with a video example.

    Memory requirements are reduced with a map reduce algorithm. A Stein kernel is used
    with a score function approximated via a neural network.
    """

    def test_pounce_map_reduce_ssm(self) -> None:
        """
        Test pounce_map_reduce_ssm.py example.

        An end-to-end test to check pounce_map_reduce_ssm.py runs without error.
        """
        in_path = Path(os.path.dirname(__file__)) / Path(
            "../../examples/data/pounce/pounce.gif"
        )
        with patch("builtins.print"), tempfile.TemporaryDirectory() as tmp_dir:
            # Run pounce_map_reduce_ssm.py
            pounce_map_reduce_ssm_main(in_path=in_path, out_path=Path(tmp_dir))

            self.assertTrue(
                Path(
                    tmp_dir
                    / Path("pounce_map_reduce_sliced_score_matching_coreset.gif")
                )
                .resolve()
                .is_file()
            )
            self.assertTrue(
                Path(
                    tmp_dir / Path("pounce_map_reduce_sliced_score_matching_frames.png")
                )
                .resolve()
                .is_file()
            )


# pylint: enable=duplicate-code
