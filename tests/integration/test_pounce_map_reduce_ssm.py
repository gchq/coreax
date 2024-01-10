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
import unittest
from pathlib import Path
from unittest.mock import patch

import coreax.util
from examples.pounce_map_reduce_ssm import main as pounce_map_reduce_ssm_main


class TestPounceMapReduce(unittest.TestCase):
    """
    Test end-to-end code run with a video example.

    Memory requirements are reduced with a map reduce algorithm. A Stein kernel is used
    with a score function approximated via a neural network.
    """

    def test_pounce_map_reduce_ssm(self) -> None:
        """
        Test pounce_map_reduce_ssm.py example.

        An end-to-end test to check pounce_map_reduce_ssm.py runs without error,
        generates output, and has coreset MMD better than MMD from random sampling.
        """
        in_path = Path(os.path.dirname(__file__)) / Path(
            "../../examples/data/pounce/pounce.gif"
        )

        # Delete output files if already present
        out_path = Path("../../examples/pounce_map_reduce_ssm/")
        if out_path.exists():
            for sub in out_path.iterdir():
                if sub.name in {
                    "pounce_map_reduce_sliced_score_matching_coreset.gif",
                    "pounce_map_reduce_sliced_score_matching_frames.png",
                }:
                    sub.unlink()

        with patch("builtins.print"):
            # Run pounce_map_reduce_ssm.py
            mmd_coreset, mmd_random = pounce_map_reduce_ssm_main(
                in_path=in_path, out_path=out_path
            )

            coreax.util.assert_is_file(
                out_path / Path("pounce_map_reduce_sliced_score_matching_coreset.gif")
            )
            coreax.util.assert_is_file(
                out_path / Path("pounce_map_reduce_sliced_score_matching_frames.png")
            )

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )
