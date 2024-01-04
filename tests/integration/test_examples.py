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

from examples.david_map_reduce_weighted import main as david_map_reduce_weighted_main
from examples.herding_approximate_gram_matrix import (
    main as herding_approximate_gram_matrix_main,
)
from examples.herding_basic import main as herding_basic_main
from examples.herding_refine import main as herding_refine_main
from examples.herding_stein_weighted import main as herding_stein_weighted_main
from examples.herding_stein_weighted_ssm import main as herding_stein_weighted_ssm_main
from examples.pounce import main as pounce_main
from examples.pounce_map_reduce import main as pounce_map_reduce_main
from examples.pounce_map_reduce_ssm import main as pounce_map_reduce_ssm_main


class TestExamples(unittest.TestCase):
    def assert_is_file(self, path: Path | str) -> None:
        """
        Assert a file exists at a given path.

        :param path: Path to file
        :raises: Exception if file does not exist at given path
        """
        self.assertTrue(
            Path(path).resolve().is_file(), msg=f"File does not exist: {path}"
        )

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

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )

    def test_pounce(self) -> None:
        """
        Test pounce.py example.

        An end-to-end test to check pounce.py runs without error, generates  output, and
        has coreset MMD better than MMD from random sampling.
        """
        directory = Path(os.path.dirname(__file__)) / Path("../../examples/data/pounce")

        # Delete output files if already present
        out_dir = directory / "coreset"
        if out_dir.exists():
            for sub in out_dir.iterdir():
                if sub.name in {"coreset.gif", "frames.png"}:
                    sub.unlink()

        with patch("builtins.print"):
            # Run pounce_map_reduce.py
            mmd_coreset, mmd_random = pounce_main(directory=directory)

            self.assert_is_file(directory / Path("coreset/coreset.gif"))
            self.assert_is_file(directory / Path("coreset/frames.png"))

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )

    def test_pounce_map_reduce(self) -> None:
        """
        Test pounce_map_reduce.py example.

        An end-to-end test to check pounce_map_reduce.py runs without error, generates
        output, and has coreset MMD better than MMD from random sampling.
        """
        directory = Path(os.path.dirname(__file__)) / Path("../../examples/data/pounce")

        # Delete output files if already present
        out_dir = directory / "coreset_map_reduce"
        if out_dir.exists():
            for sub in out_dir.iterdir():
                if sub.name in {"coreset_map_reduce.gif", "frames_map_reduce.png"}:
                    sub.unlink()

        with patch("builtins.print"):
            # Run pounce_map_reduce.py
            mmd_coreset, mmd_random = pounce_map_reduce_main(directory=directory)

            self.assert_is_file(directory / Path("coreset_map_reduce/coreset.gif"))
            self.assert_is_file(directory / Path("coreset_map_reduce/frames.png"))

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )

    def test_pounce_map_reduce_ssm(self) -> None:
        """
        Test pounce_map_reduce_ssm.py example.

        An end-to-end test to check pounce_map_reduce_ssm.py runs without error,
        generates output, and has coreset MMD better than MMD from random sampling.
        """
        directory = Path(os.path.dirname(__file__)) / Path("../../examples/data/pounce")

        # Delete output files if already present
        out_dir = directory / "coreset_map_reduce_sliced_score_matching"
        if out_dir.exists():
            for sub in out_dir.iterdir():
                if sub.name in {
                    "coreset_map_reduce_sliced_score_matching.gif",
                    "frames_map_reduce_sliced_score_matching.png",
                }:
                    sub.unlink()

        with patch("builtins.print"):
            # Run pounce_map_reduce_ssm.py
            mmd_coreset, mmd_random = pounce_map_reduce_ssm_main(directory=directory)

            self.assert_is_file(
                directory
                / Path(
                    "coreset_map_reduce_sliced_score_matching/"
                    "coreset_map_reduce_sliced_score_matching.gif"
                )
            )
            self.assert_is_file(
                directory
                / Path(
                    "coreset_map_reduce_sliced_score_matching/"
                    "frames_map_reduce_sliced_score_matching.png"
                )
            )

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )

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

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )

    def test_herding_approximate_gram_matrix(self) -> None:
        """
        Test herding_approximate_gram_matrix.py example.

        An end-to-end test to check herding_approximate_gram_matrix.py runs without
        error, generates output, and has coreset MMD better than MMD from random
        sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_approximate_gram_matrix.png"
            mmd_coreset, mmd_random = herding_approximate_gram_matrix_main(
                out_path=out_path
            )

            mock_show.assert_has_calls([call(), call()])

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )

    def test_herding_refine(self) -> None:
        """
        Test herding_refine.py example.

        An end-to-end test to check herding_refine.py runs without error, generates
        output, and has coreset MMD better than MMD from random sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_refine.png"
            mmd_coreset, mmd_random = herding_refine_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )

    def test_herding_stein_weighted(self) -> None:
        """
        Test herding_stein_weighted.py example.

        An end-to-end test to check herding_stein_weighted.py runs without error,
        generates output, and has coreset MMD better than MMD from random sampling.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "herding_stein_weighted.png"
            mmd_coreset, mmd_random = herding_stein_weighted_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )

    def test_herding_stein_weighted_ssm(self) -> None:
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
            out_path = Path(tmp_dir) / "herding_stein_weighted_ssm.png"
            mmd_coreset, mmd_random = herding_stein_weighted_ssm_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call()])

            self.assert_is_file(out_path)

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg=(
                    "MMD for random sampling was unexpectedly lower than coreset " "MMD"
                ),
            )
