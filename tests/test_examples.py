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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from examples.david import main as david_main
from examples.pounce import main as pounce_main
from examples.weighted_herding import main as weighted_herding_main


class TestExamples(unittest.TestCase):
    def assert_is_file(self, path):
        self.assertTrue(
            Path(path).resolve().is_file(), msg=f"File does not exist: {path}"
        )

    def test_david(self):
        """
        Test david.py example

        An end-to-end test to check david.py runs without error, generates output, and has coreset MMD better
        than MMD from random sampling.
        """

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "builtins.print"
        ) as mock_print, patch("matplotlib.pyplot.show") as mock_show:
            # run david.py
            in_path = Path.cwd().parent / Path("examples/data/david_orig.png")
            out_path = Path(tmp_dir) / "david_coreset.png"
            mmd_coreset, mmd_random = david_main(in_path=in_path, out_path=out_path)

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

    def test_pounce(self):
        """
        Test pounce.py example

        An end-to-end test to check pounce.py runs without error, generates output, and has coreset MMD
        better than MMD from random sampling.
        """

        directory = Path.cwd().parent / Path("examples/data/pounce")

        # delete output files if already present
        out_dir = directory / "coreset"
        if out_dir.exists():
            for sub in out_dir.iterdir():
                if sub.name in {"coreset.gif", "frames.png"}:
                    sub.unlink()

        with patch("builtins.print"):
            # run pounce.py
            mmd_coreset, mmd_random = pounce_main(directory=directory)

            self.assert_is_file(directory / Path("coreset/coreset.gif"))
            self.assert_is_file(directory / Path("coreset/frames.png"))

            self.assertLess(
                mmd_coreset,
                mmd_random,
                msg="MMD for random sampling was unexpectedly lower than coreset MMD",
            )

    def test_weighted_herding(self):
        """
        Test weighted_herding.py example

        An end-to-end test to check weighted_herding.py runs without error, generates output, and has coreset
        MMD better than MMD from random sampling.
        """

        with tempfile.TemporaryDirectory() as tmp_dir, patch("builtins.print"), patch(
            "matplotlib.pyplot.show"
        ) as mock_show:
            with self.subTest(msg="Weighted herding"):
                # run weighted herding example
                outpath = Path(tmp_dir) / "weighted_herding.png"
                mmd_coreset, mmd_random = weighted_herding_main(
                    out_path=outpath, weighted=True
                )

                mock_show.assert_has_calls([call(), call()])

                self.assert_is_file(outpath)

                self.assertLess(
                    mmd_coreset,
                    mmd_random,
                    msg="MMD for random sampling was unexpectedly lower than coreset MMD",
                )

            with self.subTest(msg="Unweighted herding"):
                # run weighted herding example
                outpath = Path(tmp_dir) / "unweighted_herding.png"
                mmd_coreset, mmd_random = weighted_herding_main(
                    out_path=outpath, weighted=False
                )

                mock_show.assert_has_calls([call(), call()])

                self.assert_is_file(outpath)

                self.assertLess(
                    mmd_coreset,
                    mmd_random,
                    msg="MMD for random sampling was unexpectedly lower than coreset MMD",
                )
