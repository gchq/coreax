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
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from examples.david import main as d
from examples.pounce import main as p


def assertisfile(path):
    if not Path(path).resolve().is_file():
        raise AssertionError("File does not exist: %s" % str(path))


class TestExamples(unittest.TestCase):

    def test_david(self):
        """
        Test david.py example
        """

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch("matplotlib.pyplot.show") as mock_show:

                # run david.py
                inpath_ = Path.cwd().parent / Path("examples/data/david_orig.png")
                outpath_ = Path(tmp_dir) / 'david_coreset.png'
                d(inpath=str(inpath_), outpath=outpath_)

                mock_show.assert_called_once()

                assertisfile(outpath_)

    def test_pounce(self):
        """
        Test pounce.py example
        """

        dir_ = Path.cwd().parent / Path("examples/data/pounce")

        # delete output files if already present
        out_dir = dir_ / "coreset"
        if out_dir.exists():
            for sub in out_dir.iterdir():
                if sub.name in ["coreset.gif", "frames.png"]:
                    sub.unlink()

        with patch('builtins.print'):
            # run pounce.py
            p(dir_=str(dir_))

            assertisfile(dir_ / Path('coreset/coreset.gif'))
            assertisfile(dir_ / Path('coreset/frames.png'))
