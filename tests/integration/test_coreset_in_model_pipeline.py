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
Integration test showing how coresets can be included in a machine learning pipeline.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from examples.coreset_in_model_pipeline import main as coreset_in_model_pipeline_main

# Integration tests are split across several files, to allow serial calls and avoid
# sharing of JIT caches between tests. As a result, ignore the pylint warnings for
# duplicated-code.
# pylint: disable=duplicate-code


class TestCoresetInModelPipeline(unittest.TestCase):
    """
    Test end-to-end code run with a coresets machine learning pipeline example.
    """

    def test_coreset_in_model_pipeline(self) -> None:
        """
        Test coreset_in_model_pipeline.py example.

        An end-to-end test to check coreset_in_model_pipeline.py runs without error.
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("builtins.print"),
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Run weighted herding example
            out_path = Path(tmp_dir) / "coreset_in_model_pipeline.png"
            coreset_in_model_pipeline_main(out_path=out_path)

            mock_show.assert_has_calls([call(), call(), call()])

            self.assertTrue(Path(out_path).resolve().is_file())


# pylint: enable=duplicate-code
