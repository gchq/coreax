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

from jax import Array
from jax.typing import ArrayLike

import coreax.data as cd
import coreax.reduction as cr


class DataReaderConcrete(cd.DataReader):
    """Concrete implementation of DataReader class to allow testing."""

    @classmethod
    def load(cls, original_data: ArrayLike) -> cd.DataReader:
        raise NotImplementedError

    def format(self, coreset: cr.Coreset) -> Array:
        raise NotImplementedError

    def render(self, coreset: cr.Coreset | None) -> None:
        raise NotImplementedError


class TestDataReader(unittest.TestCase):
    """Test operation of DataReader class."""

    def test_init_scalars(self):
        """Test that scalars are cast properly."""
        actual = DataReaderConcrete(original_data=1, pre_coreset_array=2)
        self.assertEqual(actual.original_data, Array(1))
        self.assertEqual(actual.pre_coreset_array, Array([[2]]))


if __name__ == "__main__":
    unittest.main()
