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

"""DataReader and data classes."""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import ABC

from jax.typing import ArrayLike
from matplotlib import figure

from coreax.reduction import DataReduction, ReductionStrategy


class DataReader(ABC):
    """DataReader."""

    def __init__(
        self,
        original_data: ArrayLike,
        pre_reduction_array: list[list[float]],
        reduction_indices: list[int] = [],
    ) -> None:
        """Initialise DataReader."""
        self.original_data = original_data
        self.pre_reduction_array = pre_reduction_array
        self.reduction_indices = reduction_indices
        # self._dimension_reduction_meta: dict | None

    @classmethod
    def load(cls, original_data: ArrayLike) -> DataReader:
        """
        Construct DataReader.

        Use instead of __init__.

        :param original_data:
        :return:
        """
        # Calculate pre_reduction_array
        pre_reduction_array = []
        return cls(original_data, pre_reduction_array)

    def reduce(
        self,
        reduction_strategy: str | ReductionStrategy,
        data_reducer: str | DataReduction,
    ) -> DataReduction:
        """
        Reduce data.

        :param reduction_strategy:
        :param data_reducer:
        :return:
        """
        raise NotImplementedError

    def render(self, data_reduction: DataReduction | None) -> figure:
        """
        Create matplotlib figure of data.

        :param data_reduction:
        :return:
        """
        raise NotImplementedError

    def reduce_dimension(self, num_dimension: int) -> None:
        """
        Run PCA.

        :param num_dimension: Number of dimensions.
        """
        raise NotImplementedError

    def restore_dimension(self, data_reduction: DataReduction | None) -> ArrayLike:
        """
        Expand principle components into original dimensions.

        :param data_reduction:
        :return:
        """
        raise NotImplementedError
