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

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

import coreax.reduction as cr
import coreax.util as cu


class DataReader(ABC):
    """Base class for reading data."""

    loaded_file = None

    @abstractmethod
    def load(self, file_path: Path):
        r"""
        Read in a file and save to class instance.

        :param file_path: Path to file to be read in
        """

    @abstractmethod
    def reduce(
        self,
        reduction_strategy: str | cr.ReductionStrategy,
        data_reduction: str | cr.DataReduction,
    ) -> cr.DataReduction:
        r"""
        Reduce the original data.

        :param reduction_strategy: Reduction strategy to be applied
        :param data_reduction: Data reduction of the original data
        """

    @abstractmethod
    def render(self, data_reduction: cr.DataReduction | None):
        r"""
        Render the original data or the coreset onscreen.

        :param data_reduction: Data reduction of the original data
        """

    @abstractmethod
    def save_reduction(self, file_path, data_reduction: cr.DataReduction):
        r"""
        Save the original data or the coreset to a file.

        :param file_path: File path to save data to
        :param data_reduction: Data reduction of the original data
        """


class Image(DataReader):
    """
    TODO
    """

    def __init__(self):
        self.loaded_data = None

    def load(self, file_path):
        # path to original image
        orig = cv2.imread(str(file_path))

        print(f"Image dimensions: {orig.shape}")
        X_ = np.column_stack(np.where(orig < 255))
        vals = orig[orig < 255]
        X = np.column_stack((X_, vals)).astype(np.float32)

        self.loaded_data = X

    def reduce(
        self,
        reduction_strategy: str | cr.ReductionStrategy,
        data_reduction: str | cr.DataReduction,
    ):
        reduction_strategy_obj = cr.reduction_strategy_factory.get(reduction_strategy)
        data_reduction_obj = cr.data_reduction_factory.get(data_reduction)
        reduction_strategy_obj.__init__(data_reduction_obj)
        reduction_strategy_obj.reduce()

        return reduction_strategy_obj

    def render(self, data_reduction: cr.DataReduction | None):
        if isinstance(data_reduction, cr.DataReduction):
            coreset = self.loaded_data[data_reduction.reduction_indices]
            plt.figure(figsize=(10, 5))
            plt.imshow(coreset, cmap="gray")
            plt.title("Coreset")
            plt.axis("off")
            plt.show()
        else:
            plt.figure(figsize=(10, 5))
            plt.imshow(self.loaded_file, cmap="gray")
            plt.title("Original")
            plt.axis("off")
            plt.show()

    def save_reduction(self, file_path, data_reduction: cr.DataReduction | None):
        data_reduction.render(file_path)
        if isinstance(data_reduction, cr.DataReduction):
            coreset = self.loaded_data[data_reduction.reduction_indices]
            plt.figure(figsize=(10, 5))
            plt.imshow(coreset, cmap="gray")
            plt.title("Coreset")
            plt.axis("off")
            plt.savefig(file_path)
        else:
            plt.figure(figsize=(10, 5))
            plt.imshow(self.loaded_file, cmap="gray")
            plt.title("Original")
            plt.axis("off")
            plt.show()


data_reader_factory = cu.ClassFactory(DataReader)
data_reader_factory.register("image", Image)
