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

from abc import ABC
from jax import Array
from jax.typing import ArrayLike

import coreax.util as cu
import coreax.weights as we


class DataReduction(ABC):
    """
    Methods for reducing data.
    """

    def __init__(self, original_data: ArrayLike, weighting: str):
        r"""
        Define a ... TODO
        """
        self.original_data = original_data
        self.reduced_data = original_data.copy()
        self.weighting = weighting

    def solve_weights(
            self,
            kernel: cu.KernelFunction
    ) -> Array | None:
        """
        Solve for weights. Currently implemented are MMD and SBQ weights.

        TODO: update when weights.py is OOPed.
        """

        if self.weighting is None:
            return None
        elif self.weighting == 'MMD':
            return we.simplex_weights(
                self.original_data,
                self.reduced_data,
                kernel
            )
        elif self.weighting == 'SBQ':
            return we.calculate_BQ_weights(
                self.original_data,
                self.reduced_data,
                kernel
            )
        else:
            raise ValueError(f"weight type '{self.weighting}' not recognised.")

    def fit(
            self,
            kernel
            ):
        return NotImplementedError

    def refine(self):
        return NotImplementedError

    def compute_metric(self):
        return NotImplementedError

    def render(self):
        """
        TODO
        """
        return self.original_data.render_reduction

    def save(self):
        """
        TODO
        """
        return self.original_data.save_reduction


class ReductionStrategy(ABC):
    """
    TODO
    """

    def __init__(self, reduction_method: DataReduction):
        """
        TODO
        """

        self.reduction_method = reduction_method

    def reduce(self, original_data, weighted):
        """
        TODO
        """

        return DataReduction(original_data, weighted)


class SizeReduce(ReductionStrategy):

    def __init__(self, n):

        self.n = n

    def generate_coreset(self):
        # TODO: return Coreset from coreset.py when ready
        return NotImplementedError


class ErrorReduce(ReductionStrategy):

    def __init__(self, eps):

        self.eps = eps

    def reduce_error(self):
        # TODO: vary n to meet probabilistically...
        return NotImplementedError


class MapReduce(ReductionStrategy):

    def __init__(self, n):

        self.n = n
