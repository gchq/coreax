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
This module contains classes for methods of constructing coresets.

Coresets are a type of data reduction, so these inherit from
:class:`~coreax.reduction.DataReduction`. The aim is to select a samll set of indices
that represent the key features of a larger dataset.

The abstract base class is :class:`Coreset`. Concrete implementations are:

*   :class:`KernelHerding` defines the kernel herding method for both regular and Stein
    kernels.
*   :class:`RandomSample` selects points for the coreset using random sampling. It is
    typically only used for benchmarking against other coreset methods.
"""

from reduction import DataReduction, data_reduction_factory


class Coreset(DataReduction):
    """Abstract base class for a method to construct a coreset."""


class KernelHerding(Coreset):
    """
    Apply kernel herding to a dataset.

    This class works with all kernels, including Stein kernels.
    """


data_reduction_factory.register("kernel_herding", KernelHerding)
