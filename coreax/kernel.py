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

r"""
Classes and associated functionality to use kernel functions.

A kernel is a non-negative, real-valued integrable function that can take two inputs,
``x`` and ``y``, and returns a value that decreases as ``x`` and ``y`` move further away
in space from each other. Note that *further* here may account for cyclic behaviour in
the data, for example.

In this library, we often use kernels as a smoothing tool: given a dataset of distinct
points, we can reconstruct the underlying data generating distribution through smoothing
of the data with kernels.

Some kernels are parameterizable and may represent other well known kernels when given
appropriate parameter values. For example, the :class:`SquaredExponentialKernel`,

.. math::
    k(x,y) = \text{output_scale} * \exp (-||x-y||^2/2 * \text{length_scale}^2),
which if parameterized with an ``output_scale`` of
:math:`\frac{1}{\sqrt{2\pi} \,*\, \text{length_scale}}`, yields the well known
Gaussian kernel.

A :class:`~coreax.kernel.Kernel` must implement a
:meth:`~coreax.kernel.Kernel.compute_elementwise` method, which returns the floating
point value after evaluating the kernel on two floats, ``x`` and ``y``. Additional
methods, such as :meth:`Kernel.grad_x_elementwise`, can optionally be overridden to
improve performance. The canonical example is when a suitable closed-form representation
of a higher-order gradient can be used to avoid the expense of performing automatic
differentiation.

Such an example can be seen in :class:`SteinKernel`, where the analytic forms of
:meth:`Kernel.divergence_x_grad_y` are significantly cheaper to compute that the
automatic differentiated default.
"""

from typing_extensions import deprecated

from coreax.kernels.base import (
    AdditiveKernel,  # pyright: ignore=reportAssignmentType
    DuoCompositeKernel,
    ProductKernel,  # pyright: ignore=reportAssignmentType
    ScalarValuedKernel,
    UniCompositeKernel,
)
from coreax.kernels.scalar_valued import (
    ExponentialKernel,  # pyright: ignore=reportAssignmentType
    LaplacianKernel,  # pyright: ignore=reportAssignmentType
    LinearKernel,  # pyright: ignore=reportAssignmentType
    LocallyPeriodicKernel,  # pyright: ignore=reportAssignmentType
    PCIMQKernel,  # pyright: ignore=reportAssignmentType
    PeriodicKernel,  # pyright: ignore=reportAssignmentType
    PolynomialKernel,  # pyright: ignore=reportAssignmentType
    RationalQuadraticKernel,  # pyright: ignore=reportAssignmentType
    SquaredExponentialKernel,  # pyright: ignore=reportAssignmentType
    SteinKernel,  # pyright: ignore=reportAssignmentType
)


# pylint:disable = abstract-method
# pylint:disable = function-redefined
@deprecated(
    "Renamed to `ScalarValuedKernel`; "
    + " moved to `coreax.kernels.base.ScalarValuedKernel`; "
    + " will be removed in version 0.3.0"
)
class Kernel(ScalarValuedKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ScalarValuedKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Renamed to `UniCompositeKernel`; "
    + " moved to `coreax.kernels.base.UniCompositeKernel`; "
    + " will be removed in version 0.3.0"
)
class CompositeKernel(UniCompositeKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.UniCompositeKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Renamed to `DuoCompositeKernel`; "
    + " moved to `coreax.kernels.base.DuoCompositeKernel`; "
    + " will be removed in version 0.3.0"
)
class PairedKernel(DuoCompositeKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.DuoCompositeKernel`.

    Will be removed in version 0.3.0
    """


# pylint:enable = abstract-method


@deprecated(
    "Moved to `coreax.kernels.base.AdditiveKernel`;  will be removed in version 0.3.0"
)
class AdditiveKernel(AdditiveKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.AdditiveKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.ProductKernel`;  will be removed in version 0.3.0"
)
class ProductKernel(ProductKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ProductKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LinearKernel`; "
    + " will be removed in version 0.3.0"
)
class LinearKernel(LinearKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.LinearKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LinearKernel`; "
    + " will be removed in version 0.3.0"
)
class PolynomialKernel(PolynomialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PolynomialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.SquaredExponentialKernel`; "
    + " will be removed in version 0.3.0"
)
class SquaredExponentialKernel(SquaredExponentialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.SquaredExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.ExponentialKernel`; "
    + " will be removed in version 0.3.0"
)
class ExponentialKernel(ExponentialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.RationalQuadraticKernel`; "
    + " will be removed in version 0.3.0"
)
class RationalQuadraticKernel(RationalQuadraticKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.RationalQuadraticKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.PeriodicKernel`; "
    + " will be removed in version 0.3.0"
)
class PeriodicKernel(PeriodicKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PeriodicKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LocallyPeriodicKernel`; "
    + " will be removed in version 0.3.0"
)
class LocallyPeriodicKernel(LocallyPeriodicKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.LocallyPeriodicKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LaplacianKernel`; "
    + " will be removed in version 0.3.0"
)
class LaplacianKernel(LaplacianKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.SquaredExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.PCIMQKernel`; "
    + " will be removed in version 0.3.0"
)
class PCIMQKernel(PCIMQKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PCIMQKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.SteinKernel`; "
    + " will be removed in version 0.3.0"
)
class SteinKernel(SteinKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PCIMQKernel`.

    Will be removed in version 0.3.0
    """


# pylint:enable = function-redefined
