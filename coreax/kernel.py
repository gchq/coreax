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
    AdditiveKernel as _AdditiveKernel,
    DuoCompositeKernel as _DuoCompositeKernel,
    ProductKernel as _ProductKernel,
    ScalarValuedKernel as _ScalarValuedKernel,
    UniCompositeKernel as _UniCompositeKernel,
)
from coreax.kernels.scalar_valued import (
    ExponentialKernel as _ExponentialKernel,
    LaplacianKernel as _LaplacianKernel,
    LinearKernel as _LinearKernel,
    LocallyPeriodicKernel as _LocallyPeriodicKernel,
    PCIMQKernel as _PCIMQKernel,
    PeriodicKernel as _PeriodicKernel,
    PolynomialKernel as _PolynomialKernel,
    RationalQuadraticKernel as _RationalQuadraticKernel,
    SquaredExponentialKernel as _SquaredExponentialKernel,
    SteinKernel as _SteinKernel,
)


# pylint:disable = abstract-method
@deprecated(
    "Renamed to `ScalarValuedKernel`; "
    + " moved to `coreax.kernels.base.ScalarValuedKernel`; "
    + " will be removed in version 0.3.0"
)
class Kernel(_ScalarValuedKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ScalarValuedKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Renamed to `UniCompositeKernel`; "
    + " moved to `coreax.kernels.base.UniCompositeKernel`; "
    + " will be removed in version 0.3.0"
)
class CompositeKernel(_UniCompositeKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.UniCompositeKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Renamed to `DuoCompositeKernel`; "
    + " moved to `coreax.kernels.base.DuoCompositeKernel`; "
    + " will be removed in version 0.3.0"
)
class PairedKernel(_DuoCompositeKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.DuoCompositeKernel`.

    Will be removed in version 0.3.0
    """


# pylint:enable = abstract-method


@deprecated(
    "Moved to `coreax.kernels.base.AdditiveKernel`;  will be removed in version 0.3.0"
)
class AdditiveKernel(_AdditiveKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.AdditiveKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.ProductKernel`;  will be removed in version 0.3.0"
)
class ProductKernel(_ProductKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ProductKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LinearKernel`; "
    + " will be removed in version 0.3.0"
)
class LinearKernel(_LinearKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.LinearKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LinearKernel`; "
    + " will be removed in version 0.3.0"
)
class PolynomialKernel(_PolynomialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PolynomialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.SquaredExponentialKernel`; "
    + " will be removed in version 0.3.0"
)
class SquaredExponentialKernel(_SquaredExponentialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.SquaredExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.ExponentialKernel`; "
    + " will be removed in version 0.3.0"
)
class ExponentialKernel(_ExponentialKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.ExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.RationalQuadraticKernel`; "
    + " will be removed in version 0.3.0"
)
class RationalQuadraticKernel(_RationalQuadraticKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.RationalQuadraticKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.PeriodicKernel`; "
    + " will be removed in version 0.3.0"
)
class PeriodicKernel(_PeriodicKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PeriodicKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LocallyPeriodicKernel`; "
    + " will be removed in version 0.3.0"
)
class LocallyPeriodicKernel(_LocallyPeriodicKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.LocallyPeriodicKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.LaplacianKernel`; "
    + " will be removed in version 0.3.0"
)
class LaplacianKernel(_LaplacianKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.SquaredExponentialKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.PCIMQKernel`; "
    + " will be removed in version 0.3.0"
)
class PCIMQKernel(_PCIMQKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PCIMQKernel`.

    Will be removed in version 0.3.0
    """


@deprecated(
    "Moved to `coreax.kernels.base.scalar_valued.SteinKernel`; "
    + " will be removed in version 0.3.0"
)
class SteinKernel(_SteinKernel):
    """
    Deprecated reference to :class:`~coreax.kernels.PCIMQKernel`.

    Will be removed in version 0.3.0
    """
