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

"""Per-feature input scaling for scalar-valued kernels."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Shaped
from typing_extensions import override

from coreax.kernels.base import UniCompositeKernel


def _convert_length_scale(value: ArrayLike) -> Array:
    """Convert a positive scalar or feature vector without hiding traced values."""
    scale = jnp.asarray(value)
    if scale.ndim > 1 or scale.size == 0 or jnp.iscomplexobj(scale):
        raise ValueError("'length_scale' must be a real scalar or non-empty vector")
    scale = scale.astype(jnp.result_type(scale, 0.0))
    return eqx.error_if(
        scale,
        jnp.any(~jnp.isfinite(scale) | (scale <= 0)),
        "'length_scale' must contain only finite, positive values",
    )


class AnisotropicKernel(UniCompositeKernel):
    r"""
    Apply independent length scales to the input features of a base kernel.

    For positive scales :math:`l`, define
    :math:`k_l(x, y) = k(x / l, y / l)`, with elementwise division.
    A scalar scale is shared by all features. A vector must contain exactly one
    scale per feature. This preserves the positive semi-definiteness of the base
    kernel and supports the usual kernel evaluation and derivative methods.

    Existing base-kernel parameters are retained. For an anisotropic radial kernel,
    leave its own length scale at one and supply the feature scales here. When
    constructing a Stein kernel, wrap its base kernel before applying the Stein
    operator so the score remains expressed in the original coordinates.

    :param base_kernel: Kernel evaluated on scaled inputs
    :param length_scale: Positive finite scalar or vector of feature length scales
    """

    length_scale: Shaped[Array, " d"] | Shaped[Array, ""] = eqx.field(
        converter=_convert_length_scale
    )

    @override
    def compute_elementwise(self, x, y):
        return self.base_kernel.compute_elementwise(self._scale(x), self._scale(y))

    @override
    def grad_x_elementwise(self, x, y):
        gradient = (
            self.base_kernel.grad_x_elementwise(self._scale(x), self._scale(y))
            / self.length_scale
        )
        return jnp.reshape(gradient, jnp.shape(x))

    @override
    def grad_y_elementwise(self, x, y):
        gradient = (
            self.base_kernel.grad_y_elementwise(self._scale(x), self._scale(y))
            / self.length_scale
        )
        return jnp.reshape(gradient, jnp.shape(y))

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        x = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.result_type(x, 0.0)))
        y = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.result_type(y, 0.0)))
        return super().divergence_x_grad_y_elementwise(x, y)

    def _scale(self, point: ArrayLike) -> Array:
        """Check feature dimensions before permitting NumPy broadcasting."""
        point = jnp.asarray(point)
        if jnp.ndim(self.length_scale) and jnp.atleast_1d(point).shape != jnp.shape(
            self.length_scale
        ):
            raise ValueError("'length_scale' must have one entry per input feature")
        return point / self.length_scale
