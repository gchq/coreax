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

"""Classes and associated functionality to use tensor-products of kernel functions."""

from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike

from coreax.data import SupervisedData, is_data
from coreax.kernel import Kernel, _block_data_convert
from coreax.util import pairwise_tuple, tree_leaves_repeat


class TensorProductKernel(eqx.Module):
    r"""
    Define a kernel which is a tensor product of two kernels.

    Given kernel functions :math:`r:\mathcal{X} \times \mathcal{X} \to \mathbb{R}` and
    :math:`l:\mathcal{Y} \times \mathcal{Y} \to \mathbb{R}`, define the tensor product
    kernel :math:`p:(\mathcal{X} \times \mathcal{Y}) \otimes
    (\mathcal{X} \times \mathcal{Y}) \to \mathbb{R}` where
    :math:`k((x_1, y_1),(x_2, y_2)) := r(x_1,x_2)l(y_1,y_2)`

    :param feature_kernel: Instance of :class:`Kernel` acting on data :math:`x`
    :param response_kernel: Instance of :class:`Kernel` acting on supervision :math:`y`
    """

    feature_kernel: Kernel
    response_kernel: Kernel

    def compute_elementwise(
        self, a: tuple[ArrayLike, ArrayLike], b: tuple[ArrayLike, ArrayLike]
    ) -> Array:
        r"""
        Evaluate the kernel on pairs of individual input vectors.

        Vectorisation only becomes relevant in terms of computational speed when we
        have multiple pairs `a`:math:`=(x_1, y_1)``, or `b`:math:`=(x_2, y_2)`.

        :param a: Tuple of vectors :math:`(x_1, y_1)``
        :param b: Tuple of vectors :math:`(x_2, y_2)``
        :return: Kernel evaluated at :math:`((x_1, y_1), (x_2, y_2))`
        """
        return self.feature_kernel.compute_elementwise(
            a[0], b[0]
        ) * self.response_kernel.compute_elementwise(a[1], b[1])

    def compute(self, a: SupervisedData, b: SupervisedData) -> Array:
        r"""
        Evaluate the kernel on input data.

        The 'data' can be any of:
            * tuples of floating numbers (so a single data-point in 1-dimension)
            * tuples of zero-dimensional arrays (so a single data-point in 1-dimension)
            * tuples of vectors (a single-point in multiple dimensions)
            * tuples of arrays (multiple vectors).

        Evaluation is always vectorised.

        :param a: Supervised dataset
        :param b: Supervised dataset
        :return: Kernel evaluations between pairs in `a` and `b`. If `a`:math:`=``b`,
            then this is the Gram matrix corresponding to the tensor-product RKHS inner
            product.
        """
        return pairwise_tuple(self.compute_elementwise)(
            (a.data, a.supervision), (b.data, b.supervision)
        )

    def gramian_row_mean(
        self,
        a: SupervisedData,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute the (blocked) row-mean of the kernel's Gramian matrix.

        A convenience method for calling meth:`compute_mean`. Equivalent to the call
        :code:`compute_mean(a, a, axis=0, block_size=block_size, unroll=unroll)`.

        :param a: Supervised dataset
        :param block_size: Block size parameter passed to :meth:`compute_mean`
        :param unroll: Unroll parameter passed to :meth:`compute_mean`
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        return self.compute_mean(a, a, axis=0, block_size=block_size, unroll=unroll)

    def compute_mean(
        self,
        a: SupervisedData,
        b: SupervisedData,
        axis: Union[int, None] = None,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute the mean of the matrix :math:`K_{ij} = k((x_a, y_a)_i, (x_b, y_b)_j))`.

        The :math:`n \times m` tensor-product kernel matrix
        :math:`K_{ij} = k((x_a, y_a)_i, (x_b, y_b)_j)`, where ``a`` and ``b`` are
        instances of :class:`-coreax.data.SupervisedData` containing :math:`n` and
        :math:`m` (weighted) data pairs respectively, has the following
        (weighted) means:

        - mean (:code:`axis=None`) :math:`\frac{1}{n m}\sum_{i,j=1}^{n, m} K_{ij}`
        - row-mean (:code:`axis=0`) :math:`\frac{1}{n}\sum_{i=1}^{n} K_{ij}`
        - column-mean (:code:`axis=1`) :math:`\frac{1}{m}\sum_{j=1}^{m} K_{ij}`

        The weights of `a` and `b` are used to compute the weighted mean as defined in
        :func:`jax.numpy.average`.

        .. note::
            The conventional 'mean' is a scalar, the 'row-mean' is an :math:`m`-vector,
            while the 'column-mean' is an :math:`n`-vector.

        To avoid materializing the entire matrix (memory cost :math:`\mathcal{O}(n m)`),
        we accumulate the mean over blocks (memory cost :math:`\mathcal{O}(B_a B_b)`,
        where ``B_a`` and ``B_b`` are user-specified block-sizes for blocking the ``a``
        and ``b`` parameters respectively.

        .. note::
            The supervised data ``a`` and/or ``b`` are padded with zero-valued and
            zero-weighted data points, when ``B_a`` and/or ``B_b`` are non-integer
            divisors of ``n`` and/or ``m``. Padding does not alter the result, but does
            provide the block shape stability required by :func:`jax.lax.scan`
            (used for block iteration).

        :param a: Supervised dataset containing :math:`n` (weighted) data pairs
        :param b: Supervised dataset containing :math:`m` (weighted) data pairs
        :param axis: Which axis of the kernel matrix to compute the mean over; a value
            of `None` computes the mean over both axes
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: The (weighted) mean of the kernel matrix :math:`K_{ij}`
        """
        datasets = a, b
        inner_unroll, outer_unroll = tree_leaves_repeat(unroll, len(datasets))
        _block_size = tree_leaves_repeat(block_size, len(datasets))

        # Row-mean is the argument reversed column-mean due to symmetry k(x,y) = k(y,x)
        if axis == 0:
            datasets = datasets[::-1]
            _block_size = _block_size[::-1]
        (block_a, unpadded_len_a), (block_b, _) = jtu.tree_map(
            _block_data_convert, datasets, tuple(_block_size), is_leaf=is_data
        )

        def block_sum(
            accumulated_sum: Array, block_a: SupervisedData
        ) -> tuple[Array, Array]:
            """Block reduce/accumulate over ``a``."""

            def slice_sum(
                accumulated_sum: Array, block_b: SupervisedData
            ) -> tuple[Array, Array]:
                """Block reduce/accumulate over ``b``."""
                w_a, w_b = block_a.weights, block_b.weights
                column_sum_slice = jnp.dot(self.compute(block_a, block_b), w_b)
                accumulated_sum += jnp.dot(w_a, column_sum_slice)
                return accumulated_sum, column_sum_slice

            accumulated_sum, column_sum_slices = jax.lax.scan(
                slice_sum, accumulated_sum, block_b, unroll=inner_unroll
            )
            return accumulated_sum, jnp.sum(column_sum_slices, axis=0)

        accumulated_sum, column_sum_blocks = jax.lax.scan(
            block_sum, jnp.asarray(0.0), block_a, unroll=outer_unroll
        )
        if axis is None:
            return accumulated_sum
        num_rows_padded = block_a.data.shape[0] * block_a.data.shape[1]
        column_sum_padded = column_sum_blocks.reshape(num_rows_padded, -1).sum(axis=1)
        return column_sum_padded[:unpadded_len_a]
