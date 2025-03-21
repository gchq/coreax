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
Example use of greedy kernel points.

This example shows step-by-step usage of greedy kernel points on an analytical example -
see :class:`~coreax.solvers.coresubset.GreedyKernelPoints` for a description of the
method.

We start with the supervised dataset of three points in :math:`\mathbb{R}^2`

.. math::

    X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 1 \end{pmatrix}

with supervision

.. math::

    y^{(1)} = \begin{pmatrix} 0 \\ 1 \\ 5 \end{pmatrix} .

With the :class:`~coreax.kernels.LinearKernel` with
:attr:`~coreax.kernels.LinearKernel.output_scale` :math:`= 1` and
:attr:`~coreax.kernels.LinearKernel.constant` :math:`= 0` and using the
:class:`~coreax.solvers.coresubset.GreedyKernelPoints` algorithm with zero
regularisation (:math:`\lambda = 0`), we seek a coreset of size two without duplicates.

The feature Gramian is

.. math::

    K^{(11)} &= X^T X \\
    &= \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & 1 & 5 \end{pmatrix} .

Following the notation in :class:`~coreax.solvers.coresubset.GreedyKernelPoints`, let
:math:`\mathcal{D}^{(2)} = \{(x_i, y_i)\}_{i=1}^m` be the set of points already selected
for the coreset. Then, :math:`y^{(2)} \in \mathbb{R}^m` is the vector of responses,
:math:`K^{(12)}` is the cross-matrix of kernel evaluations (a subset of columns of the
feature Gramian), and :math:`K^{(22)}` is the kernel matrix on :math:`\mathcal{D}^{(2)}`
(a subset of rows and columns of the feature Gramian). With zero regularisation, the
predictions for a given coreset are given by
:math:`z = K^{(12)} {K^{(22)}}^{-1} y^{(2)}`.

First iteration
---------------

At the first iteration, there are no data points already in the coreset, so we consider
the three candidate coresets, each of size one.

Candidate (0)
^^^^^^^^^^^^^

.. math::

    K^{(12)} &= \begin{pmatrix} 1 \\ 0 \\ 2 \end{pmatrix} ; \\
    K^{(22)} &= \begin{pmatrix} 1 \end{pmatrix} ; \\
    y^{(2)} &= \begin{pmatrix} 0 \end{pmatrix} .

Then, the inverse of the kernel matrix is

.. math::

    {K^{(22)}}^{-1} = \begin{pmatrix} 1 \end{pmatrix} ,

and the prediction is

.. math::

    z = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} ,

so the loss is

.. math::

    L &= \left\| y^{(1)} - z \right\|^2 \\
    &= 0^2 + 1^2 + 5^2 \\
    &= 26 .

Candidate (1)
^^^^^^^^^^^^^

.. math::

    K^{(12)} &= \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} ; \\
    K^{(22)} &= \begin{pmatrix} 1 \end{pmatrix} ; \\
    y^{(2)} &= \begin{pmatrix} 0 \end{pmatrix} .

Then, the inverse of the kernel matrix is

.. math::

    {K^{(22)}}^{-1} = \begin{pmatrix} 1 \end{pmatrix} ,

and the prediction is

.. math::

    z = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} ,

so the loss is

.. math::

    L &= 0^2 + 0^2 + 4^2 \\
    &= 16 .

Candidate (2)
^^^^^^^^^^^^^

.. math::

    K^{(12)} &= \begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix} ; \\
    K^{(22)} &= \begin{pmatrix} 5 \end{pmatrix} ; \\
    y^{(2)} &= \begin{pmatrix} 0 \end{pmatrix} .

Then, the inverse of the kernel matrix is

.. math::

    {K^{(22)}}^{-1} = \begin{pmatrix} \frac{1}{5} \end{pmatrix} ,

and the prediction is

.. math::

    z = \begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix} ,

so the loss is

.. math::

    L &= 2^2 + 0^2 + 0^2 \\
    &= 4 .

Selection
^^^^^^^^^

Index 2 has the lowest loss, so joins the coreset.

Second iteration
----------------

We consider two candidate coresets of size two, each containing data corresponding to
index 2 as the first element with another index in the second element.

Candidate (2 0)
^^^^^^^^^^^^^^^

.. math::

    K^{(12)} &= \begin{pmatrix} 2 & 1 \\ 1 & 0 \\ 5 & 2 \end{pmatrix} ; \\
    K^{(22)} &= \begin{pmatrix} 5 & 2 \\ 2 & 1 \end{pmatrix} ; \\
    y^{(2)} &= \begin{pmatrix} 5 \\ 0 \end{pmatrix} .

Then, the inverse of the kernel matrix is

.. math::

    {K^{(22)}}^{-1} = \begin{pmatrix} 1 & -2 \\ -2 & 5 \end{pmatrix} ,

and the prediction is

.. math::

    z &= K^{(12)} {K^{(22)}}^{-1} y^{(2)} \\
    &= \begin{pmatrix} 2 & 1 \\ 1 & 0 \\ 5 & 2 \end{pmatrix}
        \begin{pmatrix} 1 & -2 \\ -2 & 5 \end{pmatrix}
        \begin{pmatrix} 5 \\ 0 \end{pmatrix} \\
    &= \begin{pmatrix} 2 & 1 \\ 1 & 0 \\ 5 & 2 \end{pmatrix}
        \begin{pmatrix} 5 \\ -10 \end{pmatrix} \\
    &= \begin{pmatrix} 0 \\ 5 \\ 5 \end{pmatrix} ,

so the loss is

.. math::

    L &= 0^2 + 4^2 + 0^2 \\
    &= 16 .

Candidate (2 1)
^^^^^^^^^^^^^^^

.. math::

    K^{(12)} &= \begin{pmatrix} 2 & 0 \\ 1 & 1 \\ 5 & 1 \end{pmatrix} ; \\
    K^{(22)} &= \begin{pmatrix} 5 & 1 \\ 1 & 1 \end{pmatrix} ; \\
    y^{(2)} &= \begin{pmatrix} 5 \\ 1 \end{pmatrix} .

Then, the inverse of the kernel matrix is

.. math::

    {K^{(22)}}^{-1} = \frac{1}{4} \begin{pmatrix} 1 & -1 \\ -1 & 5 \end{pmatrix} ,

and the prediction is

.. math::

    z &= K^{(12)} {K^{(22)}}^{-1} y^{(2)} \\
    &= \frac{1}{4} \begin{pmatrix} 2 & 0 \\ 1 & 1 \\ 5 & 1 \end{pmatrix}
        \begin{pmatrix} 1 & -1 \\ -1 & 5 \end{pmatrix}
        \begin{pmatrix} 5 \\ 1 \end{pmatrix} \\
    &= \begin{pmatrix} 2 & 0 \\ 1 & 1 \\ 5 & 1 \end{pmatrix}
        \begin{pmatrix} 4 \\ 0 \end{pmatrix} \\
    &= \begin{pmatrix} 2 \\ 1 \\ 5 \end{pmatrix} ,

so the loss is

.. math::

    L &= 2^2 + 0^2 + 0^2 \\
    &= 4 .

Final selection
^^^^^^^^^^^^^^^

The second candidate has the lower loss, so the final coreset consists of indices
:math:`\begin{pmatrix} 2 & 1 \end{pmatrix}`. In terms of original data, this can be
expressed as

.. math::

    \hat{X} &= \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix} ; \\
    \hat{y} &= \begin{pmatrix} 5 \\ 1 \end{pmatrix} .
"""

import jax
import jax.numpy as jnp

from coreax.coreset import Coresubset
from coreax.data import SupervisedData
from coreax.kernels import LinearKernel
from coreax.solvers import GreedyKernelPoints


def main() -> Coresubset[SupervisedData]:
    """
    Run the greedy kernel points analytical example.

    :return: Object containing coreset indices and materialised coreset.
    """
    # Create supervised data
    in_data = SupervisedData(
        data=jnp.array([[1, 0], [0, 1], [2, 1]]), supervision=jnp.array([0, 1, 5])
    )

    # Set up solver
    random_seed = 1_989
    random_key = jax.random.key(random_seed)
    solver = GreedyKernelPoints(
        coreset_size=2,
        random_key=random_key,
        feature_kernel=LinearKernel(output_scale=1, constant=0),
    )

    # Calculate coreset
    coreset, _ = solver.reduce(in_data)

    # Display coreset, noting that indices and supervision need to be squeezed to
    # one-dimensional vectors
    print(f"Coresubset has indices {jnp.squeeze(coreset.indices.data)}.")
    print(
        f"Coresubset is\n{coreset.points.data}\nwith supervision "
        f"{jnp.squeeze(coreset.points.supervision)}."
    )

    return coreset


if __name__ == "__main__":
    main()
