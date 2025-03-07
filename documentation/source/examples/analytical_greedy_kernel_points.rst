Analytical example with greedy kernel points
============================================

Step-by-step usage of greedy kernel points on an analytical example (see
:class:`~coreax.solvers.coresubset.GreedyKernelPoints` for a description of the method).

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
regularisation, we seek a coreset of size two.

The feature Gramian is

.. math::

    K^{(11)} &= X^T X \\
    &= \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & 1 & 5 \end{pmatrix} .

Let :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` be a supervised dataset of
:math:`n` pairs, where, with respect to some scalar-valued feature kernel, the original
feature Gramian is

.. math::

    K^{(11)} = \begin{pmatrix}
                1 & \frac{1}{2} & \frac{1}{2} \\
                \frac{1}{2} & 1 & \frac{1}{5} \\
                \frac{1}{2} & \frac{1}{5} & 1
                \end{pmatrix}

and the vector of responses is

.. math::

    Y^{(1)} = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix} .

We wish to find a coresubset of size two.
