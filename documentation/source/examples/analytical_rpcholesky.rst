Analytical example with RPCholesky
==================================

In this example, we have data of:

.. math::
    x = \begin{pmatrix}
        0.5 & 0.2 \\
        0.4 & 0.6 \\
        0.8 & 0.3
    \end{pmatrix}

We choose a ``SquaredExponentialKernel`` with ``length_scale`` of :math:`\frac{1}{\sqrt{2}}`, which produces the following Gram matrix:

.. math::
    \begin{pmatrix}
        1.0 & 0.84366477 & 0.90483737 \\
        0.84366477 & 1.0 & 0.7788007 \\
        0.90483737 & 0.7788007 & 1.0
    \end{pmatrix}

Note that we do not need to precompute the full Gram matrix, the algorithm
only needs to evaluate the pivot column at each iteration.

The RPCholesky algorithm iteratively builds a coreset by:
    - Sampling pivot points based on the residual diagonal of the kernel Gram matrix
    - Updating an approximation matrix and the residual diagonal

We ask for a coreset of size 2 in this example. We start with an empty coreset
and an approximation matrix :math:`F = \mathbf{0}_{N \times k}`,
where :math:`N = 3, k = 2` in our case.

We first compute the diagonal of the Gram matrix as:

.. math::
    d = \begin{pmatrix}
        1 \\
        1 \\
        1
    \end{pmatrix}

For the first iteration (i=0):

1. We sample a pivot point proportional to their value on the diagonal. All choices are equally likely, so let us suppose we choose the pivot with index = 2.

2. We now compute g, the column at index 2, as:

.. math::
    g = \begin{pmatrix}
    0.90483737 \\
    0.7788007 \\
    1.0
    \end{pmatrix}

3. Remove overlap with previously chosen columns (not needed on the first iteration).

4. Update the approximation matrix:

.. math::
    F[:, 0] = g / \sqrt{(g[2])} = \begin{pmatrix}
    0.90483737 \\
    0.7788007 \\
    1.0
    \end{pmatrix}

5. Update the residual diagonal:

.. math::
    d = d - |F[:,0]|^2 = \begin{pmatrix}
    0.18126933 \\
    0.39346947 \\
    0
    \end{pmatrix}

For the second iteration (i=1):

1. We again sample a pivot point proportional to their value on the updated residual diagonal, :math:`d`. Let's suppose we choose the most likely pivot here (index=1).

2. We now compute g, the column at index 1, as:

.. math::
    g = \begin{pmatrix}
    0.84366477 \\
    1.0 \\
    0.7788007
    \end{pmatrix}

3. Remove overlap with previously chosen columns:

.. math::
    g = g - F[:, 0] F[1, 0] = \begin{pmatrix}
    0.13897679 \\
    0.39346947 \\
    0
    \end{pmatrix}

4. Update the approximation matrix:

.. math::
    F[:, 1] = g / \sqrt{(g[1])} = \begin{pmatrix}
    0.22155766 \\
    0.62727145 \\
    0
    \end{pmatrix}

5. Update the residual diagonal:

.. math::
    d = d - |F[:,0]|^2 = \begin{pmatrix}
      0.13218154 \\
      0 \\
      0
    \end{pmatrix}

After this iteration, the final state is:

.. math::
    F = \begin{pmatrix}
    0.90483737 & 0.22155766 \\
    0.7788007 & 0.62727145 \\
    1.0 & 0
    \end{pmatrix}, \quad
    d = \begin{pmatrix}
    0.13218154 \\
    0 \\
    0
    \end{pmatrix}, \quad
    S = \{2, 1\}

This completes the coreset of size :math:`k = 2`.

.. code-block::
    import jax.numpy as jnp
    import jax.random as jr
    from unittest.mock import patch

    from coreax import Data, SquaredExponentialKernel
    from coreax.solvers import RPCholesky

    # Setup example data
    coreset_size = 2
    x = jnp.array(
        [
            [0.5, 0.2],
            [0.4, 0.6],
            [0.8, 0.3],
        ]
    )

    # Define a kernel
    length_scale = 1.0 / jnp.sqrt(2)
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    # Create a mock for the random choice function
    def deterministic_choice(*_, p, **__):
        """
        Return the index of largest element of p.

        If there is a tie, return the largest index.
        This is used to mimic random sampling, where we have a deterministic
        sampling approach.
        """
        # Find indices where the value equals the maximum
        is_max = p == p.max()
        # Convert boolean mask to integers and multiply by index
        # This way, we'll get the highest index where True appears
        indices = jnp.arange(p.shape[0])
        return jnp.where(is_max, indices, -1).max()


    # Generate the coreset
    data = Data(x)
    solver = RPCholesky(
        coreset_size=coreset_size,
        random_key=jr.PRNGKey(0),  # Fixed seed for reproducibility
        kernel=kernel,
        unique=True,
    )

    # Mock the random choice function
    with patch("jax.random.choice", deterministic_choice):
        coreset, solver_state = solver.reduce(data)

    # Independently computed gramian diagonal
    expected_gramian_diagonal = jnp.array([0.13218154, 0.0, 0.0])

    # Coreset indices forced by our mock choice function
    expected_coreset_indices = jnp.array([2, 1])

    # Inspect results
    print("Chosen coreset:")
    print(coreset.unweighted_indices)  # The coreset_indices
    print(coreset.coreset.data)  # The data-points in the coreset
    print("Residual diagonal:")
    print(solver_state.gramian_diagonal)
