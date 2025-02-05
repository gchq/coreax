Analytical example with RPCholesky
==================================

Step-by-step usage of the RPCholesky algorithm (Algorithm 1 in
:cite:`chen2023randomly`) on a small example with 3 data points in 2 dimensions and a
coreset of size 2, i.e., :math:`N=3, m=2`.

In this example, we have the following data:

.. math::
    X = \begin{pmatrix}
        0.5 & 0.2 \\
        0.4 & 0.6 \\
        0.8 & 0.3
    \end{pmatrix}

We choose a ``SquaredExponentialKernel`` with ``length_scale`` of
:math:`\frac{1}{\sqrt{2}}`: for two points :math:`x, y \in X`, :math:`k(x, y) =
e^{-||x - y||^2}`. We now compute the Gram matrix, :math:`A`, of the dataset
:math:`X` with respect to the kernel :math:`k` as :math:`A_{ij} = k(X_i, X_j)`:

.. math::
    A = \begin{pmatrix}
        1.0 & 0.84366477 & 0.90483737 \\
        0.84366477 & 1.0 & 0.7788007 \\
        0.90483737 & 0.7788007 & 1.0
    \end{pmatrix}

Note that, in practice, we do not need to precompute the full Gram matrix, the algorithm
only needs to evaluate the pivot column at each iteration.

To apply the RPCholesky algorithm, we first initialise the *residual diagonal*
:math:`d = \text{diag}(A)` and the *approximation matrix* :math:`F = \mathbf{0}_{N
\times m}`, where :math:`N = 3, m = 2` in our case.

We now build a coreset iteratively by applying the following steps at each iteration i:
    - Sample a datapoint index (called a pivot) proportional to :math:`d`
    - Compute/extract column :math:`g` corresponding to the pivot index from :math:`A`
    - Remove the overlap with previously selected columns from :math:`g`
    - Normalize the column and add it to the approximation matrix :math:`F`
    - Update the residual diagonal: :math:`d = d - |F[:,i]|^2`

For the first iteration (i=0):

1. We sample a pivot point proportional to their value on the diagonal. Since
:math:`d` is initialised as :math:`(1, 1, 1)` in our case, all choices are equally
likely, so let us suppose we choose the pivot with index = 2.

2. We now compute :math:`g`, the column at index 2, as:

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

1. We again sample a pivot point proportional to their value on the updated residual
diagonal, :math:`d`. Let's suppose we choose the most likely pivot here (index=1).

2. We now compute g, the column at index 1, as:

.. math::
    g = \begin{pmatrix}
    0.84366477 \\
    1.0 \\
    0.7788007
    \end{pmatrix}

3. Remove overlap with previously chosen columns:

.. math::
    g = g - F[:, 0] F[1, 0]^T = \begin{pmatrix}
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
    S = \{2, 1\} \, .

This completes the coreset of size :math:`m = 2`. We can also use the :math:`F` to
compute an approximation to the original Gram matrix:

.. math::

    F \cdot F^T = \begin{pmatrix}
    0.86781846 & 0.84366477 & 0.90483737 \\
    0.84366477 & 1.0 & 0.7788007 \\
    0.90483737 & 0.7788007 & 1.0
    \end{pmatrix}

Note that we have recovered the original matrix except for :math:`A_{00}`, which was not
covered by any of the chosen pivots.

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
    print(coreset.points.data)  # The data-points in the coreset
    print("Residual diagonal:")
    print(solver_state.gramian_diagonal)
