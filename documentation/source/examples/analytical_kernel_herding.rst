Analytical example with kernel herding
======================================

Step-by-step usage of kernel herding on an analytical example, enforcing a unique
coreset.

In this example, we have data of:

.. math::
    x = \begin{pmatrix}
        0.3 & 0.25 \\
        0.4 & 0.2 \\
        0.5 & 0.125
    \end{pmatrix}

and choose a ``length_scale`` of :math:`\frac{1}{\sqrt{2}}` to simplify computations
with the ``SquaredExponentialKernel``, in particular it becomes:

.. math::
    k(x, y) = e^{-||x - y||^2}.

Kernel herding should do as follows:
    - Compute the Gramian row mean, that is for each data-point :math:`x` and all other
      data-points :math:`x'`, :math:`\frac{1}{N} \sum_{x'} k(x, x')` where we have
      :math:`N` data-points in total.
    - Select the first coreset point :math:`x_{1}` as the data-point where the
      Gramian row mean is highest.
    - Compute all future coreset points as
      :math:`x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)`
      where we currently have :math:`T` points in the coreset.

We ask for a coreset of size 2 in this example. With an empty coreset, we first
compute :math:`\mathbb{E}[k(x, x')]` as:

.. math::
    \mathbb{E}[k(x, x')] = \frac{1}{3} \cdot \begin{pmatrix}
        k([0.3, 0.25]', [0.3, 0.25]') + k([0.3, 0.25]', [0.4, 0.2]') + k([0.3, 0.25]', [0.5, 0.125]') \\
        k([0.4, 0.2]', [0.3, 0.25]') + k([0.4, 0.2]', [0.4, 0.2]') + k([0.4, 0.2]', [0.5, 0.125]') \\
        k([0.5, 0.125]', [0.3, 0.25]') + k([0.5, 0.125]', [0.4, 0.2]') + k([0.5, 0.125]', [0.5, 0.125]')
    \end{pmatrix}

resulting in:

.. math::
    \mathbb{E}[k(x, x')] = \begin{pmatrix}
        0.9778238600172561 \\
        0.9906914124997632 \\
        0.9767967388544317
    \end{pmatrix}

The largest value in this array is 0.9906914124997632, so we expect the first
coreset point to be [0.4, 0.2], that is the data-point at index 1 in the
dataset. At this point we have ``coreset_indices`` as [1, ?].

We then compute the penalty update term
:math:`\frac{1}{T+1}\sum_{t=1}^T k(x, x_t)` with :math:`T = 1`:

.. math::
    \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \frac{1}{2} \cdot \begin{pmatrix}
        k([0.3, 0.25]', [0.4, 0.2]') \\
        k([0.4, 0.2]', [0.4, 0.2]') \\
        k([0.5, 0.125]', [0.4, 0.2]')
    \end{pmatrix}

which evaluates to:

.. math::
    \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
        0.4937889002469407 \\
        0.5 \\
        0.4922482185027042
    \end{pmatrix}

We now select the data-point that maximises
:math:`\mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t)`, which
evaluates to:

.. math::
    \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
        0.9778238600172561 - 0.4937889002469407 \\
        0.9906914124997632 - 0.5 \\
        0.9767967388544317 - 0.4922482185027042
    \end{pmatrix}

giving a final result of:

.. math::
    \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
        0.4840349597703154 \\
        0.4906914124997632 \\
        0.4845485203517275
    \end{pmatrix}

The largest value in this array is at index 1, which would be to again choose
the point [0.4, 0.2] for the coreset. However, in this example we enforce the
coreset to be unique, that is not to select the same data-point twice, which
means we should take the next highest value in the above result to include in
our coreset. This happens to be 0.4845485203517275, the data-point at index 2.
This means our final ``coreset_indices`` should be [1, 2].

Finally, the solver state tracks variables we need not compute repeatedly. In
the case of kernel herding, we don't need to recompute
:math:`\mathbb{E}[k(x, x')]` at every single step - so the solver state from the
coreset reduce method should be set to:

.. math::
    \mathbb{E}[k(x, x')] = \begin{pmatrix}
        0.9778238600172561 \\
        0.9906914124997632 \\
        0.9767967388544317
    \end{pmatrix}


This example would be run in coreax using:

.. code-block::

    from coreax import Data, SquaredExponentialKernel, KernelHerding
    import equinox as eqx

    # Define the data
    coreset_size = 2
    length_scale = 1.0 / jnp.sqrt(2)
    x = jnp.array([
        [0.3, 0.25],
        [0.4, 0.2],
        [0.5, 0.125],
    ])

    # Define a kernel
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    # Generate the coreset, using equinox to JIT compile the code and speed up
    # generation for larger datasets
    data = Data(x)
    solver = KernelHerding(coreset_size=coreset_size, kernel=kernel, unique=True)
    coreset, solver_state = eqx.filter_jit(solver.reduce)(data)

    # Inspect results
    print(coreset.unweighted_indices)  # The coreset_indices
    print(coreset.points.data)  # The data-points in the coreset
    print(solver_state.gramian_row_mean)  # The stored gramian_row_mean

Coreax also supports weighted data. If we have the same data as described above, but
weights of:

.. math::
    w = \begin{pmatrix}
        0.8 \\
        0.1 \\
        0.1
    \end{pmatrix}

we would expect a different resulting coreset. The computation of the gramian
row mean, :math:`\mathbb{E}[k(x, x')]`, becomes:

.. math::
    \mathbb{E}[k(x, x')] = \begin{pmatrix}
        0.8 \cdot k([0.3, 0.25]', [0.3, 0.25]') + 0.1 \cdot k([0.3, 0.25]', [0.4, 0.2]') + 0.1 \cdot k([0.3, 0.25]', [0.5, 0.125]') \\
        0.8 \cdot  k([0.4, 0.2]', [0.3, 0.25]') + 0.1 \cdot k([0.4, 0.2]', [0.4, 0.2]') + 0.1 \cdot k([0.4, 0.2]', [0.5, 0.125]') \\
        0.8 \cdot  k([0.5, 0.125]', [0.3, 0.25]') + 0.1 \cdot k([0.5, 0.125]', [0.4, 0.2]') + 0.1 \cdot k([0.5, 0.125]', [0.5, 0.125]')
    \end{pmatrix}

resulting in:

.. math::
    \mathbb{E}[k(x, x')] = \begin{pmatrix}
        0.9933471580051769 \\
        0.988511884095646 \\
        0.9551646673468503
    \end{pmatrix}

The largest value in this array is 0.9933471580051769, so we expect the first coreset
point to be [0.3  0.25], that is the data-point at index 0 in the dataset. At this point
we have ``coreset_indices`` as [0, ?].

We then compute the penalty update term
:math:`\frac{1}{T+1}\sum_{t=1}^T k(x, x_t)` with :math:`T = 1` and get:

.. math::
    \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
        0.5 \\
        0.4937889002469407 \\
        0.4729468897789434
    \end{pmatrix}

Finally, we select the next coreset point to maximise:

.. math::
    \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
        0.4933471580051769 \\
        0.49472298384870533 \\
        0.48221777756790696
    \end{pmatrix}

which means our final ``coreset_indices`` should be [0, 1]. In coreax, this example
would be run as:

.. code-block::

    from coreax import Data, SquaredExponentialKernel, KernelHerding
    import equinox as eqx

    # Define the data
    coreset_size = 2
    length_scale = 1.0 / jnp.sqrt(2)
    x = jnp.array([
        [0.3, 0.25],
        [0.4, 0.2],
        [0.5, 0.125],
    ])
    weights = jnp.array([0.8, 0.1, 0.1])

    # Define a kernel
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    # Generate the coreset, using equinox to JIT compile the code and speed up
    # generation for larger datasets
    data = Data(x, weights=weights)
    solver = KernelHerding(coreset_size=coreset_size, kernel=kernel, unique=True)
    coreset, solver_state = eqx.filter_jit(solver.reduce)(data)

    # Inspect results
    print(coreset.unweighted_indices)  # The coreset_indices
    print(coreset.points.data)  # The data-points in the coreset
    print(solver_state.gramian_row_mean)  # The stored gramian_row_mean
