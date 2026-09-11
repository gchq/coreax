Solvers
========

.. automodule:: coreax.solvers
    :no-private-members:
    :no-undoc-members:

Gradient flow
-------------

``GradientFlow`` moves particle locations to approximate a weighted target measure.
It returns a ``PseudoCoreset`` whose points need not occur in the original dataset.
Initial particles are sampled with replacement, using the target weights.

.. code-block:: python

    import jax.numpy as jnp
    import jax.random as jr
    import optax
    from coreax.data import Data
    from coreax.kernels import SquaredExponentialKernel
    from coreax.solvers import GradientFlow

    data = Data(jnp.linspace(-2, 2, 100))
    solver = GradientFlow(
        coreset_size=10,
        random_key=jr.key(0),
        kernel=SquaredExponentialKernel(),
        optimiser=optax.adam(learning_rate=0.05),
        num_iterations=100,
        noise_scale=optax.exponential_decay(0.05, 25, 0.5),
    )
    coreset, state = solver.reduce(data)
    coreset, state = solver.refine(coreset, state)

The default optimiser is ``optax.sgd(0.1)``, which reproduces the fixed-step update
from Arbel et al. The optimiser state and cumulative iteration are stored in
``GradientFlowState``, so adaptive optimisers and scheduled noise continue correctly
across repeated calls to ``refine``.

``refine`` also accepts an existing coresubset or pseudo-coreset with exactly the
configured number of points. Their weights and original data are preserved.
Passing the returned state continues the optimiser, noise schedule and random sequence across calls.

The implementation uses Euler steps for the MMD flow of :cite:`arbel2019maximum`,
with a constant noise scale. Noise is applied to the locations at which the witness
gradient is evaluated. With zero noise this is the deterministic particle flow.
Use a smooth symmetric kernel and a step size suitable for the data and kernel;
large steps can increase the MMD. This does not guarantee a global optimum.
All pairwise particle-particle and particle-target gradients are evaluated on each
step, so runtime and memory grow with both dataset and coreset size.
