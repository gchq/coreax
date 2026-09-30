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
Passing the returned state continues the optimiser, noise schedule and random sequence
across calls.

The implementation uses Euler steps for the MMD flow of :cite:`arbel2019maximum`,
with a configurable non-negative noise scale, either constant or scheduled. Noise is
applied to the locations at which the witness gradient is evaluated. With zero noise this
is the deterministic particle flow.
Use a smooth symmetric kernel and a step size suitable for the data and kernel;
large steps can increase the MMD. This does not guarantee a global optimum.
All pairwise particle-particle and particle-target gradients are evaluated on each
step, so runtime and memory grow with both dataset and coreset size.

Data Twinning
-------------

``DataTwinning`` implements the nearest-neighbour grouping in Algorithm 1 of
:cite:`vakayil2022twinning`. A group size ``thinning_factor`` keeps one row per
group, giving ``ceil(N / thinning_factor)`` rows from ``N`` inputs. The returned
coresubset retains the original data and, for supervised inputs, their responses.
Both features and responses are standardised before computing distances.

.. code-block:: python

    import jax.numpy as jnp
    from coreax.data import Data
    from coreax.solvers import DataTwinning

    dataset = Data(jnp.arange(20).reshape(10, 2))
    coreset, _ = DataTwinning(thinning_factor=3, start_index=0).reduce(dataset)

The default starting point is farthest from the standardised centroid. Ties use
original row order. Only finite, real data with uniform positive weights are
supported. No random state or kernel bandwidth is required.

This implementation uses exhaustive array-based neighbour searches to support
JIT compilation. It avoids a full pairwise distance matrix, but does not reproduce
the tree-based implementation's near-linear average-case runtime. For large
inputs, account for the cost of a full distance search and sort per selected row.
