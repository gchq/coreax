Weights
========

.. automodule:: coreax.weights

Stein-kernel weights
--------------------

``KSDWeightsOptimiser`` minimises the uncorrected kernel Stein discrepancy while
requiring weights to sum to one. It permits negative weights. The supplied Stein
kernel's score function defines the target distribution, so the original dataset is
accepted for compatibility but is not used to estimate a kernel mean::

    import jax.numpy as jnp
    from coreax import Data, SquaredExponentialKernel, SteinKernel
    from coreax.weights import KSDWeightsOptimiser

    original_data = Data(jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]]))
    coreset_points = original_data[jnp.array([0, 2, 4])]
    kernel = SteinKernel(SquaredExponentialKernel(), score_function=lambda x: -x)
    optimiser = KSDWeightsOptimiser(kernel)
    weights = optimiser.solve(original_data, coreset_points, epsilon=1e-6)

Here the score ``-x`` corresponds to a standard normal target. The diagonal penalty
``epsilon`` must be finite and non-negative. With ``epsilon=0``, singular Gram
matrices and repeated points are supported by selecting the minimum-norm solution.
Very small eigenvalues are discarded using a dtype-dependent relative tolerance;
use double precision for especially ill-conditioned systems. The optimiser uses a
dense eigendecomposition, so it is intended for coresets that fit in memory.
