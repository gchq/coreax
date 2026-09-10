Solvers
========

.. automodule:: coreax.solvers
    :no-private-members:
    :no-undoc-members:

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
