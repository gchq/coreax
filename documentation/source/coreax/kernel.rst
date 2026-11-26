Kernels
========

.. automodule:: coreax.kernels

Per-feature length scales
-------------------------

``AnisotropicKernel`` wraps a scalar-valued kernel and scales its inputs separately
for each feature. A scalar scale applies to all features. A vector must have one
positive, finite value per feature; a one-element vector represents one feature,
not a scalar scale for a multidimensional input.

.. code-block:: python

    import jax.numpy as jnp
    from coreax.kernels import AnisotropicKernel, SquaredExponentialKernel

    kernel = AnisotropicKernel(
        SquaredExponentialKernel(), length_scale=jnp.array([0.5, 2.0])
    )
    points = jnp.array([[0.0, 1.0], [1.0, 3.0]])
    matrix = kernel.compute(points, points)

The wrapper evaluates ``base_kernel(x / length_scale, y / length_scale)``.
Leave the base kernel's own length scale at one when the wrapper should define
all feature scales. Existing kernel constructors remain unchanged. Kernel
compositions, weighted metric calculations, JIT compilation and differentiation
with respect to the scales use the normal kernel interface.

Input derivatives include the coordinate scaling factors. The mixed derivative
trace uses the inherited automatic differentiation implementation. For Stein
kernels, wrap the base kernel before constructing ``SteinKernel`` so the target
score remains in the original coordinates.
