from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import MapReduce

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key,
    kernel=SquaredExponentialKernel(length_scale=length_scale),
)
herding_object.fit(
    original_data=data, strategy=MapReduce(coreset_size=coreset_size, leaf_size=200)
)
