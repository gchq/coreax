from coreax import KernelHerding, SizeReduce, SquaredExponentialKernel
from coreax.weights import MMD as MMDWeightsOptimiser

# Define a kernel
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Define a weights optimiser to learn optimal weights for the coreset after creation
weights_optimiser = MMDWeightsOptimiser(kernel=kernel)

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key, kernel=kernel, weights_optimiser=weights_optimiser
)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))

# Determine optimal weights for the coreset
herding_weights = herding_object.solve_weights()
