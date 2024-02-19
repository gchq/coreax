import numpy as np

from coreax import (
    KernelDensityMatching,
    KernelHerding,
    SizeReduce,
    SquaredExponentialKernel,
    SteinKernel,
)

# Select indices to form a subset of data for learning score function
generator = np.random.default_rng(random_seed)
idx = generator.choice(len(data), subset_size, replace=False)
data_subset = data[idx, :]

# Learn a score function from the subset of the data, through a kernel density
# estimation applied to a subset of the data.
kernel_density_score_matcher = KernelDensityMatching(
    length_scale=length_scale, kde_data=data_subset
)
score_function = kernel_density_score_matcher.match()

# Define a kernel to use for herding
herding_kernel = SteinKernel(
    SquaredExponentialKernel(length_scale=length_scale),
    score_function=score_function,
)

# Compute a coreset using kernel herding with a Stein kernel
herding_object = KernelHerding(herding_key, kernel=herding_kernel)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))
