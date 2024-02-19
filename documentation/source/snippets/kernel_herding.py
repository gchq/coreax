import jax.random
import numpy as np
from sklearn.datasets import make_blobs

from coreax import ArrayData, KernelHerding, SizeReduce, SquaredExponentialKernel
from coreax.kernel import median_heuristic

# Generate some data
num_data_points = 10_000
num_features = 2
num_cluster_centers = 6
random_seed = 1989
x, _ = make_blobs(
    num_data_points,
    n_features=num_features,
    centers=num_cluster_centers,
    random_state=random_seed,
)

# Request 100 coreset points
coreset_size = 100

# Setup the original data object
data = ArrayData.load(x)

# Set the bandwidth parameter of the kernel using a median heuristic derived from
# at most 1000 random samples in the data.
num_samples_length_scale = min(num_data_points, 1_000)
generator = np.random.default_rng(random_seed)
idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
length_scale = median_heuristic(x[idx])

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_key = jax.random.key(random_seed)
herding_object = KernelHerding(
    herding_key, kernel=SquaredExponentialKernel(length_scale=length_scale)
)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))

# The herding object now has the coreset, and the indices of the original data
# that makeup the coreset as populated attributes
print(herding_object.coreset)
print(herding_object.coreset_indices)
