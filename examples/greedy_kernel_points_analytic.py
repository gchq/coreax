# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example use of greedy kernel points.

This example implements the analytic example of greedy kernel points in the examples
section of the documentation. See the documentation for the derivation.
"""

import jax
import jax.numpy as jnp

from coreax.coreset import Coresubset
from coreax.data import SupervisedData
from coreax.kernels import LinearKernel
from coreax.solvers import GreedyKernelPoints


def main() -> Coresubset[SupervisedData]:
    """
    Run the :class:`~coreax.solvers.GreedyKernelPoints` example.

    :return: Coresubset, also printed to console.
    """
    # Create supervised data
    in_data = SupervisedData(
        data=jnp.array([[1, 0], [0, 1], [2, 1]]), supervision=jnp.array([0, 1, 5])
    )

    # Set up solver
    random_seed = 1_989
    random_key = jax.random.key(random_seed)
    solver = GreedyKernelPoints(
        coreset_size=2,
        random_key=random_key,
        feature_kernel=LinearKernel(output_scale=1, constant=0),
    )

    # Calculate coreset
    coreset, _ = solver.reduce(in_data)

    # Display coreset, noting that indices and supervision need to be squeezed to
    # one-dimensional vectors
    print(f"Coresubset has indices {jnp.squeeze(coreset.indices.data)}.")
    print(
        f"Coresubset is\n{coreset.points.data}\nwith supervision "
        f"{jnp.squeeze(coreset.points.supervision)}."
    )

    return coreset


if __name__ == "__main__":
    main()
