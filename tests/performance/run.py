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

"""Script to run the performance tests defined in `cases`."""

import argparse
import json

from cases.basic_coresets import setup_herding, setup_rpc, setup_stein
from cases.normaliser import setup_normaliser

from coreax.util import speed_comparison_test

NUM_RUNS = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", default=None)
    args = parser.parse_args()

    normaliser_results, _ = speed_comparison_test(
        function_setups=[setup_normaliser()],
        num_runs=NUM_RUNS,
        log_results=True,
    )
    normaliser_compilation_time = normaliser_results[0][0][0].item()
    normaliser_execution_time = normaliser_results[0][0][1].item()

    function_setups = [
        setup_herding(),
        setup_rpc(),
        setup_stein(),
    ]

    results, _ = speed_comparison_test(
        function_setups=function_setups,
        num_runs=NUM_RUNS,
        log_results=True,
        normalisation=(normaliser_compilation_time, normaliser_execution_time),
    )

    if args.output_file is not None:
        # we have to do a little work here since JAX arrays are not JSON serializable
        dict_results = {
            "normalisation": {
                "compilation": normaliser_compilation_time,
                "execution": normaliser_execution_time,
            },
            "results": {
                setup.name: {
                    "compilation_mean": times[0][0].item(),
                    "execution_mean": times[0][1].item(),
                    "compilation_std": times[1][0].item(),
                    "execution_std": times[1][1].item(),
                    "num_runs": NUM_RUNS,
                }
                for setup, times in zip(function_setups, results, strict=False)
            },
        }
        with open(args.output_file, "w", encoding="utf8") as f:
            json.dump(dict_results, f)
