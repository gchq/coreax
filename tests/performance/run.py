"""Script to run the performance tests defined in `cases`."""

import argparse
import json

from cases.basic_coresets import setup_herding, setup_rpc, setup_stein

from coreax.util import speed_comparison_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", default=None)
    args = parser.parse_args()

    function_setups = [
        setup_herding(),
        setup_rpc(),
        setup_stein(),
    ]

    NUM_RUNS = 10

    results, _ = speed_comparison_test(
        function_setups=function_setups,
        num_runs=NUM_RUNS,
        log_results=True,
        check_hash=False,
    )

    if args.output_file is not None:
        # we have to do a little work here since JAX arrays are not JSON serializable
        dict_results = {
            setup.name: {
                "compilation_mean": float(times[0][0]),
                "execution_mean": float(times[0][1]),
                "compilation_std": float(times[1][0]),
                "execution_std": float(times[1][1]),
                "num_runs": NUM_RUNS,
            }
            for setup, times in zip(function_setups, results)
        }
        with open(args.output_file, "w", encoding="utf8") as f:
            json.dump(dict_results, f)
