"""Script to run the performance tests defined in `cases`."""

import argparse
import json

from cases.basic_coresets import setup_herding, setup_rpc, setup_stein

from coreax.util import speed_comparison_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", default=None)
    args = parser.parse_args()

    _, timings = speed_comparison_test(
        [
            setup_herding(),
            setup_rpc(),
            setup_stein(),
        ],
        10,
        log_results=True,
        check_hash=False,
    )

    if args.out_file is not None:
        with open(args.out_file, "w", encoding="utf8") as f:
            json.dump(timings, f)
