"""Script to run the performance tests defined in `cases`."""

from cases.basic_coresets import setup_herding, setup_rpc, setup_stein

from coreax.util import speed_comparison_test

if __name__ == "__main__":
    speed_comparison_test(
        [
            setup_herding(),
            setup_rpc(),
            setup_stein(),
        ],
        10,
        log_results=True,
        check_hash=False,
    )
