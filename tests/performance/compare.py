"""Script that compares performance data to previously-recorded data."""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Tuple

from scipy.stats import ttest_ind_from_stats

from coreax.util import format_number

PERFORMANCE_FILENAME_REGEX = re.compile(
    r"^performance"
    r"-(\d{4})-(\d{2})-(\d{2})"
    r"--(\d{2})-(\d{2})-(\d{2})"
    r"--([0-9a-f]{40})\.json$"
)
P_VALUE_THRESHOLD_UNCORRECTED = 0.05


def parse_args() -> Tuple[Path, Path]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "performance_file", help="The file containing the most recent performance test."
    )
    parser.add_argument(
        "reference_directory",
        help="The directory containing historic performance data.",
    )
    args = parser.parse_args()
    return Path(args.performance_file), Path(args.reference_directory)


def date_from_filename(path: Path) -> Tuple[datetime.datetime, str]:
    """Extract the date from a performance data file name."""
    filename = path.name
    match = PERFORMANCE_FILENAME_REGEX.fullmatch(filename)
    if not match:
        raise ValueError(f"Could not parse date from filename: {filename}")

    year, month, day, hour, minute, second, git_hash = match.groups()
    return datetime.datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        tzinfo=datetime.timezone.utc,
    ), git_hash


def get_most_recent_historic_data(reference_directory: Path) -> Path:
    """Get the most recent saved performance data in the given directory."""
    return max(reference_directory.iterdir(), key=date_from_filename)


def format_run_time(times: dict[str, float]) -> str:
    """Format the performance data for a specific test in a human readable form."""
    return (
        f"compilation {format_number(times['compilation_mean'])} "
        f"± {format_number(times['compilation_std'])}; "
        f"execution {format_number(times['execution_mean'])} "
        f"± {format_number(times['execution_std'])}"
    )


def main() -> None:
    """Run the command-line script."""
    performance_file, reference_directory = parse_args()
    with open(performance_file, "r", encoding="utf8") as f:
        current_performance: dict = json.load(f)

    with open(
        get_most_recent_historic_data(reference_directory), "r", encoding="utf8"
    ) as f:
        historic_performance: dict = json.load(f)

    missing = historic_performance.keys() - current_performance.keys()
    new = current_performance.keys() - historic_performance.keys()
    matched = set(historic_performance.keys()).intersection(current_performance.keys())

    print("## Performance review")

    if missing:
        print("### (!!) Missing performance tests")
        for name in missing:
            print(f"- `{name}`")
    if new:
        print("### New performance tests")
        for name in new:
            print(f"- `{name}`: {format_run_time(current_performance[name])}")

    # we're doing len(matched)*2 tests, so we need to correct the p-value accordingly
    p_value_threshold = P_VALUE_THRESHOLD_UNCORRECTED / (len(matched) * 2)
    significant = []
    for name in matched:
        t_compilation = ttest_ind_from_stats(
            mean1=current_performance[name]["compilation_mean"],
            std1=current_performance[name]["compilation_std"],
            nobs1=current_performance[name]["num_runs"],
            mean2=historic_performance[name]["compilation_mean"],
            std2=historic_performance[name]["compilation_std"],
            nobs2=historic_performance[name]["num_runs"],
            equal_var=False,
        )
        t_execution = ttest_ind_from_stats(
            mean1=current_performance[name]["execution_mean"],
            std1=current_performance[name]["execution_std"],
            nobs1=current_performance[name]["num_runs"],
            mean2=historic_performance[name]["execution_mean"],
            std2=historic_performance[name]["execution_std"],
            nobs2=historic_performance[name]["num_runs"],
            equal_var=False,
        )
        if (
            t_execution.pvalue < p_value_threshold
            or t_compilation.pvalue < p_value_threshold
        ):
            significant.append(name)

    if significant:
        print("### Significant changes")
        for name in significant:
            print(f"- `{name}`:")
            print(f"  - OLD: {format_run_time(historic_performance[name])}")
            print(f"  - NEW: {format_run_time(current_performance[name])}")

    if not missing and not new and not significant:
        print("No significant changes to performance.")


if __name__ == "__main__":
    main()
