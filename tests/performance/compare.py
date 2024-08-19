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

"""Script that compares performance data to previously-recorded data."""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import ttest_ind_from_stats

PERFORMANCE_FILENAME_REGEX = re.compile(
    r"^performance"
    r"-(\d{4})-(\d{2})-(\d{2})"
    r"--(\d{2})-(\d{2})-(\d{2})"
    r"--([0-9a-f]{40})\.json$"
)
P_VALUE_THRESHOLD_UNCORRECTED = 0.05
RATIO_THRESHOLD = 0.1  # 10% change minimum


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


def date_from_filename(path: Path) -> Optional[Tuple[datetime.datetime, str]]:
    """Extract the date from a performance data file name."""
    filename = path.name
    match = PERFORMANCE_FILENAME_REGEX.fullmatch(filename)
    if not match:
        return None

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


def get_most_recent_historic_data(
    reference_directory: Path,
) -> Dict[str, Dict[str, float]]:
    """Get the most recent saved performance data in the given directory."""
    files: Dict[Path, Tuple[datetime.datetime, str]] = {}
    for filename in reference_directory.iterdir():
        date_tuple = date_from_filename(filename)
        if date_tuple is not None:
            files[filename] = date_tuple

    if not files:
        print("**WARNING: No historic performance data found.**")
        return {}

    most_recent_file = max(files.keys(), key=files.get)

    with open(most_recent_file, "r", encoding="utf8") as f:
        return json.load(f)


def format_run_time(times: Dict[str, float]) -> str:
    """Format the performance data for a specific test in a human-readable form."""
    return (
        f"compilation {times['compilation_mean']:.4g} units "
        f"± {times['compilation_std']:.4g} units; "
        f"execution {times['execution_mean']:.4g} units "
        f"± {times['execution_std']:.4g} units"
    )


def main() -> None:  # noqa: C901
    """Run the command-line script."""
    print("## Performance review")

    performance_file, reference_directory = parse_args()
    with open(performance_file, "r", encoding="utf8") as f:
        current_performance: dict = json.load(f)
    historic_performance = get_most_recent_historic_data(reference_directory)

    missing = historic_performance.keys() - current_performance.keys()
    new = current_performance.keys() - historic_performance.keys()

    if missing:
        print("### (!!) Missing performance tests")
        for name in missing:
            print(f"- `{name}`")
    if new:
        print("### New performance tests")
        for name in new:
            print(f"- `{name}`: {format_run_time(current_performance[name])}")

    significant_changes = get_significant_differences(
        current_performance, historic_performance
    )

    if significant_changes:
        print("### Statistically significant changes")
        for name, messages in significant_changes:
            print(f"- `{name}`:")
            print(f"  - OLD: {format_run_time(historic_performance[name])}")
            print(f"  - NEW: {format_run_time(current_performance[name])}")
            for message in messages:
                print(f"  - {message}")

    if not missing and not new and not significant_changes:
        print("No significant changes to performance.")


def relative_change(before: float, after: float) -> float:
    """
    Get the relative change between two values, to be interpreted as a percent change.

    Inputs must be non-negative.

    :Example:
        >>> relative_change(2, 3)  # 3 is 50% bigger than 2
        0.5
        >>> relative_change(4, 3)  # 3 is 25% less then 4
        -0.25
        >>> relative_change(1, 0)  # 0 is 100% less than 1
        -1
        >>> relative_change(0, 1)  # 1 is infinitely larger than 0
        inf
    """
    if before < 0 or after < 0:
        raise ValueError((before, after))

    change = after - before
    try:
        return change / before
    except ZeroDivisionError:
        return float("inf")


def get_significant_differences(
    current_performance: Dict[str, Dict[str, float]],
    historic_performance: Dict[str, Dict[str, float]],
) -> List[Tuple[str, List[str]]]:
    """
    Check if there are any significant differences in performance.

    Returns a list of (test_name, messages) tuples, to be used in the main part of
    the script.
    """
    matched = set(historic_performance.keys()).intersection(current_performance.keys())
    if not matched:
        return []
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
        compilation_change = relative_change(
            historic_performance[name]["compilation_mean"],
            current_performance[name]["compilation_mean"],
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
        execution_change = relative_change(
            historic_performance[name]["execution_mean"],
            current_performance[name]["execution_mean"],
        )

        is_compilation_significant = (
            t_compilation.pvalue < p_value_threshold
            and abs(compilation_change) > RATIO_THRESHOLD
        )
        is_execution_significant = (
            t_execution.pvalue < p_value_threshold
            and abs(execution_change) > RATIO_THRESHOLD
        )

        if is_compilation_significant or is_execution_significant:
            messages = []
            if is_compilation_significant:
                direction = (
                    "increase"
                    if current_performance[name]["compilation_mean"]
                    > historic_performance[name]["compilation_mean"]
                    else "decrease"
                )
                messages.append(
                    f"Significant {direction} in compilation time "
                    f"({compilation_change:.2%}, p={t_compilation.pvalue:.4g})"
                )
            if is_execution_significant:
                direction = (
                    "increase"
                    if current_performance[name]["execution_mean"]
                    > historic_performance[name]["execution_mean"]
                    else "decrease"
                )
                messages.append(
                    f"Significant {direction} in execution time "
                    f"({execution_change:.2%}, p={t_execution.pvalue:.4g})"
                )
            significant.append((name, messages))
    return significant


if __name__ == "__main__":
    main()
