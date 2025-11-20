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
from typing import TypedDict

from scipy.stats import ttest_ind_from_stats

from coreax.util import format_time

# A good amount of this code is duplicated in the coverage comparison script, but we
# ignore this so that both scripts can be standalone.
# pylint: disable=duplicate-code

PERFORMANCE_FILENAME_REGEX = re.compile(
    r"^performance"
    r"-(\d{4})-(\d{2})-(\d{2})"
    r"--(\d{2})-(\d{2})-(\d{2})"
    r"--([0-9a-f]{40})"
    r"--v(\d+)\.json$"
)
P_VALUE_THRESHOLD_UNCORRECTED = 0.05
RATIO_THRESHOLD = 0.1  # 10% change minimum

# Increment this if any changes are made to the storage format! Remember to also
# increment the corresponding value in the `performance.yml` workflow file.
CURRENT_DATA_VERSION = 1


class NormalisationData(TypedDict):
    """Type hint for the normalisation data stored with the performance data."""

    compilation: float
    execution: float


class SinglePerformanceTestData(TypedDict):
    """Type hint for the data representing a single performance test."""

    compilation_mean: float
    execution_mean: float
    compilation_std: float
    execution_std: float
    num_runs: int


class FullPerformanceData(TypedDict):
    """Type hint for the full performance test data file."""

    results: dict[str, SinglePerformanceTestData]
    normalisation: NormalisationData


def parse_args() -> tuple[Path, Path, str, Path]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "performance_file", help="The file containing the most recent performance test."
    )
    parser.add_argument(
        "reference_directory",
        help="The directory containing historic performance data.",
    )
    parser.add_argument(
        "--commit-short-hash", help="The abbreviated hash of the commit."
    )
    parser.add_argument(
        "--commit-subject-file", help="A file containing the commit's subject line."
    )
    args = parser.parse_args()
    return (
        Path(args.performance_file),
        Path(args.reference_directory),
        args.commit_short_hash,
        Path(args.commit_subject_file),
    )


def date_from_filename(path: Path) -> tuple[datetime.datetime, str] | None:
    """
    Extract the date from a performance data file name.

    The current filename format is::

        performance-YYYY-MM-DD--HH-MM-SS--[40-char git commit hash]--vX.json

    where `YYYY-MM-DD--HH-MM-SS` is the year, month, day, hour, minute, and second
    that the file was created, the commit hash is for the commit the tests were run
    against, and the vX at the end is a version number specifier, in case we need to
    change the format at a later date.

    :param path: The path to the performance data file. Only the filename component
        (`path.name`) is used.
    :return: Tuple (date_time, commit_hash) if the filename matched the expected format,
        or :data:`None` if it did not match.
    """
    filename = path.name
    match = PERFORMANCE_FILENAME_REGEX.fullmatch(filename)
    if not match:
        return None

    year, month, day, hour, minute, second, git_hash, spec_version = match.groups()
    if int(spec_version) != CURRENT_DATA_VERSION:
        # But in future, we could try and extract at least some data?
        return None

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
) -> FullPerformanceData:
    """
    Get the most recent saved performance data in the given directory.

    Uses :py:func:`date_from_filename()` to extract the date, time and commit hash from
    each file name. The date and time are stored with an accuracy of one second, so two
    data files sharing a time are extremely unlikely but not impossible. In case two
    data files have the exact same time recorded, the latest file is selected based on
    the lexicographic ordering of the associated commit hashes.

    :param reference_directory: The directory containing historic performance data
    :return: Dictionary `(test_name -> dictionary(statistic -> value))` extracted from
        the most recent performance file
    """
    files: dict[Path, tuple[datetime.datetime, str]] = {}
    for filename in reference_directory.iterdir():
        date_tuple = date_from_filename(filename)
        if date_tuple is not None:
            files[filename] = date_tuple

    if not files:
        print("**WARNING: No historic performance data found.**")
        # the normalisation data here will cause errors if actually used, but that's
        # fine since there aren't any results present to normalise
        return {"results": {}, "normalisation": {"compilation": 0.0, "execution": 0.0}}

    most_recent_file = max(files.keys(), key=files.__getitem__)

    with open(most_recent_file, encoding="utf8") as f:
        return json.load(f)


def format_run_time(times: SinglePerformanceTestData) -> str:
    """
    Format the performance data for a specific test in a human-readable form.

    :param times: Dictionary `(statistic -> value)` describing a single performance test
    :return: A string describing the compilation and execution time of the test
    """
    return (
        f"compilation {times['compilation_mean']:.4g} units "
        f"± {times['compilation_std']:.4g} units; "
        f"execution {times['execution_mean']:.4g} units "
        f"± {times['execution_std']:.4g} units"
    )


def main() -> None:  # noqa: C901
    """Run the command-line script."""
    print("## Performance review")

    performance_file, reference_directory, commit_short_hash, commit_subject_file = (
        parse_args()
    )
    with open(commit_subject_file, encoding="utf8") as f:
        # escape any underscores to avoid formatting weirdness
        commit_subject = f.read().strip().replace("_", r"\_")

    print(f"###### Commit `{commit_short_hash}` - _{commit_subject}_")

    with open(performance_file, encoding="utf8") as f:
        current_performance_data: dict = json.load(f)
    historic_performance_data = get_most_recent_historic_data(reference_directory)

    current_performance = current_performance_data["results"]
    historic_performance = historic_performance_data["results"]

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
        # sort the changes by test name
        for name, messages in sorted(significant_changes, key=lambda tuple_: tuple_[0]):
            print(f"- `{name}`:")
            print(f"  - OLD: {format_run_time(historic_performance[name])}")
            print(f"  - NEW: {format_run_time(current_performance[name])}")
            for message in messages:
                print(f"  - {message}")

    if not missing and not new and not significant_changes:
        print("No significant changes to performance.")
    elif new or significant_changes:
        # add normalisation data
        normalisation = current_performance_data["normalisation"]
        print("---")
        print("_Normalisation values for new data:_  ")
        print(f"_Compilation: 1 unit = {format_time(normalisation['compilation'])}_  ")
        print(f"_Execution: 1 unit = {format_time(normalisation['execution'])}_")


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

    :param before: The "before" value for comparison
    :param after: The "after" value for comparison
    :return: The relative change, as described in the example above
    """
    if before < 0 or after < 0:
        raise ValueError((before, after))

    change = after - before
    try:
        return change / before
    except ZeroDivisionError:
        return float("inf")


def get_significant_differences(
    current_performance: dict[str, SinglePerformanceTestData],
    historic_performance: dict[str, SinglePerformanceTestData],
) -> list[tuple[str, list[str]]]:
    """
    Check if there are any significant differences in performance.

    Returns a list of (test_name, messages) tuples, to be used in the main part of
    the script.

    :param current_performance: Performance data representing the current state
        of performance
    :param historic_performance: The most recent historic performance data, representing
        the last known state of performance
    :return: List of tuples `(test_name, messages)` containing only those tests which
        had significant differences between their current and historic performance.
        `messages` is a list of human-readable strings that describes the significant
        differences for each test
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

# pylint: enable=duplicate-code
