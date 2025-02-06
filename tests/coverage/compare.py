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

"""Script that compares coverage data to previously-recorded data."""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

# A good amount of this code is duplicated in the performance comparison script, but we
# ignore this so that both scripts can be standalone.
# pylint: disable=duplicate-code
COVERAGE_FILENAME_REGEX = re.compile(
    r"^coverage"
    r"-(\d{4})-(\d{2})-(\d{2})"
    r"--(\d{2})-(\d{2})-(\d{2})"
    r"--([0-9a-f]{40})"
    r"--v(\d+)\.json$"
)

# Set tolerances for reduction in coverage percentage before test fails
ABSOLUTE_TOLERANCE = 0
RELATIVE_TOLERANCE = 0

# Increment this if any changes are made to the storage format! Remember to also
# increment the corresponding value in the `coverage.yml` workflow file.
CURRENT_DATA_VERSION = 1


def parse_args() -> tuple[float, Path]:
    """
    Parse command-line arguments.

    :return: Tuple of (coverage total, directory of reference data, hash of commit,
        commit subject file)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_total", help="New total coverage as a percentage.")
    parser.add_argument(
        "reference_directory", help="Directory containing historic coverage data."
    )
    args = parser.parse_args()
    return float(args.coverage_total), Path(args.reference_directory)


def date_from_filename(path: Path) -> Optional[tuple[datetime.datetime, str]]:
    """
    Extract the date from a coverage data file name.

    The current filename format is::

        coverage-YYYY-MM-DD--HH-MM-SS--[40-char git commit hash]--vX.json

    where `YYYY-MM-DD--HH-MM-SS` is the year, month, day, hour, minute, and second
    that the file was created, the commit hash is for the commit the tests were run
    against, and the vX at the end is a version number specifier, in case we need to
    change the format at a later date.

    :param path: The path to the coverage data file. Only the filename component
        (`path.name`) is used.
    :return: Tuple (date_time, commit_hash) if the filename matched the expected format,
        or :data:`None` if it did not match.
    """
    filename = path.name
    match = COVERAGE_FILENAME_REGEX.fullmatch(filename)
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


def get_most_recent_coverage_total(reference_directory: Path) -> float:
    """
    Get the most recent saved coverage total in the given directory.

    Uses :py:func:`date_from_filename` to extract the date, time and commit hash from
    each file name. The date and time are stored with an accuracy of one second, so two
    data files sharing a time are extremely unlikely but not impossible. In case two
    data files have the exact same time recorded, the latest file is selected based on
    the lexicographic ordering of the associated commit hashes.

    :param reference_directory: Directory containing historic coverage data
    :return: Total coverage extracted from the most recent coverage file, or 0 if no
        file found
    """
    files: dict[Path, tuple[datetime.datetime, str]] = {}
    for filename in reference_directory.iterdir():
        date_tuple = date_from_filename(filename)
        if date_tuple is not None:
            files[filename] = date_tuple

    if not files:
        print("**WARNING: No historic coverage data found.**")
        return 0

    most_recent_file = max(files.keys(), key=files.__getitem__)

    with open(most_recent_file, "r", encoding="utf8") as f:
        coverage_dict = json.load(f)

    return coverage_dict["total"]


def check_significant_difference(
    current_coverage: float, historic_coverage: float
) -> bool:
    """
    Check if the coverage has reduced significantly.

    Print console messages with coverage change. Display full precision for differences
    but round absolute percentages to two decimal places.

    :param current_coverage: Current coverage total
    :param historic_coverage: Most recent historic coverage total
    :return: Is there a significant reduction in coverage?
    """
    absolute_loss = historic_coverage - current_coverage
    relative_loss = absolute_loss / historic_coverage if historic_coverage > 0 else 0

    if absolute_loss == 0:
        print(f"PASS: Coverage remained the same at {current_coverage:.2f}%.")
        return False
    if absolute_loss < 0:
        print(
            f"PASS: Coverage increased by {-absolute_loss}% from "
            f"{historic_coverage:.2f}% to "
            f"{current_coverage:.2f}%."
        )
        return False

    exceed_absolute = absolute_loss > ABSOLUTE_TOLERANCE
    exceed_relative = relative_loss > RELATIVE_TOLERANCE

    if exceed_absolute or exceed_relative:
        if exceed_absolute and exceed_relative:
            tolerance_msg = "absolute and relative tolerances"
        elif exceed_absolute:
            tolerance_msg = "absolute tolerance"
        else:
            tolerance_msg = "relative tolerance"
        print(
            f"FAIL: Coverage reduced by {absolute_loss}% from {historic_coverage:.2f}% "
            f"to {current_coverage:.2f}%, exceeding {tolerance_msg}."
        )
        return True

    print(
        f"PASS: Coverage reduced slightly by {absolute_loss}% from "
        f"{historic_coverage:.2f}% to {current_coverage:.2f}%."
    )
    return False


def main() -> None:  # noqa: C901
    """Run the command-line script."""
    current_coverage, reference_directory = parse_args()
    historic_coverage = get_most_recent_coverage_total(reference_directory)
    if check_significant_difference(current_coverage, historic_coverage):
        # Return code 2 to match failure behaviour of coverage
        sys.exit(2)


if __name__ == "__main__":
    main()
# pylint: enable=duplicate-code
