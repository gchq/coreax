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
Pre-commit hook to check for a copyright notice at the start of each file.

The copyright notice must start on the first line of the file (or the second line,
if the first line is a shebang), and must exactly match the template file specified.

Exits with code 1 if any checked file does not have the copyright notice, printing
the names of offending files to standard error. Exits silently with code 0 if all
checked files have the notice.

Optionally, with the `--unsafe-fix` argument, this script can attempt to
automatically insert the copyright notice at the top of files that are missing it.
The method that this script tries is naive, though, and may cause unexpected issues,
so be sure to check on any changes it makes! Using this argument when using this
script as a pre-commit hook is not recommended.

If you want to change the copyright notice template, a multi-file find-and-replace
tool is likely your best bet, rather than trying to use `--unsafe-fix`.
"""

import argparse
import sys

SHEBANG = "#!"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    parser.add_argument(
        "--template",
        default=".copyright-template",
        help=(
            "File containing the copyright notice that must be present at the top of "
            "each file checked."
        ),
    )
    parser.add_argument(
        "--unsafe-fix",
        action="store_true",
        help=(
            "If set, attempt to insert the copyright notice at the top of files that "
            "are missing one. Respects shebangs, but is otherwise naive and may cause "
            "issues such as duplicated copyright notices. Use at your own peril."
        ),
    )
    return parser.parse_args()


def guess_newline_type(filename: str):
    r"""Guess the newline type (\r\n or \n) for a file."""
    with open(filename, newline="", encoding="utf8") as f:
        lines = f.readlines()

    if not lines:
        return "\n"
    if lines[0].endswith("\r\n"):
        return "\r\n"
    return "\n"


def main() -> None:
    """
    Run the script.

    For more information, see the module docstring and `parse_args()` above,
    or run `python check_copyright.py --help`.
    """
    failed = False

    args = parse_args()
    with open(args.template, encoding="utf8") as f:
        copyright_template = f.read()

    for filename in args.filenames:
        with open(filename, encoding="utf8") as f:
            content = f.read()
            lines = content.splitlines(keepends=True)
        if not content:
            # empty files don't need a copyright notice
            continue
        if lines[0].startswith(SHEBANG):
            # then it's a shebang line;
            # try again skipping this line and any following blank lines
            start_index = 1
            while start_index < len(lines) and not lines[start_index]:
                start_index += 1
            test_content = "".join(lines[start_index:])
        else:
            test_content = content

        if not test_content.startswith(copyright_template):
            print(f"{filename}:0 - no matching copyright notice", file=sys.stderr)
            failed = True

            if args.unsafe_fix:
                # try and add the copyright notice in
                if lines[0].startswith(SHEBANG):
                    # preserve the shebang if present
                    new_content = (
                        lines[0] + "\n" + copyright_template + "\n" + "".join(lines[1:])
                    )
                else:
                    new_content = copyright_template + "\n" + "".join(lines)

                newline_type = guess_newline_type(
                    filename
                )  # try and keep the same newline type
                with open(
                    filename, encoding="utf8", mode="w", newline=newline_type
                ) as f:
                    f.write(new_content)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
