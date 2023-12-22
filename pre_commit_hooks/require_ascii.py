#!/usr/bin/env python3
"""Based on jumanjihouse/pre-commit-hooks require-ascii hook."""
import itertools
import sys

MAX_ASCII_CODE = 255  # https://theasciicode.com.ar/


def main() -> None:
    """Run the pre-commit check."""
    failed = False

    for file_path in sys.argv[1:]:
        with open(file_path, encoding="UTF-8") as rf:
            for line_number in itertools.count(start=1):
                try:
                    line = rf.readline()
                except UnicodeDecodeError as error:
                    line = ""  # avoid being unbound
                    print(f"{file_path}: line {line_number} {error!s}")
                    failed = True

                if not line:
                    break

                for column_number, character in enumerate(line, start=1):
                    code_point = ord(character)
                    if code_point > MAX_ASCII_CODE:
                        print(
                            f"{file_path}: line {line_number} column {column_number} "
                            f"character {character!r} (decimal {code_point})"
                        )
                        failed = True

    sys.exit(failed)


if __name__ == "__main__":
    main()
