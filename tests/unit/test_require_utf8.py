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

"""Tests for the UTF-8 pre-commit hook."""

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from pre_commit_hooks.require_utf8 import main


def test_reports_actual_invalid_line_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid byte sequences should be reported against their source lines."""
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"first\nsecond\nthird \xff\nfourth\nfifth \xfe\n")
    monkeypatch.setattr(sys, "argv", ["require_utf8.py", str(file_path)])

    output_stream = StringIO()
    with redirect_stdout(output_stream), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = output_stream.getvalue()
    assert f"{file_path}: line 3" in output
    assert f"{file_path}: line 5" in output
    assert f"{file_path}: line 1" not in output


def test_valid_utf8_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid UTF-8 should continue to pass the hook."""
    file_path = tmp_path / "valid.txt"
    file_path.write_text("first\nsecond café\n", encoding="UTF-8")
    monkeypatch.setattr(sys, "argv", ["require_utf8.py", str(file_path)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
