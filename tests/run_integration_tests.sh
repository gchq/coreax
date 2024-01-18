#!/bin/bash

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

# This script runs all integration tests, one by one, on the same process. This ensures
# all tests can be run without JAX re-using JIT compiled functions outside of their
# intended scope. It returns 1 if any test failed and 0 if all pass.
#
# This script must be run from the repository root directory.

# Track number of tests and number of failures
num_tests=0

# Final total return code: 0 if successful; positive for a failure
return_code=0

for filename in ./tests/integration/test_*.py; do
  num_tests=$((num_tests + 1))
  pytest $filename
  return_code=$((return_code + $?))
done

# Write blank line to aid readability of output
echo

if [ $return_code -eq 0 ]; then
  echo "PASS: All ${num_tests} integration tests passed"
  exit 0
fi

echo "FAILED: At least one failure out of ${num_tests} integration tests"
exit 1
