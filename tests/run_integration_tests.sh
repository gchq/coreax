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
# intended scope.

for filename in ./integration/test_*.py; do
    python -m pytest $filename
done
