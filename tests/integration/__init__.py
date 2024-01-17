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
Integration tests to verify functionality of the coreax library.

Tests for end-to-end runs in typical applications of the coreax library. These should be
called individually or serially, e.g. via a bash script, to avoid errors arising from
Jax tracers being reused by the parent process.

To elaborate: where multiple unit tests are run from a parent process, Jax is likely to
re-use compiled tracers if present. This can result in errors where data types within
the tracers differ (as they are likely to do in integration tests). Unfortunately, this
doesn't seem to be rectified by multiprocessing, so a simple method to run a batch of
tests from separate parent processes is to run or loop over them individually.
"""
