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
Tests to verify performance of the coreax library.

Tests for specific functions, methods and classes in the coreax library, to ensure
performance is as expected.

The purpose of these tests is to ensure that Jax JIT applications are working as
expected. If a JIT function or decorator is removed in development, e.g. for debugging,
then these tests should catch this. Similarly, if JIT is being used unnecessarily, then
this too should be caught. We recommend adding new tests here where use of JIT is
present in the code.

Test structure is as follows: call a function that makes use of JIT a number of times,
forcing Jax to recompile it each time. Then call the function again for another number
of times, allowing Jax to reuse the JIT tracers. Both of these sets of calls are timed,
and the timings are supplied to a two-sample Kolmogorov-Smirnov test. If the timings are
significantly different (where 'significant' is a configurable parameter in the test
setup), then the test passes; else it fails.
"""
