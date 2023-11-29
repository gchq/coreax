# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest

from coreax.validation import (
    cast_variable_as_type,
    validate_number_in_range,
    validate_variable_is_instance,
)


class TestInputValidation(unittest.TestCase):
    """
    Tests relating to validation of inputs provided by the user.
    """

    def test_validate_number_in_range(self) -> None:
        """
        Test the function validate_number_in_range across reasonably likely inputs
        """
        self.assertRaises(
            ValueError,
            validate_number_in_range,
            x=0.0,
            variable_name="var",
            lower_limit=0.0,
            upper_limit=100.0,
        )
        self.assertRaises(
            ValueError,
            validate_number_in_range,
            x=-1.0,
            variable_name="var",
            lower_limit=0.0,
            upper_limit=100.0,
        )
        self.assertRaises(
            ValueError,
            validate_number_in_range,
            x=120.0,
            variable_name="var",
            lower_limit=0.0,
            upper_limit=100.0,
        )
        self.assertIsNone(
            validate_number_in_range(
                x=50.0, variable_name="var", lower_limit=0.0, upper_limit=100.0
            )
        )
        self.assertIsNone(
            validate_number_in_range(
                x=-50.0, variable_name="var", lower_limit=-100.0, upper_limit=100.0
            )
        )

    def test_validate_variable_is_instance(self) -> None:
        """
        Test the function validate_number_in_range across reasonably likely inputs
        """
        self.assertRaises(
            TypeError,
            validate_variable_is_instance,
            x=120.0,
            variable_name="var",
            expected_type=int,
        )
        self.assertRaises(
            TypeError,
            validate_variable_is_instance,
            x=120,
            variable_name="var",
            expected_type=float,
        )
        self.assertRaises(
            TypeError,
            validate_variable_is_instance,
            x=120.0,
            variable_name="var",
            expected_type=str,
        )
        self.assertIsNone(
            validate_variable_is_instance(
                x=50.0, variable_name="var", expected_type=float
            )
        )
        self.assertIsNone(
            validate_variable_is_instance(
                x=-500, variable_name="var", expected_type=int
            )
        )
        self.assertIsNone(
            validate_variable_is_instance(
                x="500", variable_name="var", expected_type=str
            )
        )

    def test_cast_variable_as_type(self) -> None:
        """
        Test the function cast_variable_as_type across reasonably likely inputs
        """
        self.assertEqual(
            cast_variable_as_type(x=123, variable_name="var", type_caster=float), 123.0
        )
        self.assertEqual(
            cast_variable_as_type(x=123.4, variable_name="var", type_caster=int), 123
        )
        self.assertEqual(
            cast_variable_as_type(x=123.4, variable_name="var", type_caster=str),
            "123.4",
        )
        self.assertRaises(
            TypeError,
            cast_variable_as_type,
            x=[120.0],
            variable_name="var",
            type_caster=int,
        )


if __name__ == "__main__":
    unittest.main()
