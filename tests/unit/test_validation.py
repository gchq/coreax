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

"""
Tests for input validation functions.

The tests within this file verify that various input validation functions written
produce the expected results on simple examples.
"""

import unittest

import jax.numpy as jnp
from jax import random

import coreax.validation


class TestInputValidationRange(unittest.TestCase):
    """
    Tests relating to validation of inputs provided by the user lying in a given range.
    """

    def test_validate_in_range_equal_lower_strict(self) -> None:
        """
        Test the function validate_in_range with the input matching the lower bound.

        The inequality is strict here, so this should be flagged as invalid.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_in_range,
            x=0.0,
            object_name="var",
            strict_inequalities=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )

    def test_validate_in_range_equal_lower_not_strict(self) -> None:
        """
        Test the function validate_in_range with the input matching the lower bound.

        The inequality is not strict here, so this should not be flagged as invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=0.0,
                object_name="var",
                strict_inequalities=False,
                lower_bound=0.0,
                upper_bound=100.0,
            )
        )

    def test_validate_in_range_below_lower(self) -> None:
        """
        Test the function validate_in_range with the input below the lower bound.

        The input is below the lower bound, so this should be flagged as invalid.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_in_range,
            x=-1.0,
            object_name="var",
            strict_inequalities=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )

    def test_validate_in_range_equal_upper_strict(self) -> None:
        """
        Test the function validate_in_range with the input matching the lower bound.

        The inequality is strict here, so this should be flagged as invalid.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_in_range,
            x=100.0,
            object_name="var",
            strict_inequalities=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )

    def test_validate_in_range_equal_upper_not_strict(self) -> None:
        """
        Test the function validate_in_range with the input matching the lower bound.

        The inequality is not strict here, so this should not be flagged as invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=100.0,
                object_name="var",
                strict_inequalities=False,
                lower_bound=0.0,
                upper_bound=100.0,
            )
        )

    def test_validate_in_range_above_upper(self) -> None:
        """
        Test the function validate_in_range with the input above the upper bound.

        The input is above the upper bound, so this should be flagged as invalid.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_in_range,
            x=120.0,
            object_name="var",
            strict_inequalities=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )

    def test_validate_in_range_input_inside_range(self) -> None:
        """
        Test the function validate_in_range with the input between the two bounds.

        The input is within the upper and lower bounds, so this should not be flagged as
        invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=50.0,
                object_name="var",
                strict_inequalities=True,
                lower_bound=0.0,
                upper_bound=100.0,
            )
        )

    def test_validate_in_range_input_inside_range_negative(self) -> None:
        """
        Test the function validate_in_range with the input between the two bounds.

        The input is within the upper and lower bounds, so this should not be flagged as
        invalid. The lower bound and input are both negative here.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=-50.0,
                object_name="var",
                strict_inequalities=True,
                lower_bound=-100.0,
                upper_bound=100.0,
            )
        )

    def test_validate_in_range_invalid_input(self) -> None:
        """
        Test the function validate_in_range with an invalid input type.

        The input is a string, which cannot be compared to the numerical bounds.
        """
        self.assertRaises(
            TypeError,
            coreax.validation.validate_in_range,
            x="1.0",
            object_name="var",
            strict_inequalities=True,
            lower_bound=0.0,
            upper_bound=100.0,
        )

    def test_validate_in_range_input_no_lower_bound(self) -> None:
        """
        Test the function validate_in_range with the input between the two bounds.

        The input is below the upper bound, so this should not be flagged as invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=50.0,
                object_name="var",
                strict_inequalities=True,
                upper_bound=100.0,
            )
        )

    def test_validate_in_range_input_no_upper_bound(self) -> None:
        """
        Test the function validate_in_range with the input between the two bounds.

        The input is above the lower bound, so this should not be flagged as invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=50.0,
                object_name="var",
                strict_inequalities=True,
                lower_bound=0.0,
            )
        )

    def test_validate_in_range_input_no_lower_or_upper_bound(self) -> None:
        """
        Test the function validate_in_range with the input between the two bounds.

        The input is below the upper bound, so this should not be flagged as invalid.
        """
        self.assertIsNone(
            coreax.validation.validate_in_range(
                x=50.0,
                object_name="var",
                strict_inequalities=True,
            )
        )


class TestInputValidationInstance(unittest.TestCase):
    """
    Tests relating to validation of inputs provided by the user are a given type.
    """

    def setUp(self) -> None:
        """
        Set variables shared across tests.
        """
        self.var_must_be_of_type_message = "^var must be of type"

    def test_validate_is_instance_float_to_int(self) -> None:
        """
        Test the function validate_is_instance comparing a float to an int.
        """
        self.assertRaisesRegex(
            TypeError,
            self.var_must_be_of_type_message,
            coreax.validation.validate_is_instance,
            x=120.0,
            object_name="var",
            expected_type=int,
        )

    def test_validate_is_instance_int_to_float(self) -> None:
        """
        Test the function validate_is_instance comparing an int to a float.
        """
        self.assertRaisesRegex(
            TypeError,
            self.var_must_be_of_type_message,
            coreax.validation.validate_is_instance,
            x=120,
            object_name="var",
            expected_type=float,
        )

    def test_validate_is_instance_float_to_str(self) -> None:
        """
        Test the function validate_is_instance comparing a float to a str.
        """
        self.assertRaisesRegex(
            TypeError,
            self.var_must_be_of_type_message,
            coreax.validation.validate_is_instance,
            x=120.0,
            object_name="var",
            expected_type=str,
        )

    def test_validate_is_instance_float_to_float(self) -> None:
        """
        Test the function validate_is_instance comparing a float to a float.
        """
        self.assertIsNone(
            coreax.validation.validate_is_instance(
                x=50.0, object_name="var", expected_type=float
            )
        )

    def test_validate_is_instance_int_to_int(self) -> None:
        """
        Test the function validate_is_instance comparing an int to an int.
        """
        self.assertIsNone(
            coreax.validation.validate_is_instance(
                x=-500, object_name="var", expected_type=int
            )
        )

    def test_validate_is_instance_str_to_str(self) -> None:
        """
        Test the function validate_is_instance comparing a str to a str.
        """
        self.assertIsNone(
            coreax.validation.validate_is_instance(
                x="500", object_name="var", expected_type=str
            )
        )

    def test_type_tuple(self) -> None:
        """Check that validation passes if a tuple of types is passed."""
        coreax.validation.validate_is_instance(
            x="500", object_name="var", expected_type=(int, str)
        )

    def test_invalid_expected_type(self) -> None:
        """
        Test that correct TypeError is raised if an invalid type is expected.

        A list of types is invalid; needs to be a tuple or union.
        """
        self.assertRaisesRegex(
            TypeError,
            "expected_type must be a type, tuple of types or a union",
            coreax.validation.validate_is_instance,
            x=500,
            object_name="var",
            expected_type=[int, str],
        )


class TestInputValidationConversion(unittest.TestCase):
    """
    Tests relating to validation of inputs provided by the user convert to a given type.
    """

    def test_cast_as_type_int_to_float(self) -> None:
        """
        Test the function cast_as_type converting an int to a float.
        """
        self.assertEqual(
            coreax.validation.cast_as_type(x=123, object_name="var", type_caster=float),
            123.0,
        )

    def test_cast_as_type_float_to_int(self) -> None:
        """
        Test the function cast_as_type converting a float to an int.
        """
        self.assertEqual(
            coreax.validation.cast_as_type(x=123.4, object_name="var", type_caster=int),
            123,
        )

    def test_cast_as_type_float_to_str(self) -> None:
        """
        Test the function cast_as_type converting a float to a str.
        """
        self.assertEqual(
            coreax.validation.cast_as_type(x=123.4, object_name="var", type_caster=str),
            "123.4",
        )

    def test_cast_as_type_list_to_int(self) -> None:
        """
        Test the function cast_as_type converting a list to an int.
        """
        self.assertRaises(
            TypeError,
            coreax.validation.cast_as_type,
            x=[120.0],
            object_name="var",
            type_caster=int,
        )

    def test_cast_as_type_str_to_float_invalid(self) -> None:
        """
        Test the function cast_as_type converting a str to a float.

        In this case, there are characters in the string beyond numbers, which we expect
        to cause the conversion to fail.
        """
        self.assertRaises(
            TypeError,
            coreax.validation.cast_as_type,
            x="120.0ABC",
            object_name="var",
            type_caster=float,
        )

    def test_validate_array_size_first_dimension_valid(self):
        """
        Test the function validate_array_size considering the first dimension.

        Test that validate_array_size does not raise an error when checking the first
        dimension of an array is the known size.
        """
        self.assertIsNone(
            coreax.validation.validate_array_size(
                x=jnp.array([[1, 2, 3], [4, 5, 6]]),
                object_name="arr",
                dimension=0,
                expected_size=2,
            )
        )

    def test_validate_array_size_second_dimension_valid(self):
        """
        Test the function validate_array_size considering the second dimension.

        Test that validate_array_size does not raise an error when checking the second
        dimension of an array is the known size.
        """
        self.assertIsNone(
            coreax.validation.validate_array_size(
                x=jnp.array([[1, 2, 3], [4, 5, 6]]),
                object_name="arr",
                dimension=1,
                expected_size=3,
            )
        )

    def test_validate_array_size_first_dimension_invalid(self):
        """
        Test the function validate_array_size considering the first dimension.

        Test that validate_array_size does raise an error when checking the first
        dimension of an array is the wrong size.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_array_size,
            x=jnp.array([[1, 2, 3], [4, 5, 6]]),
            object_name="arr",
            dimension=0,
            expected_size=4,
        )

    def test_validate_array_size_second_dimension_invalid(self):
        """
        Test the function validate_array_size considering the second dimension.

        Test that validate_array_size does raise an error when checking the second
        dimension of an array is the wrong size.
        """
        self.assertRaises(
            ValueError,
            coreax.validation.validate_array_size,
            x=jnp.array([[1, 2, 3], [4, 5, 6]]),
            object_name="arr",
            dimension=1,
            expected_size=1,
        )

    def test_validate_array_size_empty_array(self):
        """
        Test the function validate_array_size on an empty array.

        Test that validate_array_size does not raise an error when checking the
        dimension of an empty array.
        """
        self.assertIsNone(
            coreax.validation.validate_array_size(
                x=jnp.array([]), object_name="arr", dimension=0, expected_size=0
            )
        )

    def test_validate_key_array_valid(self):
        """
        Test the function validate_key_array on a valid key array.

        Test that validate_key_Array does not raise an error when checking the
        dtype of a valid key array.
        """
        self.assertIsNone(
            coreax.validation.validate_key_array(x=random.key(0), object_name="key")
        )

    def test_validate_key_array_invalid(self):
        """
        Test the function validate_key_array on a invalid arrays.

        Test that validate_key_Array does raise an error when checking the dtype
        of an invalid 'old-style' PRNGKey array, an array of integers and a non
        Array object.
        """
        self.assertRaises(
            TypeError,
            coreax.validation.validate_key_array,
            x=random.PRNGKey(0),
            object_name="key",
        )
        self.assertRaises(
            TypeError,
            coreax.validation.validate_key_array,
            x=jnp.array([0, 1]),
            object_name="key",
        )
        self.assertRaises(
            TypeError,
            coreax.validation.validate_key_array,
            x=bool,
            object_name="key",
        )


if __name__ == "__main__":
    unittest.main()
