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

    def test_validate_is_instance_float_to_int(self) -> None:
        """
        Test the function validate_is_instance comparing a float to an int.
        """
        self.assertRaisesRegex(
            TypeError,
            "^var must be of type",
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
            "^var must be of type",
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
            "^var must be of type",
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

    def test_none_valid(self):
        """Test that validates when object is :data:`None`."""
        coreax.validation.validate_is_instance(
            x=None, object_name="var", expected_type=None
        )

    def test_none_invalid(self):
        """Test that raises when object is not :data:`None` but expected type is."""
        self.assertRaisesRegex(
            TypeError,
            "^var must be of type",
            coreax.validation.validate_is_instance,
            x=120.0,
            object_name="var",
            expected_type=None,
        )

    def test_tuple_none_valid(self):
        """
        Test that validates when object is :data:`None` and have a tuple of types.
        """
        coreax.validation.validate_is_instance(
            x=None, object_name="var", expected_type=(str, None)
        )

    def test_tuple_none_invalid(self):
        """Test that raises when :data:`None` is in tuple of expected types."""
        self.assertRaisesRegex(
            TypeError,
            "^var must be of type",
            coreax.validation.validate_is_instance,
            x=120.0,
            object_name="var",
            expected_type=(str, None),
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


if __name__ == "__main__":
    unittest.main()
