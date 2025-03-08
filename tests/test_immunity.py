"""
Test immunity functions
"""
import typhoidsim as ty


def test_imm_decay_by_age():
    # Test for valid inputs
    age_bins = [0.75, 2.0, 5.0, 15.0, 125.0]
    vals = [505.27, 505.27, 505.27, 505.27]/ty.days_per_year
    default_function = ty.imm_decay_by_age(age_bins=age_bins, vals=vals)
    assert callable(default_function)

    # Test if the returned function is giving expected results
    # The value expected is corresponding to the bin of the age value
    assert default_function(1) == vals[0] / 365  # age falls in first bin
    assert default_function(3) == vals[1] / 365  # age falls in second bin

    # Test for None inputs
    default_function_none = ty.imm_decay_by_age()
    assert callable(default_function_none)
    assert default_function_none(
        1) == 505.27 / 365  # Check for expected default value

