"""
This module contains useful function to support calibration workflows for Pakistan
scenarios.

The idea is that functions that are used multiple times are contained in one
place, rather than being duplicated across multiple scripts.
"""

import pandas as pd
from functools import partial

import typhoidsim as ty


def partial_unexp2susc(sus_saturation_age, sus_age_exposure_slope):
    """
    Create a partially applied function from unexp2susc_prob_function_gauld2018
    with given sus_saturation_age and sus_age_exposure_slope arguments.

    Args:
        sus_saturation_age (float): The age at which exposure reaches saturation.
        sus_age_exposure_slope (float): The slope of exposure with increasing age.

    Returns:
        callable: A partially applied function of unexp2susc_prob_function_gauld2018.
    """
    return partial(ty.unexp2sus_youth_prob_function_gauld2018, sus_saturation_age=sus_saturation_age,
                   sus_age_exposure_slope=sus_age_exposure_slope)


def partial_env_trapezoidal(kwarg_pars):
    """
    Return a partially evaluated function. This means that the pattern is set
    according to the parameters recieved in kwargs_pars. We can get exactly
    the value of the environmental modulation at time t (expressed in years)
    by calling

    my_modulation = partial_env_trapezoidal(**kwargs_pars)
    curent_modulation = my_modulation(t)

    Args:
        kwarg_pars:

    Returns:
         callable: A partially applied function of ty.asym_trapezoidal
    """
    return partial(ty.asym_trapezoidal, **kwarg_pars)
