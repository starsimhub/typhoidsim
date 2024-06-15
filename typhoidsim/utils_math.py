"""
Numerical and mathy utilities like math function that can be used
for parametrisation of otherwise constant parameters. .
"""

import warnings
import numpy as np
import sciris as sc
import starsim as ss
import numba as nb
import pandas as pd



# Specify all externally visible things this file defines
__all__ = ['sigmoid', 'gompertz', 'gompertz_dfun']


def sigmoid(x, max_x, slope):
    """
    Compute a sigmoid-like function. The range of x is between [0, max_x].

    This function is used in the age-based exposure mechanism in typhoid.

    Arguments:
        x (array-like): The array of x values at which the function is evaluated
        max_x (float): The asymptote of the function as x approaches infinity.
        slope (float): The slope parameter that controls how 'fast' we reach
        100% exposure (or 100% susceptible). If slope=0, this function is
        a linear function, if slope > 0, higher susceptibility is achieved
        faster. If slope < 0, higher susceptibility is reached slower.

    Returns:
      (ndarray): An array of y values corresponding to the sigmoid-like function
       evaluated at the input x values -- which are expected to represent age.
    """
    return 1.0 - (max_x - x) / (x * slope + max_x)


def double_sigmoid():
    """
    A stair-like function but built with sigmoid functions.
    """
    pass


def gompertz(x, a, b, c):
    """
    Compute the Gompertz function for a given set of parameters.
    This function is used for describing mortality and ageing-like processes.

    See:
    https://en.wikipedia.org/wiki/Gompertz_function

    The Gompertz function is defined as:
    f(x) = a * exp(-b * exp(-c * x))

    Arguments
        x (array-like): The array of x values at which the function is evaluated
        a (float): The asymptote of the function as x approaches infinity.
        b (float): Displacement along the x-axis
        c (float): The growth rate

    Returns:
        (ndarray): An array of y values corresponding to the gomeprtz function
        evaluated at the input x values.
    """
    return a*np.exp(-b*np.exp(-c*x))


def gompertz_dfun(x, a, b, c):
    """
    Compute the derivative of the Gompertz function with respect to x for a
    given set of parameters.

    This function is used in ecology for modelling ageing processes.

    The derivative of the Gompertz function is defined as:
    f'(x) = a * b * c * exp(-(b / exp(c * x)) - c * x)

    Arguments
    x (array-like): The array of x values at which the function is evaluated
    a (float): The asymptote of the Gompertz function as x approaches infinity.
    b (float): Displacement along the x-axis
    c (float): The growth rate

    Returns:
       (ndarray): An array of derivative values corresponding to the input x values.
    """
    return a*b*c*np.exp(-(b/np.exp(c*x)) - c*x)


def cfu_dependent_mean_dur():
    pass


def cfu_dependent_std_dur():
    pass
