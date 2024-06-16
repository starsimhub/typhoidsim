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


def double_sigmoid_exp(x, l1, l2, l3, x_12, x_23, s12, s23):
    """
    Calculate a stair-like function with 3 steps using two exponential
    sigmoid functions.

    This function simulates a stair-like characteristic with 3 steps
    (l1, l2, l3). The transition from l1 to l2 occurs around the
    point x_12, and from l2 to l3 around point x_23.

    The steepness between any two levels is governed independently
    by s12 and s23 respectively.

    Arguments:
        x (float or numpy.array): Input data.
        l1, l2, l3 (float): y-levels for each step.
        x_12, x_23 (float): x-points that indicate halfway between two levels.
        s12, s23 (float): Steepness of the curve between l1 and l2, and l2 and l3,
        respectively. The larger s_ij is, the closer to a step-like
        stair function we get.

    Returns:
        float or numpy.array: the function evaluated at x
    """

    y = l1 + (l2 / (1.0 + np.exp(-s12 * (x - x_12))) +
              l3 / (1.0 + np.exp(-s23 * (x - x_23))))

    return y


def double_sigmoid_tanh(x, l1, l2, l3, x_12, x_23, s=1.0):
    """
    This function simulates a stair-like function with 3 steps
    (l1, l2, l3), using a single specified steepness value (s) for
    the hyperbolic tangent.

    l1> l2 > l3 or l1 < l2 < l3

    Arguments:
        x (float or numpy.array): Input data.
        l1, l2, l3 (float): y-levels for each step.
        x_12, x_23 (float): x-points that indicate halfway between two levels.
        s (float): Steepness of the curve between two levels. The larger s is,
            the closer to a step-like stair function we get.

    Returns:
        float or numpy.array: y-values.

    Example:
        >>> import numpy as np
        >>> import typhoidsim.utils_math as tyum
        >>> x = np.arange(1_000_000, 60_000_000, 1024)
        >>> tyum.double_sigmoid_tanh(x, 2.235, 2.002, 1.5487, 5_050_000, 55_000_000, 1)
    """

    y = l1 + ((l2 - l1) * ((np.tanh(s * (x-x_12)) + 1)/2) +
              (l3 - l2) * ((np.tanh(s * (x-x_23)) + 1)/2))
    return y


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
