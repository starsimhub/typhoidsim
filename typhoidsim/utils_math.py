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
__all__ = ['sigmoid']


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
      y : (ndarray)
        An array of y values corresponding to the sigmoid-like function
        evaluated at the input x values -- which are expected to represent age.
    """
    return 1.0 - (max_x - x) / (x * slope + max_x)


def double_sigmoid():
    pass