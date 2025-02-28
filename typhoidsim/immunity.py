"""
Immunity waning functions
So far, all functions are related to parameterisation of box-exponential model for
(acquired) immunity waning. These functions facilitate stratifying parameters by age,
or by other criteria.
"""
import numpy as np
import sciris as sc

from . import defaults as tyd
from . import utils as tyu

# Specify all externally visible things this file defines
__all__ = ['imm_decay_by_age_default', 'imm_constant_dur_by_age_default', 'imm_ve0_by_age_default']


def imm_decay_by_age_default(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([505.27, 505.27, 505.27, 505.27])  # Decay time constant, in days, one value per age bin of interest Maximum protection at t=0 of receiving a vaccine
    return tyu.stratify_parameter_by_age(age_bins, vals / tyd.days_per_year)


def imm_ve0_by_age_default(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([1.0, 1.0, 1.0, 1.0])  # Maximum protection at t=0 of receiving a vaccine
    return tyu.stratify_parameter_by_age(age_bins, vals)


def imm_constant_dur_by_age_default(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([940.4, 240.9, 0.0, 0.0])  # Duration of constant immunity in days, one value per age bin of interest, assuming a box exponential model of immunity dynamics
    return tyu.stratify_parameter_by_age(age_bins, vals / tyd.days_per_year)
