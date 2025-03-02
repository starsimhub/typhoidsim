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
__all__ = ['imm_decay_by_age', 'imm_constant_dur_by_age', 'imm_ve0_by_age', 'imm_draw_fn_constant']


def imm_decay_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of immunity decay values, expressed in days,  for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([505.27, 505.27, 505.27, 505.27])  # Decay time constant, in days, one value per age bin of interest Maximum protection at t=0 of receiving a vaccine
    return tyu.stratify_parameter_by_age(age_bins, vals / tyd.days_per_year)


def imm_ve0_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values of VE0, values between 0 and 1, for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([1.0, 1.0, 1.0, 1.0])  # Maximum protection at t=0 of receiving a vaccine
    return tyu.stratify_parameter_by_age(age_bins, vals)


def imm_constant_dur_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values of constant immunity duration, expressed in days, for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0.75, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([940.4, 240.9, 0.0, 0.0])  # Duration of constant immunity in days, one value per age bin of interest, assuming a box exponential model of immunity dynamics
    return tyu.stratify_parameter_by_age(age_bins, vals / tyd.days_per_year)


def imm_draw_fn_constant(module, uids, **kwargs):
    """
    This function assumes that each parameter, imm_*, follows the same
    distribution (family) across all age groups, though the distribution's
    parameters can be different across ages.

    Specifically in this case:

    - imm_constant_dur is a constant distribtion
    - imm_ve0 is a constant distribution
    - imm_decay is an constant distribution

    This means all agents aged x will have identical values of the corresponding
    immunity parameter (ie, ss.constant())
    """
    # Tell the intervention how to get the right values for each oimmunity parameter for each agent
    module.imm_constant_dur[uids] = module.imm_constant_dur_dist.pars.v(module.a_vaccinated[uids])
    # The exponential distribution gives us values of 1/duration
    module.imm_decay[uids] = module.imm_decay_dist.pars.v(module.a_vaccinated[uids])
    module.imm_ve0[uids] = module.imm_ve0_dist.pars.v(module.a_vaccinated[uids])
    return
