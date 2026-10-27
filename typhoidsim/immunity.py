"""
Immunity waning functions
So far, all functions are related to parameterisation of gamma model for
(vaccine acquired) immunity waning. These functions facilitate stratifying 
parameters by age or by other criteria.
"""
import numpy as np
import sciris as sc

from . import defaults as tyd
from . import utils as tyu
from . import utils_math as tyum

# Specify all externally visible things this file defines
__all__ = ['imm_decay_rate_by_age', 'imm_constant_dur_by_age', 'imm_ve0_by_age', 'box_gamma_waning']


def imm_decay_rate_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of immunity decay values, expressed in 1/years,  for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else tyd.days_per_year/np.array([505.27, 505.27, 505.27, 505.27])  # Decay rate constant, in 1/years, one value per age bin of interest
    return tyu.stratify_parameter_by_age(age_bins, vals)


def imm_ve0_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values of VE0, values between 0 and 1, for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([1.0, 1.0, 1.0, 1.0])  # Maximum protection at t=0 of receiving a vaccine
    return tyu.stratify_parameter_by_age(age_bins, vals)


def imm_constant_dur_by_age(age_bins=None, vals=None):
    """
    Args:
        age_bins (list | np.array): List of age values defining bins. Length n.
        vals (list | np.array): List of values of constant immunity duration, expressed in years, for each age bin. Length n-1

    Returns:
        callable: a function that takes as input the age of the agent (single float or array) and
         returns the appropriate value.
    """
    age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0, 2.0, 5.0, 15.0, 125.0])  # Age at vaccination bins
    vals = sc.promotetoarray(vals) if vals is not None else np.array([940.4, 240.9, 0.0, 0.0])/tyd.days_per_year  # Duration of constant immunity in years, one value per age bin of interest, assuming a box exponential model of immunity dynamics
    return tyu.stratify_parameter_by_age(age_bins, vals)


class box_gamma_waning:
    """
    The protection of a vaccine, as the `waning=` of an `ss.Vaccine`: full (`ve0`) for a while
    (`constant_dur`), then waning as a gamma distribution's survival (an exponential for `shape=1`),
    with values that depend on the age at vaccination (see `ty.vaccination_with_waning()`)

    Args:
        ve0 (float/func): the protection at vaccination, between 0 and 1, or a function of the age at vaccination (see `imm_ve0_by_age()`)
        constant_dur (float/func): years of full protection, or a function of the age (see `imm_constant_dur_by_age()`)
        shape (float): the shape of the gamma decay (1 for exponential)
        rate (float/func): the rate of the gamma decay per year, or a function of the age (see `imm_decay_rate_by_age()`)

    **Example**::

        vaccine = ss.Vaccine(efficacy=1.0, waning=ty.box_gamma_waning(constant_dur=0.5, rate=1.0))
    """
    def __init__(self, ve0=None, constant_dur=None, shape=1.0, rate=None):
        self.ve0 = imm_ve0_by_age() if ve0 is None else ve0
        self.constant_dur = imm_constant_dur_by_age() if constant_dur is None else constant_dur
        self.shape = shape
        self.rate = imm_decay_rate_by_age() if rate is None else rate
        return

    def __call__(self, t, sim, uids):
        """ The protection left for each agent who holds the vaccine (`uids`), `t` after their last dose (a duration) """
        t = np.asarray(t.years, dtype=float)
        age = sim.people.age.raw[uids] - t # The age at the last dose
        get = lambda par: par(age) if callable(par) else par
        out = get(self.ve0)*tyum.box_gamma(t, 0.0, get(self.constant_dur), self.shape, get(self.rate))
        return np.clip(out, 0.0, 1.0)
