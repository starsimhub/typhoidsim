"""
A collection of commonly used eligibility criteria, wrapped in functions readu
to used and be passed to interventions that take a callable as keyword arguments
"""
import numpy as np
import functools

import typhoidsim.defaults as tyd

__all__ = ['eligibility_by_age', 'eligibility_by_sex', 'eligibility_by_age_sex']
__all__ += ['eligibility_all_females', 'eligibility_all_males']


def eligibility_by_age(sim, age_min=0.0, age_max=tyd.max_age):
    """
    Returns uids of living people who are in the age group
    age_min <= age < age_max (semi-open interval).
    """
    eligible_uids = (sim.people.alive & (sim.people.age >= age_min) & (sim.people.age < age_max)).uids
    return eligible_uids


def eligibility_by_sex(sim, sex=0):
    """
    Returns uids of living of a given biological sex people who are in the age group
    age_min <= age < age_max (semi-open interval).
    """
    sex_state = sim.people.female if sex == 0 else sim.people.male  # v3: sex array removed; 0=female, 1=male
    eligible_uids = (sim.people.alive & sex_state).uids
    return eligible_uids


def eligibility_by_age_sex(sim, age_min=0.0, age_max=tyd.max_age, sex=0):
    """ Compose more complex eligibility criteria """
    eligible_age_uids = eligibility_by_age(sim, age_min=age_min, age_max=age_max)
    eligible_sex_uids = eligibility_by_sex(sim, sex=sex)
    eligible_uids = np.intersect1d(eligible_age_uids, eligible_sex_uids)
    return eligible_uids


# Aliases
eligibility_all_females = eligibility_by_age_sex
eligibility_all_males = functools.partial(eligibility_by_age_sex, age_min=0.0, age_max=tyd.max_age, sex=1)
