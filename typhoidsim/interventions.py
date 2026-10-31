"""
Typhoid-specific interventions: treatment of acute cases, and vaccination with waning
immunity and boosters. Both are Starsim deliveries (`ss.treat()` and `ss.vaccinate()`);
interventions on the environment (sanitation, food safety, seasonality) are changes to
the disease's parameters with `ss.modify()` (see the tutorials).
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import immunity as tyi

__all__ = ['acute_treatment', 'vaccination_with_waning']


def acute_treatment(efficacy=0.5, coverage=1.0, delay=None, disease='typhoid', name='acute_treatment', **kwargs):
    """
    Treatment of acute (symptomatic) typhoid: people seek treatment some days after the
    acute stage starts, and a share of them (`coverage`) are treated, which makes them
    less infectious (by `efficacy`), by contact and in what they shed, for the rest of the
    acute stage

    Returns the treatment's delivery (`ss.treat()`).

    Args:
        efficacy (float): the reduction in infectiousness while acute
        coverage (float): the share of those who seek treatment who are treated
        delay (Dist): the time from the start of the acute stage to seeking treatment (default lognormal, with a mean of 11.9 days and a median of 10.3, as in EMOD)
        disease (Typhoid/str): the disease, or its name
        name (str): the name of the treatment
        kwargs (dict): passed to `ss.treat()`

    **Example**::

        typhoid = ty.Typhoid()
        sim = ss.Sim(diseases=typhoid, networks='random', interventions=ty.acute_treatment(efficacy=0.5, coverage=0.7))
    """
    typhoid = disease if isinstance(disease, ss.Disease) else None
    acute = typhoid.acute if typhoid is not None else ss.state(f'{disease}.acute')
    onset = typhoid.onset.to(acute) if typhoid is not None else acute.enter
    if delay is None:
        delay = ss.lognorm_ex(mean=ss.days(11.93), std=ss.days(6.98)) # EMOD's lognormal with mean 2.332 and sd 0.543 of the log of the days
    drug = ss.Treatment(efficacy=dict(transmission=efficacy), during=acute, name=f'{name}_drug') # Held only while acute
    return ss.treat(drug, who=onset, delay=delay, coverage=coverage, name=name, **kwargs)


def vaccination_with_waning(start_year=None, end_year=None, age_pars=None, prob=None, prob_type='annual',
                            booster1_prob=0.0, booster2_prob=0.0, booster1_interval=None, booster2_interval=None,
                            imm_ve0=None, imm_constant_dur=None, imm_decay_shape=1.0, imm_decay_rate=None, name='vaccination'):
    """
    Routine vaccination of an age group, with optional boosters, and protection that
    is full for a while and then wanes (a box and then a gamma decay), with values
    that depend on the age at vaccination (see `ty.box_gamma_waning`)

    Each person in the age group who has not been vaccinated is vaccinated with a
    probability on each step, given by `prob` and `prob_type`. The boosters are given
    `booster1_interval` and `booster2_interval` years after the first dose (the second
    whether or not the first booster was received), each restarting the protection.

    Returns the deliveries: the first dose, and each booster with a probability above 0.

    Args:
        start_year (float): the year vaccination starts (default: the start of the sim)
        end_year (float): the year it stops (not included; default: never)
        age_pars (dict): `min_age` and `max_age` of the age group, in years
        prob (float): the probability of vaccination, over the period `prob_type` says
        prob_type (str): 'annual' (a probability per year), 'timestep' (per step, offered in daily rounds, so per day), 'interval' (over the whole time from start_year to end_year),
            or 'age_based' (over the time people are in the age group, or the interval if that is shorter)
        booster1_prob (float): the probability of the first booster, given a first dose
        booster2_prob (float): the probability of the second booster, given a first dose
        booster1_interval (float): years from the first dose to the first booster
        booster2_interval (float): years from the first dose to the second booster
        imm_ve0 (float/func): the protection at vaccination, by age at vaccination (see `ty.imm_ve0_by_age()`)
        imm_constant_dur (float/func): years of full protection, by age at vaccination (see `ty.imm_constant_dur_by_age()`)
        imm_decay_shape (float): the shape of the gamma decay
        imm_decay_rate (float/func): the rate of the gamma decay per year, by age at vaccination (see `ty.imm_decay_rate_by_age()`)
        name (str): the name of the first dose's delivery; the boosters are named after it

    **Example**::

        vx = ty.vaccination_with_waning(prob=0.66, age_pars=dict(min_age=0.75, max_age=2), booster1_interval=0.33, booster1_prob=0.8)
        sim = ss.Sim(diseases=ty.Typhoid(), networks='random', interventions=vx)
    """
    if booster1_interval is None and booster2_interval is not None:
        raise ValueError('booster2 should only be implemented if booster1 is also implemented. Please provide value for booster1_interval')
    for i,(p, interval) in enumerate([(booster1_prob, booster1_interval), (booster2_prob, booster2_interval)]):
        if interval is None and p > 0:
            raise ValueError(f'Booster {i+1} coverage {p} is non-zero, but no booster interval `booster{i+1}_interval` was provided.')
    if booster1_interval is not None and booster2_interval is not None and booster1_interval >= booster2_interval:
        raise ValueError(f'Time to first booster {booster1_interval} should be less than time to second booster {booster2_interval}')
    age_pars = sc.mergedicts(dict(min_age=0, max_age=np.inf), age_pars)
    min_age, max_age = age_pars['min_age'], age_pars['max_age']

    # The probability per year (or per step)
    prob = 1.0 if prob is None else prob
    match prob_type:
        case 'annual':
            coverage = ss.probperyear(prob)
        case 'timestep':
            coverage = prob
        case 'interval' | 'age_based':
            if prob_type == 'age_based' and (end_year is None or max_age - min_age <= end_year - start_year):
                years = max_age - min_age
            elif start_year is not None and end_year is not None:
                years = end_year - start_year
            else:
                raise ValueError(f'prob_type="{prob_type}" needs start_year and end_year')
            coverage = ss.probperyear(1 - (1 - prob)**(1/years))
        case _:
            raise ValueError(f'Invalid prob_type: {prob_type}. Must be one of annual, timestep, interval, or age_based.')

    waning = tyi.box_gamma_waning(ve0=imm_ve0, constant_dur=imm_constant_dur, shape=imm_decay_shape, rate=imm_decay_rate)
    vaccine = ss.Vaccine(efficacy=1.0, waning=waning, name=f'{name}_vaccine')
    in_group = (ss.ppl.age >= min_age) & (ss.ppl.age < max_age)
    interval = ss.days(1) if prob_type == 'timestep' else None # A typed rate as coverage= is a chance at each step while people are eligible; a bare probability is offered once, so in daily rounds
    first = ss.vaccinate(vaccine, who=in_group & ~vaccine.held, coverage=coverage, interval=interval, start=start_year, stop=end_year, name=name)
    out = [first]
    for i,(p, interval) in enumerate([(booster1_prob, booster1_interval), (booster2_prob, booster2_interval)]):
        if p > 0:
            out.append(ss.vaccinate(vaccine, who=first, delay=ss.years(interval), coverage=p, name=f'{name}_booster{i+1}'))
    return out
