"""
This script contains step 1 of the calibration workflow to reproduce results from
Kraay et al. 2024 https://www.medrxiv.org/content/10.1101/2024.08.30.24312839v1

Country: Pakistan
Province: Sindh

Transmission routes:
- Environmental only

Period:
- Prior to vaccine introduction
- January 2018 - December 2019 (2 years)

Free parameters:
- typhoid acute infectiousness (TAI) expressed in CFUs
- lambda environment: average exposure rate to the environment, expressed in (in num exposures * volume / day)

Target:
- age distribution of blood culture confirmed typhoid cases
"""

import matplotlib.pyplot as plt
import optuna
from functools import partial

import starsim as ss
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
    return partial(ty.unexp2susc_youth_prob_function_gauld2018, sus_saturation_age=sus_saturation_age,
                   sus_age_exposure_slope=sus_age_exposure_slope)


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(tai=None, lam=None):
    """
    Create a simulation instance of typhoid with a simple vaccination intervention.

    Args:
        vax_eff (float): The efficacy of the typhoid vaccine. A value between 0 and 1 where 1 represents 100% efficacy.
        vax_cov (float): The coverage of the vaccination. A value between 0 and 1 where 1 represents 100% of population coverage.
        start (float): The year when the vaccination intervention starts.

    Returns:
    sim (starsim.Sim): a starsim simulation configured with the same high level parameters for
    the populatioin and typhoid disease, but with slightly different intervention parameters.

    """

    # Define high-level simulation parameters
    start_year = 2018
    pars = dict(
        start    =1990,          # Starting year
        n_years  =50.0,          # Duration of the simulation in years
        dt       =1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  =0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(10_000)

    # Demographics
    # TODO: add life expectancy 66.51 year
    # TODO: population growth rate 2.74%/year

    # Transition to susceptible

    # Define the susceptible introduction curve
    # This curve defines an age-based transition from completeley immune to susceptible.
    sus_saturation_age = 20.0
    sus_age_exposure_slope = 2.0

    # Partially evaluated function with the parameters defined above
    p_unexp2sus_parc_fun = partial_unexp2susc(
        sus_saturation_age=sus_saturation_age,
        sus_age_exposure_slope=sus_age_exposure_slope)

    # The disease
    typhoids_pars = {'tai': tai,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.98,
                     'p_cpg': 0.108,
                     'init_prev': ss.bernoulli(0.1),
                     'p_unexp2sus': ss.bernoulli(p=p_unexp2sus_parc_fun)}

    typhoid = ty.Typhoid(pars=typhoids_pars)

    environment = ty.EnvironmentalPool(pars={'transmission': ss.Pars(env2ppl_exposure_rate=ss.poisson(lam=lam))})

    # Create seasonal pattern ramp

    # Create intervention for reporting to get positive cases 'cases'

    # Create an analyzer that will provide the results we need to compare to target empirical data
    age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']

    # Create the simulation
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=environment)
    # Run multisim with 100 sims?

    return sim


def objective_step_1(trial):
    """ The cost function to optimise in step 1 of calibration"""
    p1 = trial.suggest_float('vax_eff', 0.0, 1.0)
    p2 = trial.suggest_float('vax_cov', 0.0, 1.0)
    p3 = trial.suggest_float('start_year', 2000.0, 2001.0)

    # include variance of cost function?
    cost = None
    return cost

