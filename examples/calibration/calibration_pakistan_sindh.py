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
import pandas as pd
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
    Specify the complete model, and create a simulation instance of typhoid
    with a simple vaccination intervention.

    Args:

    Returns:
    sim (starsim.Sim): a starsim simulation configured with the same high level parameters for
    the populatioin and typhoid disease, but with slightly different intervention parameters.

    """

    # HIGH-LEVEL SIM PARAMETERS
    pars = dict(
        start    =2018,          # Starting year
        n_years  =2.0,           # Duration of the simulation in years
        dt       =1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  =0,             # Do not print details of the run
    )

    # POPULATION
    ppl = ss.People(10_000)

    # DEMOGRAPHICS
    # Load age-specific mortality rate, expressed in deaths per 1000 people
    ty_data = ty.get_data_home()
    death_rates = pd.read_csv(ty_data / 'pakistan_2020_deaths.csv')

    # Crude birth rate in Pakistan 2020 per 1000 people
    cbr = 27
    vital_dynamics = [
        ss.Births(birth_rate=cbr, units=1e-3),
        ss.Deaths(death_rate=death_rates, units=1e-3)
    ]

    # DISEASE CONFIGURATION

    # Define the susceptible introduction curve
    # This curve defines an age-based transition from completeley immune to susceptible.
    sus_saturation_age = 20.0
    sus_age_exposure_slope = 2.0

    # Partially evaluated function with the parameters defined above
    p_unexp2sus_parc_fun = partial_unexp2susc(
        sus_saturation_age=sus_saturation_age,
        sus_age_exposure_slope=sus_age_exposure_slope)

    typhoids_pars = {'tai': tai,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.98,
                     'p_cpg': 0.108,
                     'init_prev': ss.bernoulli(0.1),
                     'p_unexp2sus': ss.bernoulli(p=p_unexp2sus_parc_fun)}

    typhoid = ty.Typhoid(pars=typhoids_pars)

    # CONTAMINATED VEHICLE
    environment = ty.EnvironmentalPool(pars={'transmission': ss.Pars(env2ppl_exposure_rate=ss.poisson(lam=lam))})

    # Create seasonal pattern ramp

    # OBSERVATIONS AND REPORTING

    # Create an analyzer that will provide the results we need to compare to target empirical data
    age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']

    # CREATE THE SIMULATION
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=vital_dynamics + [environment])
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

