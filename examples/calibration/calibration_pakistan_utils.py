"""
This module contains useful functions to support calibration workflows for Pakistan
scenarios, but do not quite belong in the core typhoidsim modules,

The idea is that functions that are used multiple times are contained in one
place, rather than being duplicated across multiple calibration scripts.
"""

import numpy as np
import pandas as pd
from functools import partial

import sciris as sc
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
    return partial(ty.unexp2sus_youth_prob_function_gauld2018, sus_saturation_age=sus_saturation_age,
                   sus_age_exposure_slope=sus_age_exposure_slope)


def partial_env_trapezoidal(kwarg_pars):
    """
    Return a partially evaluated function. This means that the pattern is set
    according to the parameters recieved in kwargs_pars. We can get exactly
    the value of the environmental modulation at time t (expressed in years)
    by calling

    my_modulation = partial_env_trapezoidal(**kwargs_pars)
    curent_modulation = my_modulation(t)

    Args:
        kwarg_pars of function typhoidsim.utils_math.asym_trapezoidal

    Returns:
         callable: A partially evaluated ty.asym_trapezoidal
    """
    return partial(ty.asym_trapezoidal, **kwarg_pars)


def get_age_distribution_pakistan():
    """
    Parse data from json file used in EMOD simulations to get
    age distribution for Pakistan. Returns it in a format that
    can be passed to starsim's People, to build the appropriate population.
    """
    # Lower edge of an age bin
    json_data = load_demogrphics_pakistan()
    age_bin_lb = np.array(json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["ResultValues"])  # Drop the last value of age because is the right edge of the last age bin
    age_cum_prob = json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["DistributionValues"]
    age_probs = np.diff(age_cum_prob)
    df = pd.DataFrame({'age': age_bin_lb[0:-1], 'value': age_probs})
    return df


def load_demogrphics_pakistan(json_file='TestDemographics_pak_updated.json'):
    data_home = ty.get_data_home()  # Assumes we have placed the file in typhoidsim/data directory
    json_data = sc.loadjson(data_home + "/" + json_file)
    return json_data


def check_age_distribution(n_agents=100_000):
    """
    Run a quick sim to get the age distribution of the population, and plots the
    results.

    Args:
        n_agents (float): an integer number of agents that determines the size of the population.
        Can be changed to assess finite size effects for small agent populations.

    Returns:
         None
    """
    import matplotlib.pyplot as plt

    # Define the parameters
    pars = sc.objdict(
        start=2000,      # Starting year
        n_years=0.125,   # Number of years to simulate
        dt=1.0/365.0,    # Timestep of 1 day, expressed in years
        verbose=0,       # Print details of the run
        rand_seed=42,    # Set a non-default seed
    )

    ppl_pakistan = ss.People(n_agents,
                             age_data=get_age_distribution_pakistan())
    sim_pakistan = ss.Sim(pars=pars, people=ppl_pakistan)
    sim_pakistan.run()

    ty.plot_age_histogram(sim_pakistan.people)
    plt.show()
    return
