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

    Returns:
         df (pd.DataFrame): A dataframe with columns 'age' and 'value' that capture age distribution
    """
    # Lower edge of an age bin
    json_data = load_demogrphics_pakistan()
    age_bin_lb = np.array(json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["ResultValues"])  # Drop the last value of age because is the right edge of the last age bin
    age_cum_prob = json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["DistributionValues"]
    age_probs = np.diff(age_cum_prob)
    df = pd.DataFrame({'age': age_bin_lb[0:-1], 'value': age_probs})
    return df


def get_mortality_rates_pakistan():
    """
    Parse mortality rates from json file used in EMOD simulations.

    These rates are expressed as mortality rate in units of 1/year".

    Returns:
        df (pd.DataFrame): A dataframe with columns
    """
    json_data = load_demogrphics_pakistan()
    # Get lower bound of each age bin
    age_bin_lb = np.array(json_data["Nodes"][0]["IndividualAttributes"]["MortalityDistributionMale"]["PopulationGroups"][0])[::2]
    # Years for which we have data available
    years = np.array(json_data["Nodes"][0]["IndividualAttributes"]["MortalityDistributionMale"]["PopulationGroups"][1])
    rates = np.array(json_data["Nodes"][0]["IndividualAttributes"]["MortalityDistributionMale"]["ResultValues"])[::2, :]

    # Process data
    year_data = np.tile(years, len(age_bin_lb))
    age_data = np.repeat(age_bin_lb, len(years))
    unfolded_rates = rates.reshape(-1)
    sex_data = np.repeat("Male", len(year_data))

    df_male = pd.DataFrame({"Time": year_data, "Sex": sex_data, "AgeGrpStart": age_data, "mx": unfolded_rates})

    # The demographics module needs data for male & female, we assume the
    # same rates apply.
    df_female = df_male.copy()
    df_female["Sex"] = "Female"

    # Concatenate the original and copied DataFrames
    df = pd.concat([df_male, df_female], ignore_index=True)
    return df


def get_crude_birth_rates_pakistan():
    """
    Placeholder to load and parse birth rates over mutiple year.
    The output df should have the columns "Year" and the column "CBR".
    CBR can expressed as birth rates are per 1000 people, per year; or
    percentages per year.
    """
    pass


def load_demogrphics_pakistan(json_file='TestDemographics_pak_updated.json'):
    """
    Load demogrphics data from json file used in eMOD simulation
    This file is usually found under the Assets/ directory.
    """
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


def save_outputs(sim, output_dir=None):
    """
    Saves results from the simulation in an analysis friendly format (.csv)

    Args:
        sim(ss.Sim): a Sim object, already run
        output_dir (pathlib.Path):  a Path object with the absolute path where to save the results

    Returns:
         sim_df (pd.DataFrame): the dataframe with results for every time step of the simulation
    """

    if output_dir is None:
        import pathlib
        output_dir = pathlib.Path.cwd()

    # Export to df -- we can export results to a dataframe (and save as csv) for offline analysis
    sim_df = ty.to_df(sim)

    batch_name = "calib_pak_sindh"
    filename = ty.generate_unique_filename(root_str=batch_name)
    csv_ext = ".csv"
    sim_df.to_csv(output_dir / f"{filename}{csv_ext}", index=False)
    json_ext = ".json"
    #TODO: Save simulation parameters --  not working properly right now,
    # gets hung up in recursion and serialisation of objects (starsim distributions)
    #sim.to_json(filename=output_dir + filename + ".json", keys=['parameters'])
    return sim_df  # in case we want to do something else with the data
