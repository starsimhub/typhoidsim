"""
This module contains useful functions to support calibration workflows for Pakistan
scenarios, but do not quite belong in the core typhoidsim modules,

The idea is that functions that are used multiple times are contained in one
place, rather than being duplicated across multiple calibration scripts.
"""

import numpy as np
import pandas as pd

import sciris as sc
import starsim as ss
import typhoidsim as ty


def get_age_distribution_pakistan():
    """
    Parse data from json file used in EMOD simulations to get age distribution
    for Pakistan. Returns it in a format that can be passed to starsim's People,
    to build the appropriate population.

    Returns:
         df (pd.DataFrame): A dataframe with columns 'age' and 'value' that
             specifies the age distribution for this population.
    """
    # Lower edge of an age bin
    json_data = load_demogrphics_pakistan()
    age_bin_lb   = np.array(json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["ResultValues"])
    age_cum_prob = json_data["Nodes"][0]["IndividualAttributes"]["AgeDistribution"]["DistributionValues"]
    age_probs = np.diff(age_cum_prob)
    # Drop the last value of age_bin_lb because it is the right edge of the last
    # age bin, we only want the lower boundary (lb) values of each bin
    df = pd.DataFrame({'age': age_bin_lb[0:-1], 'value': age_probs})
    return df


def get_mortality_rates_pakistan():
    """
    Parse mortality rates from json file used in EMOD simulations.
    These rates are expressed as mortality rate in units of 1/year".

    Returns:
        df (pd.DataFrame): A dataframe (with columns Time, Sex, AgeGrpStart and mx)
            that starsim's demographic Death module understands.
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
    Placeholder to load and parse birth rates over mutiple years.
    The output df should have the columns "Year" and the column "CBR".
    CBR can expressed as birth rates are per 1000 people, per year; or
    percentages per year.
    """
    pass


def load_demogrphics_pakistan(json_file='TestDemographics_pak_updated.json'):
    """
    Load demogrphics data from json file used in EMOD simulation
    This file is usually found under the Assets/ directory.
    """
    data_home = ty.get_data_home()  # Assumes we have placed the file in typhoidsim/data directory
    json_data = sc.loadjson(data_home + "/" + json_file)
    return json_data


def load_empirical_data_pakistan(csv_file='TahirData_0928.csv'):
    """
    File with empirical data from Pakistan/Sindh
    Source: https://github.com/InstituteforDiseaseModeling/typhoid-pakistan-calibration/blob/main/calibration_Sindh/Assets/TahirData_0928.csv
    """
    data_home = ty.get_data_home()  # Assumes we have placed the file in typhoidsim/data directory
    df = pd.read_csv(data_home + "/" + csv_file, parse_dates=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def get_data_for_calibration_prevax(province="Sindh"):
    """
    Example script to prepare a df for starsim's Calibration class.
    This function extracts data from Pakistan from the yers 2018 and 2019,
    and arranges them in a dataframe with columns named "typhoid.new_infections",
    and "typhoid.prevalence". These column names correspond to 'paths' to results
    in the simulation object (after running the simulation).

    sim.results.typhoid.new_infections --> is a timeseries
    sim.results.typhoid.new_infections --> is a timeseries

    The Calibration class requires column names in dataframe that matches a
    'stream' or 'array' of data inside the simulation.
    """

    data = load_empirical_data_pakistan()

    # Column name indicate "results" available in the simulation.
    df = pd.DataFrame({"date": data[f"Date"],
                       "typhoid.new_infections": data[f"{province}_positive"],
                       "typhoid.prevalence": data[f"{province}_positivity"],   # TODO: need to update the name of the columns once we include tests with a testing report rate
                       "age_group_name": data[f"Ages"]})

    # Filter data
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'].dt.year.isin([2018, 2019]) &  # Keep only rows with year equal to 2018 or 2019.
               (df['age_group_name'] == 'All')] # Keep only rows with age_group_name equals 'All'.
    df['time'] = df['date'].dt.year + (df['date'].dt.month - 1) / 12
    df['time'] = df['time'].astype(float)
    df['typhoid.new_infections'] = df['typhoid.new_infections'].astype(float)
    df.drop(columns=['date', 'age_group_name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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


def save_simulation_outputs(sim, batch_name="calib_pak_sindh", output_dir=None):
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
    # TODO: export csv of histogram analyzer

    filename = ty.generate_unique_filename(root_str=batch_name)
    csv_ext = ".csv"
    sim_df.to_csv(output_dir / f"{filename}{csv_ext}", index=False)
    json_ext = ".json"
    #TODO: Save simulation parameters --  not working properly right now,
    # gets hung up in recursion and serialisation of objects (starsim distributions)
    #sim.to_json(filename=output_dir + filename + ".json", keys=['parameters'])
    return sim_df  # in case we want to do something else with the data


def get_reference_dataset_xls(dataset_name, filepath):
    """
    Dataset name refers to the sheet name in the excel file with all
    the reference data.
    """
    xls_file = pd.ExcelFile(filepath)
    if dataset_name in xls.sheet_names:
        df = pd.read_excel(xls, dataset_name)
        return df
    else:
        raise ValueError(f"Sheet {dataset_name} not found in {filepath}")


def get_reference_dataset_csv(dataset_name, filepath):
    """
    Dataset name refers to the sheet name in the original excel file with all
    the reference data. The sheetname now exists under the column
    'dataset_name'
    """
    csv_file = pd.read_csv(filepath)
    dataset_mask = (csv_file["dataset_name"] == dataset_name)
    df = csv_file.loc[dataset_mask, :]
    return df



def parse_bin_edges(str_bin):
    """
    Parse age or year bins defined in strings as
    [lower_bound, upper_bound) or [lower_bound, upper_bound].

    This function returns the bin edges in numeric representation.

    """
    # print str_bin
    str_bin = str_bin.replace('[', '')
    str_bin = str_bin.replace(']', '')
    str_bin = str_bin.replace(')', '')
    arr = str_bin.split(',')
    lower_bound, upper_bound = round(str(arr[0]).strip()), round(str(arr[1]).strip())
    return lower_bound, upper_bound