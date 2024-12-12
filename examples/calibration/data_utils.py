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


def simulation_outputs_to_df(sim, batch_name="calib_pak_sindh", output_dir=None,
                             do_save=False):
    """
    Saves results from the simulation, and monitors in an friendly format (.csv)

    Args:
        sim(ss.Sim): a Sim object, already run
        batch_name (str): a string that will be prepended to all filenames generated
            within this function.
        output_dir (pathlib.Path): a Path object with the absolute path where to save the results
        do_save (bool): whether to save dataframes as csv files or not. Default: False

    Returns:
         tuple with multiple dfs (pd.DataFrame): the dataframes with results for every
             time step of the simulation, and the dataframes of each monitor
             found in the simulation.
    """

    if output_dir is None:
        import pathlib
        output_dir = pathlib.Path.cwd()


    # Export to df -- we can export results to a dataframe (and save as csv) for offline analysis
    dfs = [ty.to_df(sim)]
    names = ["sim"]
    #dfs = dict("sim":)
    monitors = sim.get_analyzers(label="monitor", partial=True)

    for monitor in monitors:
        monitor_df = monitor.to_df()
        dfs.append(monitor_df)
        names.append(monitor.name)

    if do_save:
        filename = ty.generate_unique_filename(root_str=batch_name)
        csv_ext = ".csv"
        for df, name in zip(dfs, names):
            df.to_csv(output_dir / f"{filename}-{name}{csv_ext}", index=False)
    return dfs


def get_reference_dataset_xls(dataset_name, filepath):
    """
    Dataset name refers to the sheet name in the excel file with all
    the reference data.
    """
    xls_file = pd.ExcelFile(filepath)
    if dataset_name in xls_file.sheet_names:
        df = pd.read_excel(xls_file, dataset_name)
        return df
    else:
        raise ValueError(f"Sheet {dataset_name} not found in {filepath}")


def get_reference_data_prevax(filepath=None):
    if filepath is None:
        filepath = "reference_data/SindhCalibration_newage_excludelockdown_2019.xlsx"
    reference_data = get_reference_dataset_xls(dataset_name="CasesByAge_prevax",
                                               filepath=filepath)
    reference_data = reference_data.rename(columns={"Cases_sum": "cases_sum",
                                                    "Cases_sum_corrected": "cases_sum_corrected"})

    # Parse age bins
    reference_data[["age_start", "age_end"]] = reference_data["AgeBin"].apply(parse_bin_edges)
    # Parse year bins
    reference_data[["year_start", "year_end"]] = reference_data["YearBin"].apply(parse_bin_edges)

    reference_data["age_bin_label"] = list(
        map(lambda x: ty.generate_age_bin_labels([x[0], x[1]]),
            zip(reference_data["age_start"], reference_data["age_end"])))

    reference_data["year_bin_label"] = list(
        map(lambda x: ty.generate_age_bin_labels([x[0], x[1]]),
            zip(reference_data["year_start"], reference_data["year_end"])))

    return reference_data


def aggregate_data_by_time(reference_data, start_year, end_year):
    time_mask = ((reference_data["year_start"] >= start_year) &
                 (reference_data["year_start"] < end_year))

    filtered_data = reference_data[time_mask]
    year_bin_str = f"[{start_year}, {end_year})"
    cols = ["cases", "cases_corrected", "population", "population_corrected"]

    new_rows = []
    for age_bin in filtered_data.AgeBin.unique():
        new_row = {"AgeBin": age_bin, "YearBin": year_bin_str}
        temp = filtered_data[filtered_data.AgeBin == age_bin]
        for col in cols:
            new_row.update({col: [temp[col].sum()]})
        new_rows.append(pd.DataFrame(new_row))
    reference_data = pd.concat([reference_data] + new_rows, ignore_index=True)
    reference_data.reset_index()
    return reference_data


def parse_bin_edges(str_bin):
    """
    Parse age or year bins defined in strings as
    [lower_bound, upper_bound) or [lower_bound, upper_bound].

    This function returns the bin edges in numeric representation.
    """
    import re
    values = re.sub(r'[\[\)]', '', str_bin).split(',')
    values = [float(v) for v in values]
    return pd.Series(values)


def lockdown_mask(years, target_year=2019.0):
    """
    Returns a mask that can be used to exclude the months of lockdown.
    Assumes years is the time in calendar years, in float representation.
    """
    year = years.astype(int)
    month = np.round((years - year) * 12)
    is_lockdown = ((year == target_year) & (month >= 2) & (month <= 6))
    return ~is_lockdown
