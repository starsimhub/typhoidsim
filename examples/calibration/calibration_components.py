"""
This is a collection of functions, and classes to perform calibration.


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from functools import partial

import sciris as sc
import starsim as ss
import typhoidsim as ty

import data_utils as utils


def update_sim_pars_step_1(sim, calib_pars, **kwargs):
    """
    Also referred to as as build_sim function in some of starsim's tutorials.

    This function tells the Calibration class how to reach and update a parameter
    value for our specific model encapuslated in the sim object..

    The more modules our full model has, the more complex to navigate the path
    to find and update the required parameters.
    """
    # Access the modules whose parameters we need to modify dueing optimisation
    typh = sim.pars.diseases[0]
    # NOTE This way of getting to the environment is ugly but cannot do it a
    # different way atm in starsim, there are three demographics modules: births, deaths and environment
    env = sim.pars.demographics[2]

    for par_name, par_attrs in calib_pars.items():  # Loop over the calibration parameters
        v = par_attrs["value"]
        # Each item in calib_pars is a dictionary with keys like 'low', 'high',
        # 'guess', 'suggest_type', and importantly 'value'. The 'value' key is
        # the one we want to use as that's the one selected by the algorithm
        match par_name:
            case "tai":
                typh.pars.tai = v
            case "teer":
                env.pars.transmission.env2ppl_exposure_rate.lam = v
            case "rel_trans":
                env.pars.transmission.rel_trans = v
            case "init_prev":
                typh.pars.init_prev = ss.bernoulli(v)
            case _:
                raise NotImplementedError(f"Do not know how to update parameter {par_name}.")
    return sim


def update_sim_pars_step_2(sim, calib_pars, **kwargs):
    """
    Also referred to as as build_sim function in some of starsim's tutorials.

    This function tells the Calibration class how to reach and update a parameter
    value for our specific model encapuslated in the sim object..

    The more modules our full model has, the more complex to navigate the path
    to find and update the required parameters.
    """
    # Access the modules whose parameters we need to modify dueing optimisation
    vax_interventions = sim.pars.interventions
    # NOTE This way of getting to the environment is ugly but cannot do it a
    # different way atm in starsim, there are three demographics modules: births, deaths and environment
    monitor = sim.pars.analyzers
    breakpoint()

    for par_name, par_attrs in calib_pars.items():  # Loop over the calibration parameters
        v = par_attrs["value"]
        # Each item in calib_pars is a dictionary with keys like 'low', 'high',
        # 'guess', 'suggest_type', and importantly 'value'. The 'value' key is
        # the one we want to use as that's the one selected by the algorithm
        match par_name:
            case "vax_efficacy":
                pass
            case "coverage_routine":
                pass
            case "coverage_campaign":
                pass
            case "reporting_rate":  #  do we need a different value for each age group?
                pass
            case "duration_protection":
                pass
            case _:
                raise NotImplementedError(f"Do not know how to update parameter {par_name}.")
    return sim


def get_calib_pars(calibration_target="step_1"):
    match calibration_target:
        case "step_1":
            calib_pars = dict(
                # typhoid acute infectiousness
                tai=dict(low=1e4, high=1e6, guess=42_808, log=True),
                # typhoid environmental exposure rate
                teer=dict(low=0.0, high=10.0, guess=1.99),
            )
        case "step_2":
            calib_pars = dict(
                # typhoid acute infectiousness
                tai=dict(low=1e4, high=1e6, guess=42_808, log=True),
                # typhoid environmental exposure rate
                teer=dict(low=0.0, high=10.0, guess=1.99),
            )

        case "debug":
            calib_pars = dict(
                # typhoid acute infectiousness
                tai=dict(low=1e4, high=1e6, guess=42_808, log=True),
                # typhoid environmental exposure rate
                teer=dict(low=0.0, high=10.0, guess=1.99),
                rel_trans=dict(low=1e-4, high=1, guess=1.0, log=True),
                init_prev=dict(low=0.0005, high=0.001, guess=0.0005)
            )
        case _:
            raise ValueError(f"Do not have calibration parameters: {calibration_target}. "
                             f"Please define them in this function.")
    return calib_pars


def get_calib_components(calibration_target="step_1"):
    """
    Build calibration components (CalibComponents) required by
    a starsim 2.2.0 Calibration object.

    Args:
        calibration_target (str): a human-readable string describing the
           target/reference data we want to calibrate to.

    Returns:
        calib_components list[ty.CalibComponents]: a list with the necessary
            CalibComponents for that calibration_target

    """
    match calibration_target:
        case "step_1":
            # From the preprint: In step 1, we calibrated the model to the age
            # distribution of blood culture confirmed typhoid cases prior to vaccine introduction,
            # focusing on January 2019-December 2020
            # TODO: the sheet CasesByAge_prevax has data aggregated over the years [2017, 2020). Do we need to create a new sheet?
            reference_data = utils.get_reference_data_prevax()
            calib_components = make_calib_components_by_age_prevax(reference_data)
        case "step_2":
            # In step 2, we jointly calibrated parameters related to vaccination (including efficacy,
            # coverage of routine immunization, campaign coverage, and duration of protection from
            # vaccination for each age group) as well as a reporting rate to the age-specific incidence time
            # series for each age group (<2, 2-4, 5-9, 10-14, and 15+) for the full reference period (January
            # 2019-December 2023).
            reference_data = utils.get_reference_data_incidence()
            calib_components = make_calib_components_by_age_incidence(reference_data)
        case _:
            raise NotImplementedError

    return calib_components


def make_calib_components_by_age_prevax(reference_data):
    """
    Builds a list of calibration components. Each component is a data source.
    We can have multiple extraction functions, depending on what data we need
    from the simulation and how we need to aggregate simulated data to
    match the empirical/reference data. Starsim's Calibration expects
    a specific format for the dataframes it works with.

    Each CalibComponent independently assesses pseudo-likelihood as part of
    evaluating the quality of input parameters.
    """
    components = []
    num_age_bins = reference_data.age_bin_label.nunique()
    for this_age_bin in sorted(reference_data.age_bin_label.unique()):
        expected_data = extract_reference_data_prevax(reference_data,
                                                      selected_age_bin=this_age_bin)
        extract_data_from_sim_fn = partial(extract_simulated_data_prevax,
                                           selected_age_bin=this_age_bin,
                                           start_year=2017.0, end_year=2020.0)
        components.append(ty.CalibComponent220(
                name=f"cases_prevax",  # NOTE: can be ranamed to something else
                expected=expected_data,
                extract_fn=extract_data_from_sim_fn,
                conform="prevalent",
                nll_fn=ty.euclidean,
                weight=1.0/num_age_bins,  # Not strictly necessary to weight it like this
            ))
    return components


def make_calib_components_by_age_incidence(reference_data):
    """
    Builds a list of calibration components, for calibrating to age-specific incidence

    Data:
    - Cases_byage_all > Cases_corrected;
    - Population > Population_surveillance_corrected

    Sindh full time period: January 2019 - December 2023 (excluding lockdown)
    # TODO: ability to pass kwargs with start and end year
    """
    components = []
    num_age_bins = reference_data.age_bin_label.nunique()
    for this_age_bin in sorted(reference_data.age_bin_label.unique()):
        expected_data = extract_reference_data_incidence(reference_data,
                                                         selected_age_bin=this_age_bin,
                                                         start_year=2019.0, end_year=2024.0)
        extract_data_from_sim_fn = partial(extract_simulated_data_incidence,
                                           selected_age_bin=this_age_bin,
                                           start_year=2019.0, end_year=2024.0)
        components.append(ty.CalibComponent220(
                name=f"cases_incidence",  # NOTE: can be renamed to something else
                expected=expected_data,
                extract_fn=extract_data_from_sim_fn,
                conform="incident",
                nll_fn=ty.euclidean,
                weight=1.0/num_age_bins,  # Not strictly necessary to weight it like this
            ))
    return components


def extract_simulated_data_prevax(sim, selected_age_bin=None, start_year=2017.0, end_year=2020.0):
    """
    Receive a sim object, extract necessary data, and output a dataframe
    that will be used by a CalibComponent.

    This is similar to the method 'retrieve_age_data' in AgeDistAnalyzer_Count.py,
    but also does some of the steps calculate_likelihood() in AgeDistAnalyzer_Count.py,
    mainly the preprocessing steps needed to get exactly the data/qubaitty used
    to calculate the likelihgood/distance between reference data and simulated data.

    Args:
         sim (starsim Sim): a Sim object already run. Expects certain monitors to exist, as defined in models.py
         selected_age_bin (str): a human-readable labelf or the age bin of interest of the form 'age_lb-age_ub'.
             This age bin indicates the data are grouped by age in the interval [age_lb, age_ub). The
             age bin label comes from reference data.
         start_year (float): The initial year of the interval for data aggregation.
              Data collection and aggregation commences from this year.
         end_year   (float): The end year of the semi-open interval for data aggregation.
             Data are aggregated up until, but not including, this year.

         Data are aggregated over the interval [start_year, end_year)

    Returns:
        simulated_data (pandas.Dataframe): with columns:
         - 'n' number of observations; in this case is the total number of cases:
             sum over age bins and time interval of interest (start_year <= time < end year)
            - 'n' is not always used during calibration, this depends on the likelihood function used
         - 'x' value of the metric of interest (ie, can be counts like new cases, or a proportion)
         - 'age_bin' a string with the human readable label of this age bin
         - index is time; expected index name is 't', and so far it is expected to represent time
    """

    sim_results = sim.results.flatten()
    cases_key = "monitor_1_b_new_acute"        # New cases (acute), summed over the period of the monitor/report

    lbl_to_idx = sim.get_analyzers()[0].age_bin_lbl_to_idx      # Mapping between age bin string labels and index in the monitor results 2D arrays

    yearvec = sim_results["monitor_1_yearvec"][:]  # The time vector of the monitored simulated data, expressed in "float" calendar years, ie 2000.0, 2000.1 ...
    # Apply lockdown mask
    # TODO: confirm this is actually required here at all. The preprint says:
    #  The months of February 2020-June 2020 are excluded from the timeseries but
    #  the reference data for prevax is between [2017, 2020), so 2020 is not included.
    not_lockdown = utils.lockdown_mask(yearvec, target_year=2020.0)

    time_mask = ((yearvec >= start_year) & (yearvec < end_year) & not_lockdown)

    this_idx = lbl_to_idx[selected_age_bin]
    # Build the dataframe the calibration component needs
    x = sim_results[cases_key][time_mask, this_idx].sum()  # Sum over time, here we are only processing one age bin
    n = sim_results[cases_key][time_mask, :].sum().sum()   # Sum over time and over age bins

    requested_year_bin = f"{int(start_year)}-{int(end_year)}"
    year_index = np.array([(start_year + end_year)/2.0])  # The centre of the year bin
    simulated_data = pd.DataFrame(data={"n": n,
                                        "x": x/n,
                                        "age_bin": selected_age_bin,
                                        "year_bin": requested_year_bin},
                                  index=pd.Index(year_index, name="t"))
    return simulated_data


def extract_reference_data_prevax(reference_data, selected_age_bin=None):
    """
    This function is similar to get_age_ref_data in AgeDistAnalyzer_Count.py,
    plus the steps done in calculate_likelihood() to get exactly the data/quantity
    that is actually used to calculate likelihood/distance/cost during
    the calibration process.


    Args:
         reference_data (pandas DataFrame): a dataframe with the incidence data from
             sheet "CasesByAge_prevax" and "Population". Includes uncorrected and
             corrected values for cases_sum.The data have been aggregated in one single
            time bin of 3 years[2017, 2020)
        selected_age_bin (str): a human-readable labelf or the age bin of
             interest of the form 'age_lb-age_ub'. This age bin indicates the
             data are grouped by age in the interval [age_lb, age_ub). The
             age bin label comes from reference data.

    Returns:
        expected_data (pandas DataFrame): a dataframe in the format ready to be
        used by CalibComponents and Calibration classes can understand.
    """

    age_bin_mask = (reference_data["age_bin_label"] == selected_age_bin)
    year_bin = reference_data.year_bin_label.unique()[0]

    n = reference_data["cases_sum_corrected"].astype(float).sum()
    x = reference_data.loc[age_bin_mask, ["cases_sum_corrected"]].astype(float).to_numpy()[0][0]
    start = reference_data.year_start[0]
    end = reference_data.year_end[0]
    year_index = np.array([(start + end) / 2.0])   # Use the centre of the year bin as the time index
    expected_data = pd.DataFrame(data={"n": n,
                                       "x": x/n,
                                       "age_bin": selected_age_bin,
                                       "year_bin": year_bin},
                                 index=pd.Index(year_index, name="t"))
    return expected_data


def extract_simulated_data_incidence(sim, selected_age_bin=None, start_year=2017.0, end_year=2024.0):
    """
    Args:
         sim (starsim Sim): a Sim object already run. Expects certain monitors to exist, as defined in models.py
         selected_age_bin (str): a human-readable labelf or the age bin of interest of the form 'age_lb-age_ub'.
             This age bin indicates the data are grouped by age in the interval [age_lb, age_ub). The
             age bin label comes from reference data.

        start_year (float): Specifies the initial year. Data selection will
              commence from this year.
         end_year   (float): Specifies the termination year. Data selection
              will occur up to, but not including, this year.

    Returns:
        simulated_data (pandas DataFrame): a dataframe in the format ready to be
            used by CalibComponents and Calibration classes can understand,
            and its columns match those found in expected_data.
    """
    sim_results = sim.results.flatten()
    cases_key = "monitor_1_b_new_acute"       # New cases (acute), summed over the period of the monitor/report which we have set to be 1 month, so we can exclude lockdown months
    people_key = "monitor_2_b_n_alive"        # People


    lbl_to_idx = sim.get_analyzers()[0].age_bin_lbl_to_idx  # Mapping between age bin string labels and index in the monitor results which are 2D arrays

    yearvec = sim_results["monitor_1_yearvec"][:]  # The time vector of the monitored simulated data, expressed in "float" calendar years, ie 2000.0, 2000.1 ...

    # Apply lockdown mask
    not_lockdown = utils.lockdown_mask(yearvec, target_year=2020.0)
    # Select the correct interval of time
    time_mask = ((yearvec >= start_year) & (yearvec < end_year) & not_lockdown)

    # Age bin index
    this_idx = lbl_to_idx[selected_age_bin]

    # Build the dataframe the calibration component needs
    years = np.arange(start_year, end_year)
    year_masks = time_mask & (yearvec[None, :] == years[:, None])
    x = np.sum(year_masks * sim_results[cases_key][:, this_idx], axis=1)
    n = sim_results[people_key][np.argmax(year_masks, axis=1), :][:, 0]
    year_index = years + 0.5
    year_bin_labels = np.array([f"{int(year)}-{int(year + 1)}" for year in years])

    simulated_data = pd.DataFrame(data={"n": n,
                                        "x": x,
                                        "age_bin": selected_age_bin,
                                        "year_bin": year_bin_labels},
                                  index=pd.Index(year_index, name="t"))
    return simulated_data


def extract_reference_data_incidence(reference_data, selected_age_bin=None, start_year=2017.0, end_year=2024.0):
    """
    This function is similar to get_age_ref_data in IncidenceRateAnalyzer.py,
    plus the steps done in calculate_likelihood() to get exactly the data/quantity
    that is actually used to calculate likelihood/distance/cost during
    the calibration process.

    Args:
         reference_data (pandas DataFrame): a dataframe with the incidence data from
             sheet "CasesByAgeAll" and "Population". Includes uncorrected and corrected values
             for cases and n_people.The data have been aggregated in 1y bins.

         selected_age_bin (str): a human-readable labelf or the age bin of
             interest of the form 'age_lb-age_ub'. This age bin indicates the
             data are grouped by age in the interval [age_lb, age_ub). The
             age bin label comes from reference data.

         start_year (float): Specifies the initial year. Data selection will
              commence from this year.
         end_year   (float): Specifies the termination year. Data selection
              will occur up to, but not including, this year.

         The data selected will cover the interval [start_year, end_year).

    Returns:
        expected_data (pandas DataFrame): a dataframe in the format ready to be
            used by CalibComponents and Calibration classes can understand.

    expected_data should look like this (selected_age_bin="2-5")
                      n       x age_bin   year_bin
        0  1.258310e+06   526.0     2-5  2017-2018
        1  1.292788e+06   982.0     2-5  2018-2019
        2  1.328210e+06  1912.0     2-5  2019-2020
        3  7.960184e+05   624.0     2-5  2020-2021
        4  1.401993e+06  1970.0     2-5  2021-2022
        5  1.440408e+06  2115.0     2-5  2022-2023
        6  1.479875e+06  2780.0     2-5  2023-2024
    """

    age_bin_mask = (reference_data["age_bin_label"] == selected_age_bin)
    age_bin_ref_data = reference_data.loc[age_bin_mask, :]

    year_bin_mask = ((age_bin_ref_data.year_start >= start_year) & (age_bin_ref_data.year_end <= end_year))  # NOTE: we have the condition <= end_year because we still need to select the last bin [end_year-1, end_year)

    x = age_bin_ref_data.loc[year_bin_mask, ["cases_corrected"]].astype(float).to_numpy().flatten()
    n = age_bin_ref_data.loc[year_bin_mask, ["n_people_corrected"]].astype(float).to_numpy().flatten()
    year_bin = age_bin_ref_data.loc[year_bin_mask, ["year_bin_label"]].reset_index(drop=True)

    start = age_bin_ref_data.year_start
    end   = age_bin_ref_data.year_end
    year_index = np.array([(start + end) / 2.0]).flatten()   # Use the centre of the year bin as the time index
    # NOTE: if using the beta binomial for NLL, pass x:cases and n:n_people, if using euclidean 'NLL' then probably need to pass x: x/n (incidence rate)
    expected_data = pd.DataFrame(data={"n": n, "x": x, "age_bin":selected_age_bin,
                                       "year_bin": year_bin.values.flatten()},
                                 index=pd.Index(year_index, name="t"))
    return expected_data
