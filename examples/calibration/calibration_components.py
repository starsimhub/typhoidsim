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


def parse_update_sim_pars(sim, calib_pars, **kwargs):
    """
    Also referred to as as build_sim function in some of starsim's ttutorials.

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
                env.pars.transmission.rel_trans = v / sim.pars["n_agents"]
            case "init_prev":
                typh.pars.init_prev = ss.bernoulli(v)
            case _:
                raise NotImplementedError(f"Do not know how to update parameter {par_name}.")
    return sim


def get_calib_pars(calibration_step="step_1"):
    match calibration_step:
        case "step_1":
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
            raise ValueError(f"Do not have calibration parameters: {calibration_step}. "
                             f"Please define them in this function.")
    return calib_pars


def get_calib_components(calibration_targets="cases_prevax"):
    match calibration_targets:
        case "cases_prevax":
            reference_data = utils.get_reference_data_prevax()
            return make_calib_components_by_age_prevax(reference_data)
        case _:
            raise NotImplementedError


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
        expected_data = extract_reference_data_prevax(reference_data, selected_age_bin=this_age_bin)
        extract_data_from_sim_fn = partial(extract_simulated_data_prevax, selected_age_bin=this_age_bin,
                                           start_year=2017.0, end_year=2020.0)
        components.append(ty.CalibComponent220(
                name=f"cases_prevax",
                expected=expected_data,
                extract_fn=extract_data_from_sim_fn,
                conform="prevalent",
                nll_fn=ty.euclidean,
                weight=1.0/num_age_bins,  # Not strictly necessary to weight it like this
            ))
    return components


def extract_simulated_data_prevax(sim, selected_age_bin=None, start_year=2018.0, end_year=2020.0):
    """
    Receive a sim object, extract necessary data, and output a dataframe
    that will be used by a CalibComponent.

    This is similar to the method 'retrieve_age_data' in AgeDistAnalyzer_Count.py,
    but also does some of the steps calculate_likelihood() in AgeDistAnalyzer_Count.py,
    mainly the preprocessing steps needed to get exactly the data/qubaitty used
    to calculate the likelihgood/distance between reference data and simulated data.

    Returns:
        simulated_data (pandas.Dataframe): with columns:
         - 'n' number of observations; in this case is the total number of cases: sum over age bins and time interval of interest (start_year <= time < end year)
            - 'n' is not always used during calibration, this depends on the likelihood function used
         - 'x' value of the metric of interest (ie, can be counts like new cases, or a proportion)
         - 'age_bin' a string with the human readable label of this age bin
         - index is time; index name is 't'
    """

    sim_results = sim.results.flatten()
    cases_key = "monitor_1_hist_b_ti_acute"        # New cases (acute), summed over the period of the monitor/report

    lbl_to_idx = sim.get_analyzers()[0].age_bin_lbl_to_idx      # Mapping between age bin string labels and index in the monitor results 2D arrays
    yearvec = sim_results["monitor_1_yearvec"][:]  # The time vector of the monitored simulated data, expressed in "float" calendar years, ie 2000.0, 2000.1 ...
    # Apply lockdown mask
    not_lockdown = utils.lockdown_mask(yearvec)
    time_mask = ((yearvec >= start_year) & (yearvec < end_year) & not_lockdown)

    this_idx = lbl_to_idx[selected_age_bin]
    # Build the dataframe the calibration component needs
    x = sim_results[cases_key][time_mask, this_idx].sum()  # Sum over time, here we are only processing one age bin
    n = sim_results[cases_key][time_mask, :].sum().sum()   # Sum over time and over age bins

    year_index = np.array([0.0])
    simulated_data = pd.DataFrame(data={"n": n,
                                        "x": x/n,
                                        "age_bin": selected_age_bin},
                                  index=pd.Index(year_index, name="t"))
    return simulated_data


def extract_reference_data_prevax(reference_data, selected_age_bin=None):
    """
    This function is similar to get_age_ref_data in AgeDistAnalyzer_Count.py,
    plus the steps done in calculate_likelihood() to get exactly the data/quantity
    that is actually used to calculate likelihood/distance/cost during
    the calibration process.
    """

    age_bin_mask = (reference_data["age_bin_label"] == selected_age_bin)

    n = reference_data["cases_sum"].astype(float).sum()
    x = reference_data.loc[age_bin_mask, ["cases_sum"]].astype(float).to_numpy()[0][0]
    year_index = np.array([0.0])
    expected_data = pd.DataFrame(data={"n": n,
                                       "x": x/n,
                                       "age_bin": selected_age_bin},
                                 index=pd.Index(year_index, name="t"))
    return expected_data
