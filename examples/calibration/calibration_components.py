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
        v = par_attrs['value']
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


def get_calib_pars(calibration_step="step_1"):
    match calibration_step:
        case "step_1":
            calib_pars = dict(
                # typhoid acute infectiousness
                tai=dict(low=1e-4, high=1e5, guess=42_808, log=True),
                # typhoid environmental exposure rate
                teer=dict(low=0.0, high=10.0, guess=1.99),
                rel_trans=dict(low=1e-4, high=1, guess=1.0, log=True),
                init_prev=dict(low=0.0005, high=0.001, guess=0.0005)
            )
        case _:
            raise ValueError(f"Do not have calibration parameters: {calibration_step}. "
                             f"Please define them in this function.")
    return calib_pars


def get_calib_components(calibration_targets="all_cases"):

    match calibration_targets:
        case "all_cases":
            reference_data = utils.get_reference_dataset_csv(dataset_name="CasesByAgeAll",
                                                             filepath="reference_data/reference_data_flat_sindh.csv")
            return make_calib_components_by_age_yearly_incidence(reference_data)
        case _:
            raise NotImplementedError


def make_calib_components_by_age_yearly_incidence(reference_data):
    """
    Builds a list of calibration components. Each component is a data source
    """
    components = []
    num_age_bins = reference_data.age_bin_label.nunique()
    for this_age_bin in sorted(reference_data.age_bin_label.unique()):
        expected_data = extract_reference_data(reference_data, selected_age_bin=this_age_bin)
        extract_data_from_sim_fn = partial(extract_simulated_data_incidence, selected_age_bin=this_age_bin)
        components.append(ty.CalibComponent220(
                name=f"cases_by_age_{this_age_bin}",
                expected=expected_data,
                extract_fn=extract_data_from_sim_fn,
                conform="incident",
                nll_fn="beta",
                weight=1.0/num_age_bins,  # Not strictly necessary to weight it like this
            ))
    return components


def extract_simulated_data_incidence(sim, selected_age_bin="<2"):
    """
    Receive a sim object, extract necessary data, and output a dataframe
    that will be used by a CalibComponent.

    This is similar to the method 'retrieve_age_data' in AgeDistAnalyzer_Count.py

    We can have multiple extraction functions, depending on what data we need
    from the simulation and how we need to aggregate simulated data to
    match the empirical/reference data. Starsim's Calibration expects
    a specific format for the dataframes it works with.

    Each CalibComponent independently assesses pseudo-likelihood as part of
    evaluating the quality of input parameters.

    Returns:
        simulated_data (pandas.Dataframe): with columns:
         - 'n' number of agents in an age bin)
         - 'x' counts of the metric of interest (ie, new cases)
         - 'age_bin' a string with the human readable label of this age bin
         - index is time; index name is 't'
    """

    sim_results = sim.results.flatten()
    cases_key = "monitor_1_hist_b_ti_acute"    # New cases (acute), summed over the period of the monitor/report
    population_key = "monitor_2_hist_b_alive"  # Average (mean) number of agents alive over the period of the monitor/report
    lbl_to_idx = sim.get_analyzers()[0].age_bin_lbl_to_idx    # Mapping between age bin string labels and index in the results 2D arrays
    yearvec = pd.Series(sim_results["monitor_1_yearvec"][:])  # The time vector of the simulated data, expressed in "float" calendar years, ie 2000.0, 2000.1 ...
    this_idx = lbl_to_idx[selected_age_bin]
    # Build the dataframe the calibration component needs
    x = sim_results[cases_key][:, this_idx]
    n = sim_results[population_key][:, this_idx]
    simulated_data = pd.DataFrame(data={"n": n,
                                        "x": x,
                                        "age_bin": selected_age_bin},
                                  index=pd.Index(yearvec, name="t"))
    return simulated_data


def extract_reference_data(reference_data, selected_age_bin="<2"):

    """
    This function is similar to get_age_ref_data in AgeDistAnalyzer_Count.py
    """

    age_bin_mask = (reference_data["age_bin_label"] == selected_age_bin)
    dataset_data = reference_data.loc[age_bin_mask, :]
    yearvec = dataset_data["year_start"].astype(float)

    n = dataset_data["Population_surveillance"].astype(float).to_numpy()
    x = dataset_data["Cases"].astype(float).to_numpy()
    expected_data = pd.DataFrame(data={"n": n,
                                       "x": x,
                                       "age_bin": selected_age_bin},
                                 index=pd.Index(yearvec, name="t"))
    return expected_data
