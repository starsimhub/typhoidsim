import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

import sciris as sc
import starsim as ss
import typhoidsim as ty

import data_utils as utils


def parse_update_sim_pars(sim, calib_pars, **kwargs):
    """
    Also referred to as as build_sim function.

    Tell the Calibration class how to reach and update a parameter value
    for our specific model.

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
                teer=dict(low=0.0, high=10.0, guess=1.99)
                # The path always consists of three components/steps.
            )
        case _:
            raise ValueError(f"Do not have calibration parameters: {calibration_step}. "
                             f"Please define them in this function.")
    return calib_pars


def make_calib_components(targets="all_cases"):
    df = utils.load_empirical_data_pakistan()
    # Add a column with a similar representation of time
    df["yearvec"] = np.array([sc.datetoyear(t) for t in df["Date"] if isinstance(t, datetime.date)])
    df_2_to_15 = df.loc[(df["Ages"] == "Kids2to15"), :]
    data = df_2_to_15["Sindh_positive"].astype(float).to_numpy()

    # TODO: find a better way to do this, in case we change the age bins
    age_bin_labels = ['<2', '2-15', '15+']  # human readable labels
    # Make dictionary to map labes to array index in analyzer result array
    age_bins_dict = {label: idx for idx, label in enumerate(age_bin_labels)}

    # Example with females between 2 <= age < 4
    # NOTE: This calib components objects are like the Analyzers in typhoid-pakistan-calibration repo
    f_infectious_2_to_15 = ty.CalibComponent220(
        name='f_positive_2_15',
        # Reference data
        expected=pd.DataFrame({
            'n': data * 5.0,  # !!! TODO: CHANGE!!! Made up scaling because "Sindh_tested" is all NaNs for this age group
            'x': data,  # Count/Number of individuals found to test positive
        }, index=pd.Index(df_2_to_15["yearvec"], name='t')),  # On these dates
        # Extract equivalent data from the simulation
        extract_fn=lambda sim: pd.DataFrame({
            'n': sim.analyzers.hist_by_age_sex.results.hist_f_alive[:, age_bins_dict['2-15']],     # Number of individuals who were tested
            'x': sim.analyzers.hist_by_age_sex.results.hist_f_infected[:, age_bins_dict['2-15']],  # Number of individuals whose test was positive
        }, index=pd.Index(sim.analyzers.hist_by_age_sex.yearvec, name='t')),  # Index is time

        conform='prevalent',
        nll_fn='beta',
        weight=1,  # Not required if only one component
    )
    components = [f_infectious_2_to_15]
    return components
