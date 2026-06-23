"""
Define calibration helpers for ss.Calibration()
"""
import os
import datetime

import numpy as np
import pandas as pd
import sciris as sc
import optuna as op
import matplotlib.pyplot as plt
import starsim as ss
from scipy.special import gammaln

from . import utils as tyu


__all__ = ['euclidean', 'weighted_euclidean', 'normalized_median_absolute_error', 'calib_to_df']


def validate_sim_data(data=None, die=None):
    """
    Validate data intended to be compared to the sim outputs, e.g. for calibration

    Args:
        data (df/dict): a dataframe (or dict) of data, with a column "time" plus data columns of the form "module.result", e.g. "hiv.new_infections"
        die (bool): whether to raise an exception if the data cannot be converted (default: die if data is not None but cannot be converted)

    """
    success = False
    if data is not None:
        # Try loading the data
        try:
            data = sc.dataframe(data) # Convert it to a dataframe
            timecols = ['t', 'timevec', 'tvec', 'time', 'day', 'date', 'year'] # If a time column is supplied, use it as the index
            found = False
            for timecol in timecols:
                if timecol in data.cols:
                    if found:
                        errormsg = f'Multiple time columns found: please ensure only one of {timecols} is present; you supplied {data.cols}.'
                        raise ValueError(errormsg)
                    data.set_index(timecol, inplace=True)
                    found = True
            success = True

        # Data loading failed
        except Exception as E:
            errormsg = f'Failed to add data "{data}": expecting a dataframe-compatible object. Error:\n{E}'
            if not die:
                print(errormsg)
            else:
                raise ValueError(errormsg)

    # Validation
    if not success and die == True:
        errormsg = 'Data "{data}" could not be converted and die == True'
        raise ValueError(errormsg)

    return data


def return_fig(fig, **kwargs):
    """ Do postprocessing on the figure: by default, don't return if in Jupyter, but show instead """
    is_jupyter = False
    if is_jupyter:
        print(fig)
        plt.show()
        return None
    else:
        return fig


def calib_to_df(calib_object):
    if self.after_msim is None:
        print("Please run .check_fit()")
        return
    dfs = []
    for component in self.components:
        sim_df = []
        for sim in self.after_msim.sims:
            df1 = component.extract_fn(sim)
            df1["source_data"] = "predicted"
            df1["rand_seed"] = int(sim.pars["rand_seed"])
            df1["component_name"] = component.name
            sim_df.append(df1)
        sim_df = pd.concat(sim_df)
        exp_df = component.expected
        exp_df["source_data"] = "expected"
        exp_df["component_name"] = component.name
        df = pd.concat([sim_df, exp_df])
        df.reset_index(inplace=True)
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def linear_interp(expected, actual):
    """
    Use for prevalent data like prevalence

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually from empirical sources).
             The index should be the time in either floating point 'calendar years' or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually from simulated sources) data
             The index should be the time in either floating point 'calendar years' or datetime.
    Returns:
        conformed (pd.Dataframe): dataframe containing the actual (aka simulated/predictec) data
            that have been interpolated to match a common timeframe with the data in `expected`.
            The interpolation ensures that the two datasets (expected and actual)
            can be compared directly (one-to-one) or used together in further
            analysis, because they are now aligned to the same time grid.
    """
    conformed = pd.DataFrame(index=expected.index)
    common_time_grid = expected.index
    interp_cols = ["x", "n"]
    for col in interp_cols:
        conformed[col] = np.interp(x=common_time_grid, xp=actual.index, fp=actual[col])
    conformed = _handle_categorical_columns(actual, conformed, interp_cols)
    return conformed


def linear_accum(expected, actual):
    """
    Interpolate in the cumulative of column "x", then differentiate.
    Use for incident-like data such as new_deaths

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually from empirical sources).
             The index should be the time in either floating point 'calendar years' or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually from simulated sources) data
             The index should be the time in either floating point 'calendar years' or datetime.

    Returns:
        conformed (pd.Dataframe): dataframe containing the actual or current data
            that have been interpolated to match a common timeframe with the data in `expected`.
            The interpolation ensures that the two datasets (expected and actual)
            can be compared directly (one-to-one) or used together in further
            analysis, because they are now aligned to the same time grid.

    """
    conformed = pd.DataFrame(index=expected.index)
    common_time_grid = expected.index
    expected_sampling_period = np.diff(common_time_grid)
    assert np.all(expected_sampling_period == expected_sampling_period[0])  # Check we have regularly sampled data

    # Make cumulative
    cum_time_grid = np.append(common_time_grid, common_time_grid[-1] + expected_sampling_period[0])  # Add one more because later we'll diff

    if isinstance(actual.index, pd.DatetimeIndex):
        actual_time_grid = np.array([sc.datetoyear(t) for t in actual.index if isinstance(t, datetime.date)])
    else:
        actual_time_grid = actual.index
    interp_cols = ["x", "n"]
    for col in interp_cols:
        sdi = np.interp(x=cum_time_grid, xp=actual_time_grid, fp=actual[col].cumsum())
        conformed[col] = pd.Series(np.diff(sdi), index=common_time_grid)
    conformed = _handle_categorical_columns(actual, conformed, interp_cols)
    return conformed


def _handle_categorical_columns(actual, conformed, interp_cols):
    """
    A convinience function to map other columns, like categorical columns, from
    the index in the actual dataframe, to the index of the
    actual conformed dataframe.
    """
    other_cols = list(set(actual.columns) - set(interp_cols))
    # Handle other columsn, inlcuding those with categorical data
    actual_index_array = actual.index.to_numpy()
    conformed_index_array = conformed.index.to_numpy()
    for col in other_cols:
        abs_diff_matrix = np.abs(conformed_index_array[:, None] - actual_index_array)
        min_diff_indices = np.argmin(abs_diff_matrix, axis=1)
        temp = actual.loc[actual_index_array[min_diff_indices], [col]].reset_index(drop=True)
        temp.index = conformed.index
        conformed[col] = temp[col]
    return conformed


def euclidean(expected, predicted):
    """
    Euclidean distance between expected and predictec/simulated data

    Args:
        expected (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from the reference dataset.
        predicted (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from simulated dataset.

    Returns:
        nll (float): negative Euclidean distance between expected and predicted values.
    """
    e_x, a_x = expected["x"], predicted["x"]
    ll = np.sqrt(((e_x - a_x)**2).sum())
    return ll


def weighted_euclidean(expected, predicted):
    """
    Weighted Euclidean distance between expected and predictec/simulated data. Also called
    weighted_squares in calibra ll calculators.

    Args:
        expected (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from the reference dataset.
        predicted (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from simulated dataset.

    Returns:
        nll (float): negative weighted Euclidean distance between expected and predicted values.
    """
    e_x, a_x = expected["x"], predicted["x"]
    ll = np.sqrt(((e_x - a_x)**2 / e_x).sum())
    return ll


def normalized_median_absolute_error(expected, predicted):
    """
    The normalized median absolute error. A highly robust goodness-of-fit metric.

    Args:
        expected (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from the reference dataset.
        predicted (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from simulated dataset.

    Returns:
        ngof (float): negative goodness of fit -- the Calibration class tries to maximise the score of the objective function.
    """
    e_x, a_x = expected["x"], predicted["x"]
    gof = compute_gof(e_x, a_x, as_scalar='median')  # Normalized median absolute error -- highly robust
    return gof
