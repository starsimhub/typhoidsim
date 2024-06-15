"""
General utilities
"""
import os
import warnings

import numpy as np
import numba as nb
import pandas as pd

import sciris as sc
import starsim as ss

import typhoidsim as ty

# Specify all externally visible things this file defines
__all__ = ['get_data_home', 'load_dataset']


def get_data_home(data_home=None):
    """
    Return a path to the directory for default datasets.
    This funciton is needed by `load_dataset()`, and avoids the
    problem of using relative paths.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the `TYPHOIDSIM_DATA` environment variable (if it exists)
    or otherwise it will use the default data directory.
    """

    if data_home is None:
        data_home = os.environ.get("TYPHOID_DATA", ty.options.data_dir)
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def load_dataset(ds_name, data_home=None, **kwargs):
    """
    Load default dataset from typhoidsim data directory.

    This function provides access to a small collection of datasets
    that are useful to set empirical distributions (ie, demographics, or
    gallstone probs by age and gender), rather than having those hardcoded
    in the code.

    The small datasets are expected to be simple tabular data in saved in csv
    files. This function may apply some small amount of preprocessing, but it's
    not intended to be a full ingest and preprocessing pipelines. The csv files
    are expected to be in an already 'ingestable' form and simply loaded with
    pandas.read_csv().

    Use `get_dataset_names` to see a list of available datasets.

    Arguments:
      ds_name (str): name of the dataset (``{name}.csv``).
      data_home (str/Path, optional): the directory in which to cache data.
      kwargs: additional keyword arguments passed through to `pandas.read_csv`.

    Returns:
       df(`pandas.DataFrame`): tabular data, with some preprocessing applied
       (depends on the dataset)

    # Inspired by seaborn example datasets functionality
    """

    pass