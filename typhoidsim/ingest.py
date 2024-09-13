"""
Functionality related to ingesting data from files. Ingesting encompasses
loading data from files, parsing data and reformatting data, and processing
so they tcan be consumed by typhoidsim or starsim.
"""
import numpy as np

import sciris as sc

from .data import country_household_size_distribution as household_size_distribution
from .data import country_household_head_age_distribution as household_head_age_distribution

__all__ = ["get_age_distribution", "get_household_size",
           "get_household_size_distribution", "get_household_head_age_distribution",
           "get_household_head_age_by_size_distribution"]


def get_age_distribution(location):
    """
    A small wrapper of synthpops.people.loaders.get_age_distributon().
    This function returns the array of age distribution of 'location',
    in the shape required by starsim People's age_data keyword argument.

    Args:
        location (str): the name of the country

    Returns:
        age_data (np.array): the array of age distribution of 'location'
    """
    import synthpops.people.loaders as spl
    # synthpops returns an Nx3 array, but people needs a Nx2 array
    # with the lower bound of each age bin and the count/proportion in that age bin
    age_data = spl.get_age_distribution(location)[:, [0, 2]]
    return age_data


def get_household_size(location):
    """
    A small wrapper of synthpops.people.loaders.get_household_size().
    This function returns the average household size for that location. This
    parameter is necessary when creating a HouseholdNet.

    Args:
        location (str): the name of the country

    Returns:
        household_size (float): the average household size of 'location'.
    """
    import synthpops.people.loaders as spl
    return spl.get_household_size(location)


def get_household_size_distribution(location=None):
    """
    Load household size distribution.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        household_size_distribution (np.array): Nx2 array. First column household size;
        second column: fraction of household of that size

    """
    import synthpops.people.loaders as spl

    # Load the raw data
    data = sc.dcp(household_size_distribution.data)

    entries = spl.map_entries(data, location)

    max_hh_size = 10
    result = {}

    for loc, hh_size_distribution in entries.items():
        hhs_data = []
        for hh_size, hh_size_perc in hh_size_distribution.items():
            if hh_size[-1] == '+':
                 val = [int(hh_size[:-1]), max_hh_size, hh_size_perc]
            else:
                size = hh_size.split('-')
                if len(size) == 1:
                     val = [int(size[0]), int(size[0]), hh_size_perc]
                else:
                     val = [int(size[0]), int(size[1]), hh_size_perc]
            hhs_data.append(val)
        hhs_data = np.array(hhs_data)
        bins = np.hstack([np.arange(start, end + 1) for start, end in hhs_data[:, :2]])
        bin_widths = (hhs_data[:, 1] - hhs_data[:, 0] + 1).astype(int)
        vals = np.repeat(hhs_data[:, 2] / bin_widths, bin_widths) / 100.0  # transform to proportion
        hh_size_dist = np.column_stack((bins, vals))
        result[loc] = hh_size_dist

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_household_head_age_distribution(location=None):
    """
    Load household head's age distribution.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        household_head_age_distribution (float): N x 2: household head age bracket and proportion

    """
    import synthpops.people.loaders as spl

    # Load the raw data
    data = sc.dcp(household_head_age_distribution.data)

    entries = spl.map_entries(data, location)

    max_age = 100
    result = {}

    for loc, hh_ha_distribution in entries.items():
        hhha_data = []  # (h)ouse(h)old (h)ead (a)ge
        for hh_ha, hh_ha_perc in hh_ha_distribution.items():
            if hh_ha[-1] == '+':
                 val = [int(hh_ha[:-1]), max_age, hh_ha_perc]
            else:
                size = hh_ha.split('-')
                val = [int(size[0]), int(size[1]), hh_ha_perc]
            hhha_data.append(val)
        hhha_data = np.array(hhha_data)
        bins = np.hstack([np.arange(start, end + 1) for start, end in hhha_data[:, :2]])
        bin_widths = (hhha_data[:, 1] - hhha_data[:, 0] + 1).astype(int)
        vals = np.repeat(hhha_data[:, 2] / bin_widths, bin_widths) / 100.0  # transform to proportion
        hh_ha_dist = np.column_stack((bins, vals))
        result[loc] = hh_ha_dist

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_household_head_age_by_size_distribution(location):
    """
    Calculate the contigency table, representing thr probabilities of each
    combination of household size and head age. The assumption is
    that household size and household head age are independent.

    If they are not independent, then we would need the joint distribution
    of these two variables which needs to be collected or estimated separately.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        household_head_age_distribution (float): N x 2: household head age bracket and proportion

    """
    import pandas as pd
    hh_size_dist     = get_household_size_distribution(location)
    hh_head_age_dist = get_household_head_age_distribution(location)

    df = pd.DataFrame(np.outer(hh_size_dist[:, 1], hh_head_age_dist[:, 1]),
                      columns=hh_head_age_dist[:, 0], index=hh_size_dist[:, 0])

    return df
