"""
Functionality related to ingesting data from files. Ingesting encompasses
loading data from files, parsing data and reformatting data, and processing
so they tcan be consumed by typhoidsim or starsim.
"""
import numpy as np

import sciris as sc

from .data import country_household_size_distribution as household_size_distribution

__all__ = ["get_age_distribution", "get_household_size",
           "get_household_size_distribution"]


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
        house_size (float): Size of household, or dict if multiple locations

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
        vals = np.repeat(hhs_data[:, 2] / bin_widths, bin_widths)
        hh_size_dist = np.column_stack((bins, vals))
        result[loc] = hh_size_dist

    if len(result) == 1:
        result = list(result.values())[0]

    return result
