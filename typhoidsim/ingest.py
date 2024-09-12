"""
Functionality related to ingesting data from files. Ingesting encompasses
loading data from files, parsing data and reformatting data, and processing
so they tcan be consumed by typhoidsim or starsim.
"""

__all__ = ['get_age_distribution', 'get_household_distribution']


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
