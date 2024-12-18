
"""
General utilities
"""
import os
import re
import numpy as np
import numba as nb
import pandas as pd

import sciris as sc
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['get_data_home', 'load_dataset', 'get_dataset_names']
__all__ += ['get_attr_vals']
__all__ += ['digitize_ages_1yr']
__all__ += ['test_cpu_performance']
__all__ += ['generate_unique_filename', 'to_df', 'promotetoiterable', 'generate_age_bin_labels']


@nb.njit(cache=True)
def digitize_ages_1yr(ages):
    """
    This function returns the indices of the 1-year age bins in the range
    (0, ss.options.max_age). The bin index is used as an integer representation
    of the agent's age.
    """
    # Create 1y age bins because ppl.age is a continuous variable
    age_cutoffs = np.arange(0, ss.options.max_age)
    return np.digitize(ages, age_cutoffs) - 1  # "rounds to the integer part of age"


@nb.njit(cache=True)
def digitize_ages(ages, age_group_lb):
    """
    This function returns the 0-based indices of the age bins passed in age_group_lb
    """
    return np.digitize(ages, age_group_lb) - 1  # returns 0-based indices of the group


def get_attr_vals(sim, attr_path, attr_name):
    """Get values of the attribute in attr_path"""
    attr = sim
    for attr_link in attr_path:
        attr = getattr(attr, attr_link)
    target_attr = attr_name
    vals = getattr(attr, target_attr)
    return vals


def get_data_home(data_home=None):
    """
    Return a path to the directory for default datasets.
    This function is needed by `load_dataset()`, and avoids the
    problem of using relative paths.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the `STARSIM_DATA` environment variable (if it exists)
    or otherwise it will use the default data directory.
    """

    if data_home is None:
        data_home = ss.options.data_home
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def get_dataset_names(data_home=None):
    """
    Provides a list of available datasets in data_home
    """
    data_home = get_data_home(data_home)
    dataset_names = os.listdir(data_home)
    dataset_names = list(filter(None, dataset_names))

    pattern = r'.*\.csv$'
    csv_files = [name for name in dataset_names if re.match(pattern, name)]

    output = f"Available datasets in {data_home}:\n"
    for name in csv_files:
        namestr = sc.colorize(f'  {name}\n', fg='yellow', output=True)
        output += f"{namestr}"
    print(output)
    return


def load_dataset(ds_name, data_home=None, **kwargs):
    """
    Load default dataset from starsim data directory.

    This function provides access to a small collection of datasets
    that are useful to set empirical distributions (ie, demographics, or
    gallstone probs by age and gender), rather than having those hardcoded
    in the code.

    The small datasets are expected to be simple tabular data saved in csv
    files. This function may apply some small amount of preprocessing, but it's
    not intended to be a full ingest and preprocessing pipeline. The csv files
    are expected to be in an already 'ingestable' form and simply loaded with
    pandas.read_csv().

    Use `get_dataset_names` to see a list of available datasets.

    Arguments:
      ds_name (str): name of the dataset (``{name}.csv``).
      data_home (str/Path, optional): the directory in which to cache data.
      kwargs: additional keyword arguments passed through to `pandas.read_csv`.

    Returns: data, type depends on dataset
       df(`pandas.DataFrame`): tabular data, with some preprocessing applied
       (depends on the dataset)

       or

       arr(`numpy.ndarray`): array data

    """

    filename = f"{ds_name}.csv"

    if data_home is None:
        data_home = get_data_home()

    data_path = os.path.join(data_home, filename)

    if not os.path.exists(data_path):
        get_dataset_names(data_home=data_home)
        raise ValueError(f"'{ds_name}' is not one of the existing datasets.")

    df = pd.read_csv(data_path, **kwargs)
    df = remove_empty_rows(df)

    match ds_name:
        case "gallstone_probs":
            arr = process_gallstone_data(df, coi="prob")
            return arr
        case "gallstone_prev":
            arr = process_gallstone_data(df, coi="prev")
            return arr
        case "prepatent_dur_dist_pars":
            complete_df = df.set_index('parameter').T.to_dict()
            return complete_df
        case _:
            mssg = (f"Unknown dataset {ds_name}. Known datasets are 'gallstone_probs',"
                    f"'gallstone_prev', and 'prepatent_dur_dist_pars'")
            raise ValueError(mssg)
    return


def remove_empty_rows(df):
    # Check for empty rows -- a rather common problem with csv files
    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]
    return df


def process_gallstone_data(df, coi="prob"):
    """
    Processes gallstone data from csv files.
    The files
    - gallstone_probs.csv and
    - gallstone_prev.csv are assumed to have the same structure.

    If age_lo and age_hi define an age bin > 1 year, then this function inflates
    the dataset to have a complete range of ages in 1yr bins.

    It then transforms the dataframe into an array that can be safely indexed
    with an integer version of agent ages.

    Arguments:
      df(`pandas.DataFrame`): dataframe with the data
      coi(str): column of interest in the dataframe, for gallstone probabilities
          is "prob"; for gallstone prevalence is "prev".

    Returns: data, type depends on dataset
       arr(`numpy.ndarray`): array data of size (ss.options.max_age x 2)
    """

    complete_df = pd.DataFrame({
        'age': np.concatenate(
            [np.arange(lo, hi if pd.notnull(hi) else ss.options.max_age) for lo, hi
             in df[['age_lo', 'age_hi']].values]),
        coi: np.repeat(df[coi].values,
                       df['age_hi'].fillna(ss.options.max_age).sub(
                           df['age_lo']).astype(int)),
        'sex': np.repeat(df['sex'].values,
                         df['age_hi'].fillna(ss.options.max_age).sub(
                             df['age_lo']).astype(int))
    })

    arr = complete_df.pivot(index='age',
                            columns='sex',
                            values=coi).fillna(0).to_numpy()
    return arr


def test_cpu_performance():
    """ Normalize performance across CPUs """
    t_bls = []
    bl_repeats = 50
    n_outer = 10
    n_inner = 1e6
    for r in range(bl_repeats):
        t0 = sc.tic()
        for i in range(n_outer):
            a = np.random.random(int(n_inner))
            b = np.random.random(int(n_inner))
            a * b
        t_bl = sc.toc(t0, output=True)
        t_bls.append(t_bl)
    t_bl = min(t_bls)
    return t_bl  # baseline performance in seconds


def generate_age_bin_labels(age_bins, inclusive_range=False):
    """
    Generates consistent age bin labels.

    This function creates age bin labels in the format of "start-end", and they
    define a semi-open interval of age [start, end) by default. If inclusive is
    True, the age bin labels are [start, end-1]

    Args:
        age_bins (list): A list of age bins given as single age values.
        inclusive_range (bool): whether the age bin labels should represent an inclusive
            age range [age_low, age_high] or a semi-open interval [age_low, age_high)

    Returns:
        list: A list of strings representing each age bin in the format
              "start-end".

    Example:
        >>> generate_age_bin_labels([10, 20, 30, 40])
        ["10-20", "20-30", "30-40"]
        >>> generate_age_bin_labels([10, 20, 30, 40], inclusive_range=True)
        ["10-19", "20-19", "30-39"]
    """
    if inclusive_range:
        offset = 1
    else:
        offset = 0
    age_bin_labels = [f"{age_bins[i]:.0f}-{age_bins[i + 1] - offset:.0f}" for i in range(len(age_bins) - 1)]
    if len(age_bin_labels) == 1:
        age_bin_labels = age_bin_labels[0]
    return age_bin_labels


def generate_unique_filename(root_str="starsim"):
    """
    Generate a unique filename -- useful when running lots of simulations, in a
    distributed manner

    Args:
        root_str (str): the string that will be used at the beginning of any filename
            generated by this function.

    Returns:
        filename (str): a unique filename string without file extension.

    """
    if root_str is None:
        root_str = ""
    now = sc.now(dateformat="%Y-%m-%d_%H-%M-%s")
    u_id = sc.uuid(which="hex")  # Creates a length-6 hex string
    filename = root_str + "_" + now + u_id
    return filename


def to_df(sim, sep='_'):
    """
    Export results as a Pandas dataframe. Available in newer starsim versions (ie, 2.0)
    Only saves 1D result arrays, discards the results from analyzers, unless they
    are 1D arrays.
    """
    if not sim.results_ready:  # pragma: no cover
        errormsg = 'Please run the sim before exporting the results'
        raise RuntimeError(errormsg)

    flat = sim.results.flatten(sep=sep)
    for arr in flat.keys():
        if len(flat[arr].shape) > 1 or arr.startswith("monitor"):
            del flat[arr]
    df = sc.dataframe.from_dict(flat)
    return df


def promotetoiterable(axs):
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    if not sc.isiterable(axs):
        axs = [axs]
    return axs
