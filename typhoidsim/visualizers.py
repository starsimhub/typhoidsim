"""
A collection of useful plotting functions that could eventually be part of some
class such as People, Network or Analyzer.
"""
import numpy as np
import matplotlib.pyplot as plt

import sciris as sc

from .networks import CommunityNet

__all__ = ["plot_age_histogram", "plot_age_mixing"]


def plot_age_histogram(people, bins=None, width=1.0, alpha=0.6,
                       fig_args=None, axis_args=None, plot_args=None, fig=None):
    """
    Plot population age distribution

    Args:
        people       (starsim People): the people object. Must be intialized.
        bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
        width     (float) : bar width
        alpha     (float) : transparency of the plots
        fig_args  (dict)  : passed to pl.figure()
        axis_args (dict)  : passed to pl.subplots_adjust()
        plot_args (dict)  : passed to pl.plot()
        fig       (fig)   : handle of existing figure to plot into

    Returns:
        fig       (fig)   : handle of figure where data has been plotted
    """

    if not people.initialized:
        ValueError("People must have been initialized via a simulation.")

    # Handle inputs
    if bins is None:
        bins = np.arange(0, 101)

    # Set defaults
    color     = [0.1, 0.1, 0.1]  # Color for the age distribution
    n_rows    = 1    # Number of rows of plots
    offset    = 0.5  # For ensuring the full bars show up
    gridspace = 10   # Spacing of gridlines
    zorder    = 10   # So plots appear on top of gridlines

    # Handle other arguments
    fig_args  = sc.mergedicts(dict(figsize=(18 ,11)), fig_args)
    axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
    plot_args = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)

    # Compute statistics
    min_age = min(bins)
    max_age = max(bins)
    edges = np.append(bins, np.inf)  # Add an extra bin to end to turn them into edges
    age_counts = np.histogram(people.age, edges)[0]

    # Create the figure
    if fig is None:
        fig = plt.figure(**fig_args)
    plt.subplots_adjust(**axis_args)

    # Plot age histogram
    plt.subplot(n_rows, 2, 1)
    plt.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
    plt.xlim([min_age-offset, max_age+offset])
    plt.xticks(np.arange(0, max_age+1, gridspace))
    plt.grid(True)
    plt.xlabel('Age (years)')
    plt.ylabel('Number of people (agents)')
    plt.title(f'Age distribution ({len(people):n} agents total)')

    # Plot cumulative distribution
    plt.subplot(n_rows, 2, 2)
    age_sorted = sorted(people.age)
    y = np.linspace(0, 100, len(age_sorted))
    plt.plot(age_sorted, y, '-', **plot_args)
    plt.xlim([0, max_age])
    plt.ylim([0, 100])  # Percentage
    plt.xticks(np.arange(0, max_age+1, gridspace))
    plt.yticks(np.arange(0, 101, gridspace))  # Percentage
    plt.grid(True)
    plt.xlabel('Age')
    plt.ylabel('Cumulative proportion (%)')
    plt.title(f'Cumulative age distribution (mean age: {people.age.mean():0.2f} years)')
    return fig


def plot_age_mixing(network, fig_args=None, axis_args=None, fig=None):

    """
    Plot the empirical age mixing matrix and optionally the
    simulated age mixing matrix.

    Args:
        location (str)    : the geographical location of the age mixing we want to plot
        network   (starsim Network): the network object,must be intialized and must be an instance of CommunityNet.
        fig_args  (dict)  : passed to pl.figure()
        axis_args (dict)  : passed to pl.subplots_adjust()
        fig       (fig)   : handle of existing figure to plot into

    Returns:
        fig       (fig)   : handle of figure where data has been plotted
    """


    if not network.initialized:
        ValueError("Newtork must have been initialized via a simulation.")

    if not isinstance(network, CommunityNet):
        TypeError(f"Network must be an instance of CommunityNet. Got {type(network)}")

    # Set defaults
    color = [0.1, 0.1, 0.1]  # Color for the age distribution
    n_rows = 1  # Number of rows of plots

    # Handle other arguments
    fig_args = sc.mergedicts(dict(figsize=(18, 11)), fig_args)
    axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)

    # Create the figure
    if fig is None:
        fig = plt.figure(**fig_args)

    plt.subplots_adjust(**axis_args)
    plt.subplot(n_rows, 1, 1)

    # Plot the density
    im = plt.imshow(np.rot90(network.age_mix_matrix_probs), cmap=plt.cm.Blues, vmin=0.0, vmax=0.5)
    cb = plt.colorbar(im)
    plt.xticks(np.arange(network.num_age_groups),
               network.pars.age_mixing['age_group'],
               rotation=30)
    plt.yticks(np.arange(network.num_age_groups),
               network.pars.age_mixing['age_group'][::-1],
               rotation=30)

    plt.xlabel('Age of individual (years)')
    plt.ylabel('Age of contact (years)')
    plt.suptitle('Empirical age mixing density')

    fig_net = network.plot_age_mixing_density()
    return fig
