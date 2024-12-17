
"""
A collection of useful plotting functions that could eventually be part of some
class such as People, Network or Analyzer.
"""
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import seaborn as sns

from .networks import CommunityNet
import starsim as ss

__all__ = ["plot_age_histogram", "plot_age_mixing", "plot_sim", "plot_calib"]

def plot_sim(sim, key=None, fig=None, style='fancy', fig_kw=None, plot_kw=None, display_from=None, display_until=None):
    """
    Plot all results in the Sim object after the simulation has run

    Args:
        sim (ss.Sim): the simulation object
        key (str): the results key to plot (by default, all)
        fig (Figure): if provided, plot results into an existing figure
        style (str): the plotting style to use (default "fancy"; other options are "simple", None, or any Matplotlib style)
        fig_kw (dict): passed to ``plt.subplots()``
        plot_kw (dict): passed to ``plt.plot()``
        display_from (float): start time for display
        display_until (float): end time for display
    """
    sim.check_results_ready('Please run the sim before plotting')

    # Configuration
    flat = sim.results.flatten()
    n_cols = np.ceil(np.sqrt(len(flat)))  # Number of columns of axes
    default_figsize = np.array([8, 6])
    figsize_factor = np.clip((n_cols - 3) / 6 + 1, 1, 1.5)
    figsize = default_figsize * figsize_factor
    fig_kw = sc.mergedicts({'figsize': figsize}, fig_kw)
    plot_kw = sc.mergedicts({'lw': 2}, plot_kw)

    # Do the plotting
    with sc.options.with_style(style):
        time_masks = [
            sim.t.yearvec >= display_from if display_from is not None else True,
            sim.t.yearvec <= display_until if display_until is not None else True]
        mask = np.logical_and(*time_masks) if time_masks else Ellipsis

        if key is not None:
            flat = {k: v for k, v in flat.items() if k.startswith(key)}

        if fig is None:
            fig, axs = sc.getrowscols(len(flat), make=True, **fig_kw)
            if isinstance(axs, np.ndarray):
                axs = axs.flatten()
        else:
            axs = fig.axes
        if not sc.isiterable(axs):
            axs = [axs]

        for ax, (key, res) in zip(axs, flat.items()):
            ax.plot(sim.t.yearvec[mask], res[mask], **plot_kw, label=sim.label)
            ax.set_title(res.full_label)
            ax.set_xlabel('Year')

    sc.figlayout(fig=fig)

    return ss.return_fig(fig)

def plot_calib(calib, fig=None, style='fancy', fig_kw=None, plot_kw=None):
    """
    Plot all results in the Calibration object after check_fit() has been run

    Args:
        calib (ss.Calibration): a Calibration object for which check_fit() has been run
        fig (Figure): if provided, plot results into an existing figure
        style (str): the plotting style to use (default "fancy"; other options are "simple", None, or any Matplotlib style)
        fig_kw (dict): passed to ``plt.subplots()``
        plot_kw (dict): passed to ``plt.plot()``
    """
    if calib.after_msim is None:
        print("Please run calib.check_fit()")
        return

    df = calib.to_df()

    # Configuration
    n_cols = np.ceil(np.sqrt(df["t"].nunique()))  # Number of columns of axes
    default_figsize = np.array([8, 6])
    figsize_factor = np.clip((n_cols - 3) / 6 + 1, 1, 1.5)
    figsize = default_figsize * figsize_factor
    fig_kw = sc.mergedicts({'figsize': figsize}, fig_kw)
    plot_kw = sc.mergedicts({'lw': 2}, plot_kw)

    # Do the plotting
    with sc.options.with_style(style):
        if fig is None:
            n = np.array([df["t"].nunique()])
            fig, axs = sc.getrowscols(n, make=True, remove_extra=True, **fig_kw)
            if isinstance(axs, np.ndarray):
                axs = axs.flatten()
        else:
            axs = fig.axes
        if not sc.isiterable(axs):
            axs = [axs]

        for ax, t in zip(axs, df["t"].unique()):
            sim_data = df.loc[df["t"] == t, :]
            sim_data["sort_age"] = sim_data['age_bin'].apply(lambda x: int(x.split('-')[0]))
            sim_data.sort_values("sort_age", inplace=True)
            ax = sns.barplot(data=sim_data, x="age_bin", y="x", hue="source_data", estimator="mean", errorbar="sd", ax=ax)
            component_name = df["component_name"].unique()[0]
            ax.set_title(f"{component_name} - year: {t}")
            ax.set_xlabel('Year')
    sc.figlayout(fig=fig)

    return ss.return_fig(fig)

def plot_age_histogram(people, bins=None, width=1.0, alpha=0.6, fig_args=None, axis_args=None, plot_args=None, fig=None):
    """
    Plot population age distribution

    Args:
        people (starsim People): the people object. Must be initialized.
        bins (arr): age bins to use (default, 0-100 in one-year bins)
        width (float): bar width
        alpha (float): transparency of the plots
        fig_args (dict): passed to plt.figure()
        axis_args (dict): passed to plt.subplots_adjust()
        plot_args (dict): passed to plt.plot()
        fig (fig): handle of existing figure to plot into

    Returns:
        fig (fig): handle of figure where data has been plotted
    """
    if not people.initialized:
        raise ValueError("People must have been initialized via a simulation.")

    # Handle inputs
    if bins is None:
        bins = np.arange(0, 101)

    # Set defaults
    color = [0.1, 0.1, 0.1]
    n_rows = 1
    offset = 0.5
    gridspace = 10
    zorder = 10

    # Handle other arguments
    fig_args = sc.mergedicts(dict(figsize=(18, 11)), fig_args)
    axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
    plot_args = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)

    # Compute statistics
    edges = np.append(bins, np.inf)
    age_counts = np.histogram(people.age, edges)[0]

    # Create the figure
    if fig is None:
        fig = plt.figure(**fig_args)
    plt.subplots_adjust(**axis_args)

    # Plot age histogram
    plt.subplot(n_rows, 2, 1)
    plt.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
    plt.xlim([min(bins)-offset, max(bins)+offset])
    plt.xticks(np.arange(0, max(bins)+1, gridspace))
    plt.grid(True)
    plt.xlabel('Age (years)')
    plt.ylabel('Number of people (agents)')
    plt.title(f'Age distribution ({len(people):n} agents total)')

    # Plot cumulative distribution
    plt.subplot(n_rows, 2, 2)
    age_sorted = sorted(people.age)
    y = np.linspace(0, 100, len(age_sorted))
    plt.plot(age_sorted, y, '-', **plot_args)
    plt.xlim([0, max(bins)])
    plt.ylim([0, 100])
    plt.xticks(np.arange(0, max(bins)+1, gridspace))
    plt.yticks(np.arange(0, 101, gridspace))
    plt.grid(True)
    plt.xlabel('Age')
    plt.ylabel('Cumulative proportion (%)')
    plt.title(f'Cumulative age distribution (mean age: {people.age.mean():0.2f} years)')
    return ss.return_fig(fig)

def plot_age_mixing(network, fig_args=None, axis_args=None, fig=None):
    """
    Plot the empirical age mixing matrix and optionally the
    simulated age mixing matrix.

    Args:
        network (starsim Network): the network object, must be initialized and must be an instance of CommunityNet.
        fig_args (dict): passed to plt.figure()
        axis_args (dict): passed to plt.subplots_adjust()
        fig (fig): handle of existing figure to plot into

    Returns:
        fig (fig): handle of figure where data has been plotted
    """
    if not network.initialized:
        raise ValueError("Network must have been initialized via a simulation.")

    if not isinstance(network, CommunityNet):
        raise TypeError(f"Network must be an instance of CommunityNet. Got {type(network)}")

    # Set defaults
    fig_args = sc.mergedicts(dict(figsize=(18, 11)), fig_args)
    axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)

    # Create the figure
    if fig is None:
        fig = plt.figure(**fig_args)

    plt.subplots_adjust(**axis_args)
    plt.subplot(1, 1, 1)

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
    return ss.return_fig(fig)
