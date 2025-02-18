"""
Define "passive" observation methods that do not interfere with the course
of a disease or with a simulation.

These classes are derived from starsim's Analyzers anyway because they need
to be executed in a specific part of the simulation workflow.

This module exists to emphasize a functional distinction between classes
that only subsamples and/or aggregates simulated data (monitors), and
classes that can optionally take as input empirical data and
perform additional calculations and be used as "components" or "steps" in
an optimization process.
"""

import numpy as np
import pandas as pd

import sciris as sc
import starsim as ss
import typhoidsim

import typhoidsim.defaults as tyd
import typhoidsim.utils as tyu

__all__ = ["states_consistency_monitor", "histograms_by_age_sex_monitor", "histogram_by_vaccination_status",
           "track_individuals_monitor"]


class Monitor(ss.Analyzer):
    """
    Base class for passive measurements / observation processes.
    """

    def __init__(self, period=None, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def plot(self):
        raise NotImplementedError(tyd.sorry_mssg)

    def to_df(self):
        raise NotImplementedError(tyd.sorry_mssg)


class track_individuals_monitor(Monitor):
    """
    A class used to record a certain state of a group of agents at each time
    step. This monitor is meant to be used for debugging purposes and with
    a population that does not change size, though agents could age.
    """

    def __init__(self, to_record=None, eligibility=None, eligibility_kwargs=None, name=None, **kwargs):
        super().__init__()
        self.to_record = to_record
        self.eligibility = eligibility
        self.eligibility_kwargs = eligibility_kwargs
        self.validate_records()
        self.validate_eligibility()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def init_results(self):
        super().init_results()
        results = [ss.Result(state, shape=(self.sim.t.npts, self.sim.pars["n_agents"]),
                             dtype=float, label=f"{state}") for state in self.to_record.keys()]
        self.define_results(*results)
        return

    def validate_records(self):
        if self.to_record is None:
            states_of_interest = ["rel_sus"]
            self.to_record = {state: dict(path=("diseases", "typhoid")) for state in
                              states_of_interest}
        return

    def validate_eligibility(self):
        import functools
        if self.eligibility is None and self.eligibility_kwargs is None:
            self.eligibility = self.default_eligibility

        if callable(self.eligibility):
            self.eligibility_kwargs = sc.mergedicts(self.eligibility_kwargs)
            self.eligibility = functools.partial(self.eligibility, **self.eligibility_kwargs)
        return

    @staticmethod
    def default_eligibility(sim, **kwargs):
        # By default only track properties of people alive
        is_alive = sim.people.alive
        return is_alive

    def check_eligibility(self):
        if callable(self.eligibility):
            return self.eligibility(self.sim)
        else:  # Assume self.eligibility is an array of uids
            return self.eligibility

    def step(self):
        sim = self.sim
        ti = sim.ti
        is_eligible = self.check_eligibility()
        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            self.results[attrname][ti, (is_eligible).uids] = vals[(is_eligible).uids]
            self.results[attrname][ti, (~is_eligible).uids] = np.nan
        return


    def plot_ridge(self, uids=None, num_agents=32, keys=None, y_scaling=1.0, style='fancy', fig_kw=None, plot_kw=None):
        """
        Plot each individual's timeseries of the given state.
        Individual traces are vertically stacked.

        Args:
            key (str): the results key to plot (by default, all, and can be a lot of figs, be warned)
            max_timepoints (int, optional): The maximum number of timepoints to plot, defaults to 16.
            style (str): the plotting style to use (default "fancy"; other options are "simple", None, or any Matplotlib style)
            fig_kw (dict): passed to ``plt.subplots()``
            plot_kw (dict): passed to ``plt.plot()``

        Returns:
            a list of figures
        """
        figs = []

        from scipy.stats import gaussian_kde

        # Configuration
        flat = self.results.flatten()
        n_cols = 1  # Number of columns of axes
        default_figsize = np.array([8, 6])
        figsize_factor = np.clip((n_cols - 3) / 6 + 1, 1,1.5)  # Scale the default figure size based on the number of rows and columns
        figsize = default_figsize * figsize_factor
        fig_kw = sc.mergedicts({'figsize': figsize}, fig_kw)
        plot_kw = sc.mergedicts({'lw': 2, 'y_scaling': 0.9}, plot_kw)

        if uids is None:
            uids = ss.uids(np.arange(num_agents))

        y_offsets = y_scaling * np.arange(len(uids))
        # Do the plotting
        with sc.options.with_style(style):
            if keys is not None:
                flat = {k: v for k, v in flat.items() if k.startswith(key)}
            for key, res in flat.items():
                fig, ax = sc.getrowscols(n=1, nrows=1, make=True, **fig_kw)
                ax.plot(res.timevec, res[:, uids]+y_offsets, color=[0.3, 0.3, 0.3])
                # Labels and annotations
                ax.set_xlim([res.timevec[0], res.timevec[-1]])
                ax.set_xlabel('Time')
                # Set the y-axis (time) labels
                #ax.set_yticks(y_scaling * np.arange(n_agents))
                #ax.set_yticklabels(yticklbls)
                #ax.set_ylabel('Y')
                title = getattr(res, 'label', key)
                ax.set_title(title)
                sc.figlayout(fig=fig)
                figs.append(fig)
        return figs


class histograms_by_age_sex_monitor(Monitor):
    """
    A class used to record statistics (counts) by age and sex for each timestep,
    or a user-defined sampling period.

    Args:
        age_bins (list, optional): The bins to use for age. Defaults to `None`.
        age_bin_labels (list, optional): Labels for the age bins. Defaults to `None`.
        to_record (dict): nested dictionary with the path to the quantity to record:
             For instance: dict(ti_acute=dict(path=("diseases", "typhoid"), label="cases"))
             If None (default), it records the main states of typhoid, both new cases in a given time step,
             and total number of people in that state at each timestep.
        record_from (float, optional): Time to start recording from, assumes it's time expressed in years in float representation.
             If None, it records from the start of the simulation.
        record_until (int, optional): Time until which to record. Assumes it's time expressed in years in float representation.
             If None, it records until the end of the simulation.
             The monitor records on semiopen interval [record_from, record_until),
             such that if we had two monitors, A: [record_from_a, record_until_a),
             and B: [record_from_b, record_until_b), and record_until_a == record_from_b,
             then that time point would not be counted twice.
        aggregate_sex (bool, optional): Whether to record each quantity separetely by sex.
             Defaults to `False`, ie, records the quantities in to_record separately for females and males.
        aggregate_time (str, optional): If the monitor downsamples results with respect to the original simulation resoliution,
            tell the monitor how to downsample. Options are "subsample", "mean", "median", "min", "max", "sum".
        resampling_period (float, optional): New sampling period of the output/results of this monitor.
        scaling (float, optional):  Scaling factor for adjusting the counts. Defaults to 1.0.
            This is a crude way to scale down counts to mimic imperfect observation/reporting processes such as testing.
        name (str, optional): The name of the monitor. Defaults to `None`.

    Note:
        Scaling factor value example: 0.6 * 0.75 * reporting_rate (emod parameter) is used
        for 60% blood culture sensitivity and 75% healthcare seeking in Pakistan simulations with
        EMOD. By default, scaling=1.0, as if we had perfect sampling of the whole population.
    """
    def __init__(self, age_bins=None, age_bin_labels=None, to_record=None, record_from=None,
                 record_until=None, aggregate_sex=False, aggregate_time=None, scaling=1.0,
                 resampling_period=None,
                 name=None):
        super().__init__()
        self.name = "monitor_by_age_sex" if name is None else name
        self.age_bins = sc.promotetoarray(age_bins) if age_bins is not None else np.array([0, tyd.max_age])
        self.age_bin_labels = age_bin_labels
        self.age_bin_centers = None
        self.age_bin_lbl_to_idx = None
        self.to_record = to_record
        self.record_from = record_from
        self.record_until = record_until
        self.scaling = scaling
        self.aggregate_sex = aggregate_sex
        self.aggregate_time = aggregate_time
        self.resampling_period = resampling_period
        self.tidx = 0
        # Attributes that will be set later
        self.monitor_step_size = None  # number of "dt" that fit in one resampling period
        self.monitor_period = None
        self.ntpts = None  # Number of timepoints to record
        self.nags  = None  # Number of age groups to record
        self.timevec_results = None  # This monitor timevec
        self.record = None
        self.agg_func = None
        self.sampling_fn = None
        self._apply = None
        self.stock_ntpts = None
        self.stocks = ss.Results(self.name)
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        self.set_observation_interval(sim)

        aggregation_functions = {
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
            "median": np.median,
            "sum": np.sum,
            "subsample": None
        }
        if self.resampling_period and not self.aggregate_time:
            raise ValueError("Provided a resampling period, but didn't specify how to aggregate over time. "
                             f"Available aggregation methods are: {list(aggregation_functions.keys())}")

        if self.aggregate_time in set(aggregation_functions):
            # integer number of timesteps we need to aggregate to downsample result arrays
            self.monitor_step_size = round(self.resampling_period / self.t.dt) # CK: TODO: use time units
            self.monitor_period = self.resampling_period
            self.agg_func = aggregation_functions.get(self.aggregate_time)
        else:
            self.monitor_step_size = 1
            self.monitor_period = self.t.dt  # CK: TODO: use time units

        self.start_point = sc.findnearest(sim.timevec - self.record_from, 0.0)
        self.end_point = sc.findnearest(sim.timevec - self.record_until, 0.0)
        self.timepoints = sc.inclusiverange(self.start_point, self.end_point - 1).astype(int)  # TODO: when integrating with self.t, check if -1 (minus 1 timestep) still needed.
        self.stock_ntpts = len(self.timepoints) # number of time points for the internal stock arrays  # CK: TODO: use time units

        if self.aggregate_time is None or self.aggregate_time == "subsample":
            self.sampling_fn = self._default_sampling
        else:
            self.sampling_fn = self._aggregate_sampling

        (ntpts, remainder) = divmod(self.stock_ntpts, self.monitor_step_size)

        self.ntpts = ntpts if not remainder else ntpts+1

        # Output year vector
        if not remainder:
            self.timevec_results = sc.inclusiverange(sim.timevec[self.timepoints[0]],
                                                     sim.timevec[self.timepoints[-1]], self.monitor_period) # CK: TODO: use time units
        else:
            self.timevec_results = sim.timevec[self.timepoints[::self.monitor_step_size]]


        self.nags = len(self.age_bins) - 1  # Number of age groups

        if self.age_bin_labels is None:
            self.age_bin_labels = tyu.generate_age_bin_labels(self.age_bins)

        # Save a mapping between human readable age bin label and column index in the results array
        self.age_bin_lbl_to_idx = {lbl: idx for idx, lbl in enumerate(self.age_bin_labels)}
        self.age_bin_centers = (self.age_bins[0:-1] + self.age_bins[1:])/2.0

        if self.to_record is None:
            states_of_interest = ["ti_infected", "infected",
                                  "ti_prepatent", "prepatent",
                                  "ti_acute", "acute",
                                  "ti_subclinical", "subclinical",
                                  "ti_chronic", "chronic",
                                  "ti_recovered", "recovered"]
            self.to_record = {state: dict(path=("diseases", "typhoid")) for state in states_of_interest}
            alive_dict = dict(alive=dict(path=("people",)))
            self.to_record.update(alive_dict)

        self.attrname_to_stockname = dict()
        for attrname, specs in self.to_record.items():
            if "path" not in specs:
                raise ValueError(f"Not be able to record {attrname} because 'path' is "
                                 f"missing the `to_record` configuration dictionary.")

            else:
                res_dtype = specs["path"] if "dtype" in specs else float
                if attrname.startswith("ti_"):
                    attrlbl = attrname.replace("ti_", "new_")
                else:
                    attrlbl = f"n_{attrname}"

                reslbl = specs["label"] if "label" in specs else attrlbl
                self.attrname_to_stockname[attrname] = attrlbl

                if self.aggregate_sex:
                    sexes = ["b"]   # aggregate both sexes
                else:
                    sexes = ["f", "m"]
                for sex in sexes:
                    self.stocks += [ss.Result(f"{sex}_{attrlbl}",
                                              dtype=res_dtype,
                                              shape=(self.stock_ntpts, self.nags),
                                              scale=False,
                                              timevec=sim.timevec[self.timepoints],
                                              label=f"{sex}_{reslbl}"), ]

                    self.results += [
                        ss.Result(f"{sex}_{attrlbl}",
                                  dtype=res_dtype,
                                  shape=(self.ntpts, self.nags),
                                  scale=True,
                                  timevec=self.timevec_results,
                                  label=f"{sex}_{reslbl}"), ]
        # Configure the monitor
        self.configure_recording_functions()
        return

    def configure_recording_functions(self):
        # Select which function should be used
        if self.aggregate_sex:
            self.record_fn = self._record_b
            self.count_fn = self._count_aggregated_sexes
        else:
            self.record_fn = self._record_fm
            self.count_fn = self._count_individual_sexes
        return

    def set_observation_interval(self, sim): # CK: TODO: use time units
        """ Set the correction endpoints of the observation period recorded by this monitor"""
        if self.record_from is None and self.record_until is None:
            start_year = sim.pars.start
            stop_year = sim.pars.stop
        elif self.record_from is not None and self.record_until is None:
            start_year = self.record_from if not self.record_from < sim.pars.start else sim.pars.start
            stop_year = sim.pars.stop
        elif self.record_from is None and self.record_until is not None:
            start_year = sim.pars.start
            stop_year = self.record_until if not self.record_until > sim.pars.stop else sim.pars.stop
        else:
            start_year = self.record_from
            stop_year = self.record_until
        # Update
        self.record_from = start_year
        self.record_until = stop_year
        return

    def _record_fm(self, f_vals, m_vals, stock_name):
        self.stocks[f"m_{stock_name}"][self.tidx, :] = m_vals
        self.stocks[f"f_{stock_name}"][self.tidx, :] = f_vals
        return

    def _record_b(self, b_vals, stock_name):
        self.stocks[f"b_{stock_name}"][self.tidx, :] = b_vals
        return

    def _count_individual_sexes(self, sim):
        ti = sim.ti
        living_folks = sim.people.alive
        living_males = sim.people.male & living_folks
        living_femal = sim.people.female & living_folks

        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            if attrname.startswith("ti_"):
                f_uids = ((vals == ti) & living_femal).uids
                m_uids = ((vals == ti) & living_males).uids
            else:
                f_uids = (vals & living_femal).uids
                m_uids = (vals & living_males).uids
            f_vals = self.scaling * \
                     np.histogram(sim.people.age[f_uids], bins=self.age_bins)[0]
            m_vals = self.scaling * \
                     np.histogram(sim.people.age[m_uids], bins=self.age_bins)[0]

            stockname = self.attrname_to_stockname[attrname]
            self.record_fn(f_vals, m_vals, stockname)
        return

    def _count_aggregated_sexes(self, sim):
        ti = sim.ti
        living_folks = sim.people.alive

        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            if attrname.startswith("ti_"):
                b_uids = ((vals == ti) & living_folks).uids
            else:
                b_uids = (vals & living_folks).uids
            b_vals = self.scaling * np.histogram(sim.people.age[b_uids], bins=self.age_bins)[0]
            stockname = self.attrname_to_stockname[attrname]
            self.record_fn(b_vals, stockname)
        return

    def _default_sampling(self, sim):
        if sim.ti % self.monitor_step_size == 0:
            self.count_fn(sim)
            self.tidx += 1
        return

    def _aggregate_sampling(self, sim):
        # Stores everything, then aggregates at the end
        self.count_fn(sim)
        self.tidx += 1
        return

    def aggregate_time_fn(self, vals):
        """ Aggregate time"""
        remainder = self.stock_ntpts % self.monitor_step_size
        reshaped_data = vals[:self.stock_ntpts - remainder].reshape(-1, self.monitor_step_size, self.nags)
        if not remainder:
            #  monitor
            arr = self.agg_func(reshaped_data, axis=1)
        else:
            downsampled_main = self.agg_func(reshaped_data, axis=1)
            if downsampled_main.shape[0] == self.ntpts:
                arr = downsampled_main
            else:
                # We have remainder data to handle
                remainder_data = vals[self.stock_ntpts - remainder:].reshape(-1, remainder, self.nags)
                downsampled_remainder = self.agg_func(remainder_data, axis=1)
                arr = np.vstack([downsampled_main, downsampled_remainder])
        return arr

    def report(self, vals):
        if self.aggregate_time is None:
            return vals
        return self.aggregate_time_fn(vals)

    def step(self):
        sim = self.sim
        if sim.ti in self.timepoints:
            self.sampling_fn(self.sim)
        return

    def finalize_results(self):
        for stock_name in self.stocks:
            self.results[stock_name][:] = self.report(self.stocks[stock_name][:]) if self.agg_func is not None else self.stocks[stock_name][:]
            # Repalce timevec with correct timevec for these results
            self.results.timevec = self.timevec_results
        super().finalize_results()
        return

    def to_df(self):
        """ Transform results to a pandas dataframe """
        #TODO: export year bin information too, will be useful for calibration
        dfs = []
        for res_name, res_value in self.results.items():
            if res_name in ["timevec"]:
                continue
            for ab_idx in range(res_value.shape[1]):
                x = res_value[:, ab_idx]
                data = {"label": pd.Series([res_name] * len(x)),
                        "x": x,
                        "age_bin_lb": pd.Series([self.age_bins[ab_idx]] * len(x)),   # Lower bound
                        "age_bin_ub": pd.Series([self.age_bins[ab_idx+1]] * len(x)),  # Upper bound
                        "age_bin_label": pd.Series([self.age_bin_labels[ab_idx]] * len(x)),
                        "time": res_value.timevec}
                breakpoint()
                dfs.append(pd.DataFrame(data))
        df = pd.concat(dfs, axis=0)
        return df

    def plot(self, key=None, t_index=None, fig=None, style='fancy', fig_kw=None, plot_kw=None):
        """
        Plot all results in the Sim object after the simulation has run

        Args:
            key (str): the results key to plot (by default, all)
            t_index (int): the time index in the monitor's timevec vector
            fig (Figure): if provided, plot results into an existing figure
            style (str): the plotting style to use (default "fancy"; other options are "simple", None, or any Matplotlib style)
            fig_kw (dict): passed to ``plt.subplots()``
            plot_kw (dict): passed to ``plt.plot()``
        """
        # Configuration
        flat = self.results.flatten()
        n_cols = np.ceil(np.sqrt(len(flat)))  # Number of columns of axes
        default_figsize = np.array([8, 6])
        figsize_factor = np.clip((n_cols - 3) / 6 + 1, 1,
                                 1.5)  # Scale the default figure size based on the number of rows and columns
        figsize = default_figsize * figsize_factor
        fig_kw = sc.mergedicts({'figsize': figsize}, fig_kw)
        plot_kw = sc.mergedicts({'lw': 2}, plot_kw)

        # Time vector
        timevec = self.timevec_results

        if t_index is None:
            n_tpts = np.min([len(timevec), 7])
            t_index = np.linspace(0, len(timevec) - 1, n_tpts, dtype=int)
        else:
            t_index = sc.promotetoarray(t_index)
        # Do the plotting
        with sc.options.with_style(style):
            if key is not None:
                flat = {k: v for k, v in flat.items() if k.startswith(key)}

            # Get the figure
            if fig is None:
                fig, axs = sc.getrowscols(len(flat), make=True, **fig_kw)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]

            xticks = []
            xticklabels = []

            for idx in range(self.nags):
                xticks.append(idx)
                if self.nags == 1:
                    xticklabels.append(self.age_bin_labels)
                else:
                    xticklabels.append(self.age_bin_labels[idx])

            # Do the plotting
            for ax, (key, res) in zip(axs, flat.items()):
                for tidx in sorted(t_index):
                    ax.bar(xticks, res[tidx, :], **plot_kw, label=f"t={timevec[tidx]:.4f}", alpha=0.2)

                title = getattr(res, 'label', key)
                if self.nags == 1:
                    ax.set_xlim([-0.5, 0.5])
                ax.set_title(title)
                ax.set_xticks(xticks, labels=xticklabels)
                ax.set_xlabel('Age (years)')
                ax.legend()

        sc.figlayout(fig=fig)
        return fig

    def plot_waterfall(self, key=None, max_timepoints=16, fig=None, style='fancy', fig_kw=None, plot_kw=None):
        """
        Plot a waterfall plot showing the evolution of the distribution of
        a given metric (ie, number of new acute cases) with respect to age.

        Args:
            key (str): the results key to plot (by default, all)
            max_timepoints (int, optional): The maximum number of timepoints to plot, defaults to 16.
            fig (Figure): if provided, plot results into an existing figure
            style (str): the plotting style to use (default "fancy"; other options are "simple", None, or any Matplotlib style)
            fig_kw (dict): passed to ``plt.subplots()``
            plot_kw (dict): passed to ``plt.plot()``

        Returns:
            figure handle

        The function generates uses kernel density estimation to visualize the data. If there's not data for the
        min max age specified, for a specific time step (ie, there are no agents in that age group), it adds a
        textbox. This is an edge case that can happen for a simulation with very few agents, and a very narrow
        age group.
        """
        if self.nags == 1:
            print("Waterfall plot not available for a single age group.")
            return

        from scipy.stats import gaussian_kde

        # Configuration
        flat = self.results.flatten()

        n_cols = np.ceil(np.sqrt(len(flat)))  # Number of columns of axes
        default_figsize = np.array([8, 6])
        figsize_factor = np.clip((n_cols - 3) / 6 + 1, 1,
                                 1.5)  # Scale the default figure size based on the number of rows and columns
        figsize = default_figsize * figsize_factor
        fig_kw = sc.mergedicts({'figsize': figsize}, fig_kw)
        plot_kw = sc.mergedicts({'lw': 2, 'y_scaling': 0.9}, plot_kw)

        n_tpts = np.min([self.ntpts, max_timepoints])
        t_indices = np.linspace(0, self.ntpts-1, n_tpts, dtype=int)

        # Time vector
        timevec = self.timevec_results
        y_scaling = plot_kw['y_scaling']

        # Do the plotting
        with sc.options.with_style(style):
            if key is not None:
                flat = {k: v for k, v in flat.items() if k.startswith(key)}

            # Get the figure
            if fig is None:
                fig, axs = sc.getrowscols(n=len(flat), nrows=1, make=True, **fig_kw)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]

            # Do the plotting
            for ax, (key, res) in zip(axs, flat.items()):
                # Loop through the selected time points and create kernel density estimates
                for idx, ti in enumerate(t_indices):
                    data_ti = res[ti, :]
                    try:
                        resamples = np.random.choice(self.age_bin_centers, size=self.nags * 100,
                                                     p=sc.safedivide(data_ti, data_ti.sum()))
                        kde = gaussian_kde(resamples)
                        kde_data = kde(self.age_bin_centers)
                        kde_data = kde_data / kde_data.max() + y_scaling * idx
                        data_ti = data_ti / data_ti.max()
                        ax.fill_between(self.age_bin_centers, y_scaling * idx, kde_data, color='#2f72de', alpha=0.3)
                        ax.bar(self.age_bin_centers, data_ti, bottom=y_scaling * idx, color='#2f72de', alpha=0.3, edgecolor="white")
                        ax.plot(self.age_bin_centers, kde_data, color='black', alpha=0.7)
                    except Exception as e:
                        pass

                # Labels and annotations
                ax.set_xlim([self.age_bins[0], self.age_bins[-1]])
                ax.set_xlabel('Age (years)')
                # Set the y-axis (time) labels
                ax.set_yticks(y_scaling * np.arange(len(t_indices)))
                yticklbls = [f"{t:.4f}" for t in timevec[t_indices]]
                ax.set_yticklabels(yticklbls)
                ax.set_ylabel('Year')
                title = getattr(res, 'label', key)
                ax.set_title(title)
#                ax.legend()
        sc.figlayout(fig=fig)
        return fig


class states_consistency_monitor(Monitor):
    """ Analyzer to track everything -- use for debug purposes """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'states_consistency'
        self.success = True
        return

    def update_results(self):
        return self.step()

    def step(self):
        """
        Checks states that should be mutually exlusive and collectively exhaustive
        """
        typ = self.sim.diseases.typhoid

        # Mutually exclusive estates
        mut_exc_1 = ~(typ.unexposed & typ.susceptible & typ.prepatent & typ.acute &
                      typ.subclinical & typ.chronic & typ.recovered
                      ).any()
        mut_exc_2 = ~(typ.asymptomatic & typ.symptomatic).any()
        mut_exc_3 = ~(typ.susceptible & typ.infected).any()
        mut_exc_4 = ~(typ.unexposed & typ.infected).any()

        if not mut_exc_1:
            raise ValueError(
                'Individual Boolean States should be mutually exclusive but are not.')

        if not mut_exc_2:
            raise ValueError(
                'States Symptomatic and Asymptomatic should be mutually exclusive but are not.')

        if not mut_exc_3:
            raise ValueError(
                'States Susceptible and Infected should be mutually exclusive but are not.')

        if not mut_exc_4:
            raise ValueError(
                'States Immune and Infected should be mutually exclusive but are not.')

        # Collectively exhaustive
        coll_exh = (typ.unexposed | typ.susceptible | typ.prepatent | typ.acute |
                    typ.subclinical | typ.chronic | typ.recovered | self.sim.people.dead
                    ).all()

        if not coll_exh:
            raise ValueError(
                'Individual Boolean States should be collectively exhaustive but are not.')

        checkall = np.array(
            [mut_exc_1, mut_exc_2, mut_exc_3, mut_exc_4, coll_exh])
        if not checkall.all():
            self.success = False
        return


class histogram_by_vaccination_status(histograms_by_age_sex_monitor):
    """
    This class exist to avoid slowing down the parent class if there are no
    vaccination interventions.
    """
    def __init__(self, track_vaccinated=True, **kwargs):
        super().__init__(**kwargs)  # keyword arguments passed to histograms
        self.track_vaccinated = track_vaccinated
        self.vax_interventions = None
        self.vax_state_a = ss.BoolArr('vax_state_a', default=False)
        self.vax_state_b = ss.BoolArr('vax_state_b', default=False)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        # Get any sim.interventions that have an attribute "vaccinated"
        self.vax_interventions = []
        for intervention_name in sim.interventions:
            if hasattr(sim.interventions[intervention_name], "vaccinated"):
                self.vax_interventions.append(intervention_name)

    def _count_individual_sexes(self, sim):
        ti = sim.ti
        eligible_males = sim.people.alive & sim.people.male
        eligible_females = sim.people.alive & sim.people.female

        self.vax_state_a[eligible_males.uids] = False  # Reset tracking state to False
        self.vax_state_b[eligible_females.uids] = False  # Reset tracking state to False

        for vax_interv in self.vax_interventions:
            # Find if agent has received any vaccination across all possible vax interventions
            self.vax_state_a[:] = self.vax_state_a[:] | \
                                  [~sim.interventions[vax_interv].vaccinated,
                                    sim.interventions[vax_interv].vaccinated][self.track_vaccinated]  # If False tracks unvaccinated, if True tracks vaccinated

            self.vax_state_b[:] = self.vax_state_b[:] | \
                                  [~sim.interventions[vax_interv].vaccinated,
                                   sim.interventions[vax_interv].vaccinated][
                                      self.track_vaccinated]  # If False tracks unvaccinated, if True tracks vaccinated

        eligible_males = eligible_males & self.vax_state_a
        eligible_females = eligible_females & self.vax_state_b


        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            if attrname.startswith("ti_"):
                f_uids = ((vals == ti) & eligible_females).uids
                m_uids = ((vals == ti) & eligible_males).uids
            else:
                f_uids = (vals & eligible_females).uids
                m_uids = (vals & eligible_males).uids
            f_vals = self.scaling * \
                     np.histogram(sim.people.age[f_uids], bins=self.age_bins)[0]
            m_vals = self.scaling * \
                     np.histogram(sim.people.age[m_uids], bins=self.age_bins)[0]

            stockname = self.attrname_to_stockname[attrname]
            self.record_fn(f_vals, m_vals, stockname)
        return

    def _count_aggregated_sexes(self, sim):
        ti = sim.ti
        eligible_folks = sim.people.alive
        self.vax_state_a[:] = False  # Reset tracking state to False
        for vax_interv in self.vax_interventions:
            # Find if agent has received any vaccination across all possible vax interventions
            vax_status  = [~sim.interventions[vax_interv].vaccinated, sim.interventions[vax_interv].vaccinated][self.track_vaccinated]
            self.vax_state_a[:] = self.vax_state_a[:] | vax_status # union of all interventions (an agent could have received two vaccines, but we count them once)
        eligible_folks = eligible_folks & self.vax_state_a

        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            if attrname.startswith("ti_"):
                b_uids = ((vals == ti) & eligible_folks).uids
            else:
                b_uids = (vals & eligible_folks).uids
            b_vals = self.scaling * \
                     np.histogram(sim.people.age[b_uids], bins=self.age_bins)[0]
            stockname = self.attrname_to_stockname[attrname]
            self.record_fn(b_vals, stockname)
        return

    def plot(self, **plot_kwargs):
        fig = super().plot(**plot_kwargs)
        fig_title = ["Unvaccinated", "Vaccinated"][self.track_vaccinated]
        fig.suptitle(fig_title)
        return fig

    def plot_waterfall(self, **plot_waterfall_kwargs):
        fig = super().plot_waterfall(**plot_waterfall_kwargs)
        fig_title = ["Unvaccinated", "Vaccinated"][self.track_vaccinated]
        fig.suptitle(fig_title)
        return fig
