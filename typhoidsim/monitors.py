"""
Define "passive" observation methods that do not interfere with the course
of a disease or with a simulation.

These classes are derived from starsim's Analyzers anyway because they neeed
to be executed in a specific part of the simulation workflow.

This module exists to emphasise a functional distinction between classes
that only subsamples and/or aggregates simulated data (monitors), and
classes that can optionally take as input empirical data and
perform additional calculations and be used as "components" or "steps" in
an optimisation process.
"""

import numpy as np

import sciris as sc
import starsim as ss

import typhoidsim.defaults as tyd
import typhoidsim.utils as tyu


__all__ = ["states_consistency_monitor", "histograms_by_age_sex_monitor"]


class Monitor(ss.Analyzer):
    """
    Base class for passive measurments / observation processes.

    Args:
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


class histograms_by_age_sex_monitor(Monitor):
    """
    Records statistics (counts) by age and sex for each timestep.
    By default, this analyzer records new cases for every time step.

    # A number to adjust the number of cases for imperfect sampling/testing
    in the real-world

    For instance in the Pakistan simulations with EMOD a value of 0.6 * 0.75 *
    reporting_rate (emod parameter) is used 60% blood culture sensitivity
    and 75% health care seeking. By default scaling=1.0, as if we had
    perfect sampling of the whole population.
    """
    def __init__(self, age_bins=None, age_bin_labels=None, to_record=None, record_from=None,
                 record_until=None, aggregate_sex=False, aggregate_time=None, scaling=1.0,
                 resampling_period=None,
                 name=None):
        super().__init__()
        self.name = "hist_by_age_sex" if name is None else name
        self.age_bins = age_bins
        self.age_bin_labels = age_bin_labels
        self.to_record = to_record
        self.record_from = record_from
        self.record_until = record_until
        self.scaling = scaling
        self.aggregate_sex = aggregate_sex
        self.aggregate_time = aggregate_time
        self.resampling_period = resampling_period
        self.ti = 0
        # Attributes that will be set later
        self.monitor_step = None  # number of "dt" that fit in one resampling period
        self.monitor_period = None
        self.ntpts = None  # Number of timepoints to record
        self.nags  = None  # Number of age groups to record
        self.yearvec = None  # This monitor yearvec
        self.record = None
        self.agg_func = None
        self.sample = None
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
        if self.aggregate_time in set(aggregation_functions):
            self.monitor_step = round(self.resampling_period/sim.dt)
            self.monitor_period = self.resampling_period
            self.agg_func = aggregation_functions.get(self.aggregate_time)
        else:
            self.monitor_step = 1
            self.monitor_period = sim.dt

        # Output year vector
        self.yearvec = sc.inclusiverange(self.record_from, self.record_until, self.monitor_period)

        if self.aggregate_time is None or self.aggregate_time == "subsample":
            self.sample = self._default_sampling
            self.ntpts = len(self.yearvec)
            self.stock_ntpts = len(self.yearvec)
        else:
            self.sample = self._aggregate_sampling
            self.ntpts = len(self.yearvec)
            self.stock_ntpts = len(sc.inclusiverange(self.record_from, self.record_until, sim.dt))

        self.nags = len(self.age_bins) - 1  # Number of age groups

        if self.age_bin_labels is None:
            self.age_bin_labels = [f"{self.age_bins[i]:.0f}-{self.age_bins[i + 1] - 1:.0f}" for i in range(self.nags)]

        if self.to_record is None:
            states_of_interest = ["ti_infected", "infected",
                                  "ti_prepatent", "prepatent",
                                  "ti_acute", "acute", "ti_subclinical", "subclinical",
                                  "ti_chronic", "chronic",
                                  "ti_recovered", "recovered"]
            self.to_record = {state: dict(path=("diseases", "typhoid")) for state in states_of_interest}
            alive_dict = dict(alive=dict(path=("people",)))
            self.to_record.update(alive_dict)
        else:
            # Add alive as we need this to enable proportions
            if "alive" not in self.to_record.keys():
                alive_dict = dict(alive=dict(path=("people",)))
                self.to_record.update(alive_dict)

        for attrname, specs in self.to_record.items():
            if "path" not in specs:
                raise ValueError(f"Will not be able to record {attrname} because 'path' is "
                                 f"missing the `to_record` configuration dictionary.")

            else:
                res_dtype = specs["path"] if "dtype" in specs else float
                res_lbl   = specs["label"] if "label" in specs else attrname
                if self.aggregate_sex:
                    sexes = ["b"]   # aggregate both sexes
                else:
                    sexes = ["f", "m"]
                for sex in sexes:
                    self.stocks += [ss.Result(self.name, f"hist_{sex}_{attrname}",
                                               (self.stock_ntpts, self.nags), dtype=res_dtype,
                                               scale=False, label=f"m_{res_lbl}"),]

                    self.results += [
                        ss.Result(self.name, f"hist_{sex}_{attrname}",
                                  (self.ntpts, self.nags), dtype=res_dtype,
                                  scale=False, label=f"m_{res_lbl}"), ]
        if self.aggregate_sex:
            self.record = self._record_b
            self._apply = self._apply_aggregated_sexes
        else:
            self.record = self._record_fm
            self._apply = self._apply_individual_sexes
        return

    def set_observation_interval(self, sim):
        """ Get thr gith number of timepoints for the Result arrays"""
        if self.record_from is None and self.record_until is None:
            start_year = sim.pars.start
            stop_year = sim.pars.end
        elif self.record_from is not None and self.record_until is None:
            start_year = self.record_from
            stop_year = sim.pars.end
        elif self.record_from is None and self.record_until is not None:
            start_year = sim.pars.start
            stop_year = self.record_until
        else:
            start_year = self.record_from
            stop_year = self.record_until
        # Update
        self.record_from = start_year
        self.record_until = stop_year
        return

    def _record_fm(self, f_vals, m_vals, attr_name):
        self.stocks[f"hist_m_{attr_name}"][self.ti, :] = m_vals
        self.stocks[f"hist_f_{attr_name}"][self.ti, :] = f_vals
        return

    def _record_b(self, b_vals, attr_name):
        self.stocks[f"hist_b_{attr_name}"][self.ti, :] = b_vals
        return

    def _apply_individual_sexes(self, sim):
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
            self.record(f_vals, m_vals, attrname)
        return

    def _apply_aggregated_sexes(self, sim):
        ti = sim.ti
        living_folks = sim.people.alive

        for attrname, specs in sorted(self.to_record.items()):
            attrpath = specs["path"]
            vals = tyu.get_attr_vals(sim, attrpath, attrname)
            if attrname.startswith("ti_"):
                b_uids = ((vals == ti) & living_folks).uids
            else:
                b_uids = (vals & living_folks).uids
            b_vals = self.scaling * \
                     np.histogram(sim.people.age[b_uids], bins=self.age_bins)[0]
            self.record(b_vals, attrname)
        return

    def _default_sampling(self, sim):
        if sim.ti % self.monitor_step == 0:
            self._apply(sim)
            self.ti += 1
        return

    def _aggregate_sampling(self, sim):
        # Stores everything, then aggregates at the end
        self._apply(sim)
        self.ti += 1
        return

    def aggregate(self, vals):
        if self.aggregate_time is None:
            return vals
        remainder = self.ntpts % self.monitor_step
        reshaped_data = vals[:self.ntpts - remainder].reshape(-1, self.monitor_step, self.nags)
        if remainder != 0:
            downsampled_main = self.agg_func(reshaped_data, axis=1)
            downsampled_remainder = self.agg_func(vals[-remainder:], axis=0)
            return np.vstack([downsampled_main, downsampled_remainder[None, :]])
        else:
            return self.agg_func(reshaped_data, axis=1)

    def apply(self, sim):
        if sim.year >= self.record_from and (sim.year <= self.record_until):
            self.sample(sim)
        return

    def finalize_results(self):
        super().finalize_results()
        for stock_name in self.stocks:
            self.results[stock_name][:] = self.aggregate(self.stocks[stock_name][:]) if self.agg_func is not None else self.stocks[stock_name][:]
        return


class states_consistency_monitor(Monitor):
    """ Analyzer to track everything -- use for debug pruposes """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'states_consistency'
        self.success = True
        return

    def update_results(self, sim):
        return self.apply(sim)

    def apply(self, sim):
        """
        Checks states that should be mutually exlusive and collectively exhaustive
        """
        typ = sim.diseases.typhoid

        # Mutually exclusive estates
        mut_exc_1 = ~(typ.immune & typ.susceptible & typ.prepatent & typ.acute &
                      typ.subclinical & typ.chronic & typ.recovered
                      ).any()
        mut_exc_2 = ~(typ.asymptomatic & typ.symptomatic).any()
        mut_exc_3 = ~(typ.susceptible & typ.infected).any()
        mut_exc_4 = ~(typ.immune & typ.infected).any()

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

        # Collectively ehaustive
        coll_exh = (typ.immune | typ.susceptible | typ.prepatent | typ.acute |
                    typ.subclinical | typ.chronic | typ.recovered | sim.people.dead
                    ).all()

        if not coll_exh:
            raise ValueError(
                'Individual Boolean States should be collectively exhaustive but are not.')

        checkall = np.array(
            [mut_exc_1, mut_exc_2, mut_exc_3, mut_exc_4, coll_exh])
        if not checkall.all():
            self.success = False
        return
