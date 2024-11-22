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


__all__ = ["states_consistency_monitor", "histograms_by_age_sex_monitor"]


class Monitor(ss.Analyzer):
    """
    Base class for screening and triage.

    Args:
         product        (Product)       : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, period=None, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def downsample(self, mode=None):
        """"
        Implement temporal downsample of results
        This modifies the self.results ndict.
        """
        if mode is None:
            pass
        elif mode == "subsample":
            pass
        elif mode == "average":
            pass
        raise NotImplementedError(tyd.sorry_mssg)

    def plot(self):
        raise NotImplementedError(tyd.sorry_mssg)

    def to_df(self):
        raise NotImplementedError(tyd.sorry_mssg)


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
        mut_exc_1 = ~(typ.immune & typ.susceptible & typ.prepatent & typ.acute & typ.subclinical & typ.chronic & typ.recovered).any()
        mut_exc_2 = ~(typ.asymptomatic & typ.symptomatic).any()
        mut_exc_3 = ~(typ.susceptible & typ.infected).any()
        mut_exc_4 = ~(typ.immune & typ.infected).any()

        if not mut_exc_1:
            raise ValueError('Individual Boolean States should be mutually exclusive but are not.')

        if not mut_exc_2:
            raise ValueError('States Symptomatic and Asymptomatic should be mutually exclusive but are not.')

        if not mut_exc_3:
            raise ValueError('States Susceptible and Infected should be mutually exclusive but are not.')

        if not mut_exc_4:
            raise ValueError('States Immune and Infected should be mutually exclusive but are not.')

        # Collectively ehaustive
        coll_exh = (typ.immune | typ.susceptible | typ.prepatent | typ.acute | typ.subclinical | typ.chronic | typ.recovered | sim.people.dead).all()

        if not coll_exh:
            raise ValueError('Individual Boolean States should be collectively exhaustive but are not.')

        checkall = np.array([mut_exc_1, mut_exc_2, mut_exc_3, mut_exc_4, coll_exh])
        if not checkall.all():
            self.success = False
        return


class histograms_by_age_sex_monitor(Monitor):
    """
    Records statistics (counts) by age and sex for each timestep.
    By default, this analyzer records new cases for every time step.
    """
    def __init__(self, age_bins=None, age_bin_labels=None, to_record=None, record_from=None, record_until=None, modality="counts", name=None):
        super().__init__()
        self.name = "hist_by_age_sex" if name is None else name
        self.age_bins = age_bins
        self.age_bin_labels = age_bin_labels
        self.to_record = to_record
        self.record_from = record_from
        self.record_until = record_until
        self.record_modality = modality
        self.ti = 0
        self.ntpts = None  # Number of timepoints to record
        self.nags  = None  # Number of age groups to record
        self.yearvec = None # This monitor yearvec
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        self.ntpts = self.get_ntpts(sim)    # Get right number of timepoints
        self.nags = len(self.age_bins) - 1  # number of age groups

        if self.age_bin_labels is None:
            self.age_bin_labels = [f"{self.age_bins[i]:.0f}-{self.age_bins[i + 1] - 1:.0f}" for i in range(self.nags)]

        if self.to_record is None:
            states_of_interest = ['ti_infected', 'infected', 'ti_prepatent', 'prepatent',
                                  'ti_acute', 'acute', 'ti_subclinical', 'subclinical',
                                  'ti_chronic', 'chronic']
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
                self.results += [ss.Result(self.name, f"hist_m_{attrname}", (self.ntpts, self.nags),
                                           dtype=res_dtype, scale=False, label=f"m_{res_lbl}"),
                                 ss.Result(self.name, f"hist_f_{attrname}", (self.ntpts, self.nags),
                                           dtype=res_dtype, scale=False, label=f"f_{res_lbl}"),]
        return

    def get_ntpts(self, sim):
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
        self.yearvec = sc.inclusiverange(start_year, stop_year, sim.dt)
        ntpts = len(self.yearvec)

        # Update
        self.record_from = start_year
        self.record_until = stop_year
        return ntpts

    def get_attr_vals(self, sim, attr_path, attr_name):
        """Get values of the attribute in attr_path"""
        attr = sim
        for attr_link in attr_path:
            attr = getattr(attr, attr_link)
        target_attr = attr_name
        vals = getattr(attr, target_attr)
        return vals

    def record(self, f_vals, m_vals, attr_name):
        self.results[f"hist_m_{attr_name}"][self.ti, :] = m_vals
        self.results[f"hist_f_{attr_name}"][self.ti, :] = f_vals
        return

    def apply(self, sim):
        if sim.year >= self.record_from and (sim.year <= self.record_until):
            ti = sim.ti
            living_folks = sim.people.alive
            living_males = sim.people.male & living_folks
            living_femal = sim.people.female & living_folks

            for attrname, specs in sorted(self.to_record.items()):
                attrpath = specs["path"]
                vals = self.get_attr_vals(sim, attrpath, attrname)
                if attrname.startswith("ti_"):
                    f_uids = ((vals == ti) & living_femal).uids
                    m_uids = ((vals == ti) & living_males).uids
                else:
                    f_uids = (vals & living_femal).uids
                    m_uids = (vals & living_males).uids
                f_vals = np.histogram(sim.people.age[f_uids], bins=self.age_bins)[0]
                m_vals = np.histogram(sim.people.age[m_uids], bins=self.age_bins)[0]
                self.record(f_vals, m_vals, attrname)
            self.ti += 1
        return

    def finalize_results(self):
        super().finalize_results()
        if self.record_modality in ["proportion", "proportions", "props", "perc"]:
            for resname, vals in self.results.items():
                if not resname.endswith("_alive"):
                    # Proportion of people with attribute "resname" relative to the number of people in each age group
                    if "_f_" in resname:
                        denom = "hist_f_alive"
                    else:
                        denom = "hist_m_alive"
                    self.results[f"{resname}"] /= self.results[f"{denom}"]
                else:
                    # Proportion of people alive in each age group with respect to total population
                    self.results[f"hist_m_alive"] /= self.results[f"hist_m_alive"].sum(axis=1)[:, None]
                    self.results[f"hist_f_alive"] /= self.results[f"hist_f_alive"].sum(axis=1)[:, None]
        return
