"""
Analyzers specific to Typhoid.
"""

import numpy as np

import sciris as sc
import starsim as ss


__all__ = ["states_consistency", "age_histogram"]


class states_consistency(ss.Analyzer):
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
            breakpoint()
            raise ValueError('Individual Boolean States should be collectively exhaustive but are not.')

        checkall = np.array([mut_exc_1, mut_exc_2, mut_exc_3, mut_exc_4, coll_exh])
        if not checkall.all():
            self.success = False
        return


class age_histogram(ss.Analyzer):
    """
    Records age-specific statistics for each timestep.
    By default it records age-specific new cases for every time step.
    """
    def __init__(self, age_bins=None, age_bin_labels=None, to_record=None):
        super().__init__()
        self.name = "age_based_histogram"
        self.age_bins = age_bins
        self.age_bin_labels = age_bin_labels
        self.to_record = to_record
        self.target_attr_path = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = self.sim.npts
        nags = len(self.age_bins) - 1  # number of age groups
        self.results += [
            ss.Result(self.name, "age_histogram", (nags, npts), dtype=float,
                      scale=True),
        ]
        if self.to_record is not None and not self.to_record.startswith("ti_"):
            raise ValueError(f"This analyzers operates on event-tracking states that start with 'ti_'. "
                             f"Received {self.to_record}")

        if self.target_attr_path is None:
            self.target_attr_path = ["diseases", "typhoid", "ti_infected"]
        else:
            self.target_attr_path = ["diseases", "typhoid", self.to_record]

        if self.age_bin_labels is None:
            self.age_bin_labels = [f"{self.age_bins[i]:.0f}-{self.age_bins[i + 1] - 1:.0f}" for i in range(nags)]
        return

    def _get_target_arr(self, sim):
        """Get target values of an attribute that is an interable"""
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        vals = getattr(attr, target_attr)
        return vals

    def apply(self, sim):
        ti = sim.ti
        vals = self._get_target_arr(sim)
        uids = (vals == ti).uids
        ages = sim.people.age[uids]
        self.results.age_histogram[:, ti] = np.histogram(ages, bins=self.age_bins)[0]
        return

    def plot(self):
        pass
