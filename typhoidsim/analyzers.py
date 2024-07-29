"""
Analyzers specific to Typhoid.
"""

import numpy as np

import starsim as ss


__all__ = ["states_consistency"]


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


class prepatent_state_monitor(ss.Analyzer):
    """ Analyzer to track everything -- use for debug pruposes """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prepatent_state_monitor'
        self.success = True
        return

    def update_results(self, sim):
        return self.apply(sim)

    def apply(self, sim):
        """
        """
        typ = sim.diseases.typhoidsimeple

        prep_dur_acu = (typ.ti_acute - typ.ti_prepatent) * sim.dt
        prep_dur_sbl = (typ.ti_subclinical - typ.ti_prepatent) * sim.dt


        self.results['prep_durs'][sim.ti] = np.concatenate((prep_dur_acu, prep_dur_sbl), axis=0)
        self.results['prep_durs_acu'][sim.ti] = prep_dur_acu
        self.results['prep_durs_acu'][sim.ti] = prep_dur_acu

        return


class infectiousness(ss.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires = [ss.Typhoid]
        self.name = 'infectiousness_monitor'
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'infectiousness_levels', npts, dtype=float,
                      scale=True),
        ]
        return

    def apply(self, sim):
        ti = sim.ti
        return


class lifeof(ss.Analyzer):
    """
    Plots a schematic with the events of person.
    Stores a lot of data. Should be used for debugging purposes mainly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires = [ss.Typhoid]
        self.name = 'life_of_a_person'
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'label_me', npts, dtype=float,
                      scale=True),
        ]
        return

    def apply(self, sim):
        ti = sim.ti
        return

class natural_history(ss.Analyzer):
    """
    Provides statistics about the natural history of the disease:
    - proportion of people in each stage
    - mean duration of each stage
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires = [ss.Typhoid]
        self.name = 'natural_history'
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new_result', npts, dtype=float,
                      scale=True),
        ]
        return

    def apply(self, sim):
        ti = sim.ti
        return


class environmental_monitor(ss.Analyzer):
    """
    Monitors what's going on with the environment
    Maybe not needed
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires = [ss.Typhoid]
        self.name = 'natural_history'
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new_result', npts, dtype=float,
                      scale=True),
        ]
        return

    def apply(self, sim):
        ti = sim.ti
        return



# Debug Analyzers
# Examine prepatent only properties
# Examine infectiousness
# Examine infected states (acute/sublicnical)
# Examine chronic
# Examine environment properties
