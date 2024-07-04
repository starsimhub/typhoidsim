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
        typ = sim.diseases.typhoidsimple

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


# Debug Analyzers
# Examine prepatent only properties
# Examine infectiousness
# Examine infected states (acute/sublicnical)
# Examine chronic
# Examine environment properties
