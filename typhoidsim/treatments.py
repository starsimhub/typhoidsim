"""
Define Typhoid-specific treatments (interventions)
"""

import starsim as ss
import sciris as sc
import numpy as np
from .typhoid import TyphoidSimple

__all__ = ['basic_treatment']


class ViVax(ss.Vx):
    """ Vaccine product """

    def __init__(self, diseases=None, pars=None, *args, **kwargs):
        super().__init__(pars, *args, **kwargs)
        self.diseases = sc.tolist(diseases)
        return

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        pass


class AcuteTreatment(ss.BaseTreatment):

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.under_treatment = ss.BoolArr('under_treatment')
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        eligible_uids = self.check_eligibility(sim)  # Check eligibility
        self.coverage_dist.set(p=self.prob)
        seek_uids = self.coverage_dist.filter(eligible_uids)

        if len(seek_uids):
            self.product.administer(sim.people, seek_uids)
            self.under_treatment[seek_uids] = True
            # How may accepted and started treatment today
            n_treated = self.under_treatment[seek_uids].sum()

        self.results['n_treated'][sim.ti] = n_treated

        return n_treated

    def check_eligibility(self, sim):
        may_seek_treatment_today = (sim.typhoidsimple.ti_seek_treatment == sim.ti).uids
        return may_seek_treatment_today


class basic_treatment(ss.Vx):
        """
        Reduction in infectiousness. This product is applied to acute cases
        and results in a reduction in shedding.
        """

        def __init__(self, pars=None, *args, **kwargs):
            super().__init__()
            self.default_pars(multiplier=0.5)
            self.update_pars(pars, **kwargs)
            return

        def administer(self, people, uids):
            people.typhoidsimple.infectiousness[uids] *= self.pars.multiplier
            return
