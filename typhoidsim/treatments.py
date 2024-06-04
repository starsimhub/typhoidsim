"""
Define Typhoid-specific treatments (interventions)
"""

import starsim as ss
import sciris as sc
import numpy as np
from .typhoid import TyphoidSimple

__all__ = ['acute_treatment', 'infectiousness_redux']


class ViVax(ss.Vx):
    """ Vaccine product """

    def __init__(self, diseases=None, pars=None, *args, **kwargs):
        super().__init__(pars, *args, **kwargs)
        self.diseases = sc.tolist(diseases)
        return

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        pass


class acute_treatment(ss.BaseTreatment):

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.under_treatment = ss.BoolArr('under_treatment')
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        # Who receives the treatment
        # 1. Get acute candidates
        # 2. Check who is untreated and may seek treatment today (asess ti_seek_treatment is close to
        # this timestep (new candidates); agents under treatment continue to be under treatment until
        # the end of their acute phase
        # 3. Of the ones that are eligible to 'seek' treatment today, decide who
        # does and who doesnt't
        # 4. Search candidates that are
        acute_candidates = self.get_candidates(sim)
        eligible_uids = self.check_eligibility(sim, acute_candidates)  # Check eligibility
        self.coverage_dist.set(p=self.prob)
        seek_uids = self.coverage_dist.filter(eligible_uids)

        if len(seek_uids):
            self.product.administer(sim.people, seek_uids)
            self.under_treatment[seek_uids] = True
            # How may accepted and started treatment today
            n_treated = self.under_treatment[seek_uids].sum()

        self.results['n_treated'][sim.ti] = n_treated

        return n_treated

    def check_eligibility(self, sim, uids):
        may_seek_treatment_today = (sim.typhoidsimple.ti_seek_treatment == sim.ti).uids
        return may_seek_treatment_today

    def get_candidates(self, sim):
        """
        Get candidates for treatment on this timestep.
        """
        # Only agents experience the acute stage of infection
        acute_uids = (sim.people.typhoisimple.acute).uids
        return acute_uids


class infectiousness_redux(ss.Vx):
        """
        Reduction in infectiousness. This product is applied to acute cases
        and results in a reduction or blocking in shedding.
        """

        def __init__(self, pars=None, *args, **kwargs):
            super().__init__()
            self.default_pars(multiplier=0.5)
            self.update_pars(pars, **kwargs)
            return

        def administer(self, people, uids):
            people.typhoidsimple.infectiousness[uids] *= self.pars.multiplier
            return
