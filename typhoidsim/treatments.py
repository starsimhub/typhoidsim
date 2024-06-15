"""
Define Typhoid-specific treatments (interventions)
"""
import numpy as np

import sciris as sc
import starsim as ss

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


class acute_treatment(ss.Intervention):
    """

    Treat acute (symptomatic) subjects.

    This is expected to use the "infectiousness_redux" product, which
    results in a reduction or blocking in shedding

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: acute agents who are untreated and would
            seek treatment today
       - b. previous candidates: acute agents under treatment continue to
            be under treatment until they are no longer acute
    - 1. Of the ones that are eligible to 'seek' treatment today, decide who
         does and who doesnt't
    - 2. Treat new and previous candidates
    - 3. Count how many people receive treatment today (includes new
         and previous candidates)
    """

    def __init__(self, product=None, prob=1.0, eligibility=None, **kwargs):
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.treated = ss.BoolArr('treated')
        super().__init__(**kwargs)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        new_candidates, under_treatment = self.get_eligible(sim)
        receive_uids = self.coverage_dist.filter(new_candidates)
        if len(receive_uids):
            self.product.administer(sim.people, receive_uids)
            self.treated[receive_uids] = True
            # How may accepted and started treatment today
            n_treated = self.treated[receive_uids].sum()
        else:
            total_treated = 0.0
            newly_treated = 0.0

        #self.results['n_treated'][sim.ti] = n_treated
        #self.results['n_treated'][sim.ti] = n_treated

        return

    def check_eligibility(self, sim, uids):
        may_seek_treatment_today = (sim.typhoidsimple.ti_seek_treatment == sim.ti).uids
        return may_seek_treatment_today

    def get_eligible(self, sim):
        """
        Get candidates for treatment on this timestep.
        """
        # Only agents experience the acute stage of infection
        acute_uids = (sim.people.typhoidsimple.acute).uids
        # Those who would seek treatment today
        seeks_treatment = (sim.people.typhoidsimple.ti_seek_trtmnt == sim.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)

        # Those who would seek treatment today
        under_treatment = (sim.people.acute_treatment.treated).uids
        patients = acute_uids.intersect(under_treatment)
        return new_candidates, patients


#

# - Products
class infection_clearence(ss.Intervention):
    """
    Individual-targeted intervention immediately clears an infected individual’s
    Typhoid infection (and expires).
    """
    pass


class infectiousness_redux(ss.Product):
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
