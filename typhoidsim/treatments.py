"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
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
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level.

    For instance, Infectiousness is constant throughout the acute duration and is
    determined TAI (Typhoid Acute Infectiousness). This is mediated by the
    product's treatment multiplier when an individual seeks treatment.
    For an agent who sought and received treatment,
    from treatment day to the end of the stage, its new infectiousness level is
    I(Acute) = TAI * product treatment multiplier

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: acute agents who are untreated and would
            seek treatment today
       - b. previous candidates: acute agents under treatment continue to
            be under treatment until they are no longer acute
    - 1. Of the ones that are eligible to 'seek' treatment today, decide who
         does and who doesnt't
    - 2. Treat new candidates
    - 3. Count how many people receive treatment today
    """

    def __init__(self, product=None, prob=1.0, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.treated = ss.BoolArr('treated')
        self.results = ss.ndict()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)  # count how many were treated today, includes new and old patients
        self.initialized = True
        return

    def apply(self, sim):
        new_candidates = self.get_eligible(sim)
        receive_uids = self.coverage_dist.filter(new_candidates)
        newly_treated = len(receive_uids)
        if newly_treated:
            self.product.administer(sim.people, receive_uids)
            self.treated[receive_uids] = True
            # How may accepted and started treatment today
        self.results['n_treated'][sim.ti] = newly_treated
        return

    def get_eligible(self, sim):
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        acute_uids = (sim.people.typhoidsimple.acute).uids
        # Those who would seek treatment today
        seeks_treatment = (sim.people.typhoidsimple.ti_seek_trtmnt == sim.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)
        return new_candidates


class infection_clearence(ss.Intervention):
    """
    Clears an infected individual's Typhoid infection. It will clear Typhoid
    infections of all types (prepatent, acute, subclinical, chronic)

    This is expected to use the "infectiousness_clearence" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level at every timestep.

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: infected agents
       - b. previous candidates: people under treatment continue to
            be under treatment until they are no longer infected.
    - 2. Treat new candidates AND patients under treatment
    - 3. Count how many people receive treatment today and how many NEW started treatment today
    """

    def __init__(self, product=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self._parse_product(product)
        self.treated = ss.BoolArr('treated')
        self.results = ss.ndict()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)  # count how many were treated today, includes new and old patients
        self.results += ss.Result(self.name, 'n_started_tr', sim.npts, dtype=int)  # count how many started treatment today, includes only new patients
        self.initialized = True
        return

    def apply(self, sim):
        new_patients, under_treatment = self.get_eligible(sim)
        newly_treated = len(new_patients)
        all_treated = newly_treated + len(under_treatment)

        if len(new_patients):
            self.product.administer(sim.people, new_patients)
            self.treated[new_patients] = True

        if len(under_treatment):
            self.product.administer(sim.people, under_treatment)

        self.results['n_started_tr'][sim.ti] = newly_treated
        self.results['n_treated'][sim.ti] = all_treated

        # Check if infectiousness was cleared in this timestep
        treated = ss.uids.cat(new_patients, under_treatment)
        cleared_uids = (sim.people.typhoidsimple.infectiousness[treated] <= 0.0).uids
        if len(cleared_uids):
            # Reset infectiousness
            sim.people.typhoidsimple.infectiousness[cleared_uids] = 0.0

            # Reset infected states
            for state in [
                "prepatent",
                "acute",
                "subclinical",
                "chronic",
            ]:
               sim.people.typhoidsimple.statesdict[state][cleared_uids] = False

            # Reset time of death if this patient was supposed die
            for state in [
                "ti_prepatent",
                "ti_acute",
                "ti_seek_trtment",
                "ti_subclinical",
                "ti_chronic",
                "ti_dead",
            ]:
                sim.people.typhoidsimple.statesdict[state][cleared_uids] = np.nan

            # Set recovered state and when this agent becomes susceptible
            sim.people.typhoidsimple.statesdict["recovered"][cleared_uids] = True
            sim.people.typhoidsimple.statesdict["ti_susceptible"][cleared_uids] = sim.ti + 1

        return

    def get_eligible(self, sim):
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        infected_uids = (sim.people.typhoidsimple.infected).uids

        # Those who are under treatment
        under_treatment = (sim.people.infection_clearence.treated).uids
        # and still are in the acute stage
        old_patients = infected_uids.intersect(under_treatment)
        new_patients = infected_uids.intersect(~under_treatment)
        return new_patients, old_patients


class environmental_intervention(ss.Intervention):
    """
    An environmental intervention that targets number of (CFU) doses
    could impact the number of times the exposure could happen.
    """
    pass


# - Products
class infectiousness_redux(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
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


class infectiousness_clearence(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(clearence_rate=0.2, dt=1.0)  # in fraction of CFUs per day that are cleared
        self.update_pars(pars, **kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.pars.dt = sim.dt
        self.initialized = True
        return

    def administer(self, people, uids):
        clearence = people.typhoidsimple.infectiousness[uids] * self.pars.clearence_rate * self.pars.dt   # multiply by dt for cases when dt < 1
        people.typhoidsimple.infectiousness[uids] -= clearence
        return
