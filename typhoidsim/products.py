"""
Define Typhoid-specific products used with typhoid interventions.
"""

import numpy as np
import sciris as sc
import starsim as ss

# Products
__all__ = ['infectiousness_redux', 'infectiousness_clearence',
           'blocking_vaccine', 'typhoid_vaccine', 'typhoid_test']


class typhoid_test(ss.Product):
    """
    This product can be used to mimic a blood culture, or rapid antobody tests.
    The main parameter is sensitivity: the probability of someone who is infected
    tests positive (true positives).
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(sensitivity=ss.bernoulli(p=1.0))
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        """ Apply the diagnostic test to the requested uids. """
        # uids are the uids of people who were eligible to receive the test or accepted to have the test
        infected_uids = (people.typhoid.infected).uids
        are_pos_uids = ss.uids(np.intersect1d(infected_uids, uids))
        # Decide whether the test actually comes back positive
        tested_pos_uids = self.pars.sensitivity.filter(are_pos_uids)
        return tested_pos_uids


class infectiousness_redux(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(multiplier=0.5)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        people.typhoid.infectiousness[uids] *= self.pars.multiplier
        return


class infectiousness_clearence(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(clearance_rate=0.2) # TODO SOON: ss.perday(0.2)  # in fraction of infectiousness CFUs per day that are cleared
        self.update_pars(pars, **kwargs)
        return

    def administer(self, uids):
        sim = self.sim
        # estimate how many CFUs are cleared in one timestep
        clearance = sim.people.typhoid.infectiousness[uids] * self.pars.clearance_rate
        sim.people.typhoid.infectiousness[uids] -= clearance
        return


class blocking_vaccine(ss.Product):
    """
    An Acquisition Blocking vaccine that impacts the overall probability of infection after exposure,
    by modifying the 'susceptibility level' state (typhoid.immunity). If the immunity level is 0, then
    the agen can't acquire an infection, if the level is 1.0, it can acquire the infection -- also
    depends on other factors.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(efficacy=1.0)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        people.typhoid.susceptibility[uids] -= self.pars.efficacy * people.typhoid.susceptibility[uids]
        return


class typhoid_vaccine(ss.Vx):
    """
    Create a vaccine product that affects the probability of infection.

    The vaccine can be either "leaky", in which everyone who receives the vaccine
    receives the same amount of protection (specified by the efficacy parameter)
    each time they are exposed to an infection. The alternative (leaky=False) is
    that the efficacy is the probability that the vaccine "takes", in which case
    that person is 100% protected (and the remaining people are 0% protected).

    Args:
        efficacy (float): efficacy of the vaccine (0<=efficacy<=1)
        leaky (bool): see above
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            efficacy=0.9,
            leaky=True
        )
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        if self.pars.leaky:
            people.typhoid.rel_sus[uids] *= 1 - self.pars.efficacy
        else:
            people.typhoid.rel_sus[uids] *= np.random.binomial(1, 1 - self.pars.efficacy, len(uids))
        return
