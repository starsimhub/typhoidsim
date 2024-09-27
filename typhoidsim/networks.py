import numpy as np
import numba as nb
import sciris as sc
import starsim as ss

from . import ingest as tyi
from . import utils as tyu

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int

__all__ = ["CommunityNet"]


class CommunityNet(ss.DynamicNetwork):
    def __init__(self, pars=None, key_dict=None, **kwargs):
        super().__init__(key_dict=key_dict, **kwargs)
        self.default_pars(
            age_mixing=None,
            location='Chile',
            dur=0  # Duration of zero ensures that new random edges are formed on each time step
        )
        self.update_pars(pars, **kwargs)
        self.mixing_matrix = self.get_mixing_matrix()
        self.n_contact_rate_by_age, self.contact_mixing_matrix = self.get_contact_rates()

        self.add_states(
            ss.FloatArr('age_group', default=0, dtype=ss_int_, label='Age group')
        )

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.pars.age_mixing = tyi.get_age_mix_distribution(self.pars.location)

        return

    def init_post(self, add_pairs=True):
        if add_pairs:
            self.add_pairs()
        return

    def get_contact_rates(self):
        """ """
        contact_rate_matrix = self.age_mixing['matrix']  # in average contacts per day
        n_contact_rate = sc.randround(contact_rate_matrix.sum(axis=1)
        # Transform number of daily contacts into proportion of contacts in each age bin
        contact_rate_probs = contact_rate_matrix / n_contact_rate.reshape(-1, 1)
        return n_contact_rate, contact_rate_probs

    def get_contacts(self, uids, n_contacts):
        a = []
        b = []
        return a, b

    def add_pairs(self):
        """ Generate contacts using a specific age mixing pattern """
        people = self.sim.people
        born = people.alive & (people.age > 0)

        # Convert age into age group
        born_age_group = tyu.digitize_ages(people.age[born.uids], self.pars.age_mixing['age_lb'])

        # total (integer) number of average contacts per day for a given age group
        n_contacts_by_age_grp = sc.randround(self.n_contact_rate_by_age)

        avail_age_groups = np.arange(0, self.pars.age_mixing['age_lb'])

        # Get n_contact per person
        n_contacts = n_contacts_by_age_grp[born_age_group]

        p1, p2 = self.get_contacts(born.uids, n_contacts)

        for p1_uid in born.uids:
            probs = self.mixing_matrix[born_age_group[p1_uid], :]
            p2_age_group = np.random.choice(avail_age_groups,
                                            n_contacts_by_age_grp[born_age_group[p1_uid]],
                                            p=probs)


        beta = np.ones(len(p1), dtype=ss_float_)
        dur = np.full(len(p1),  self.pars.dur)
        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return


    def update(self):
        self.end_pairs()
        self.add_pairs()
        return


class HouseholdNet(ss.DynamicNetwork):
    """
    [WIP]: Microstructured connectivity between agents. Households can change in size
    if one of their members die.
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.default_pars(
            n_contacts=ss.constant(10),
            location=None,
            dur=20, # average duration of a household in years
        )
        self.update_pars(pars, **kwargs)
        self.dist = ss.Dist(distname='HouseholdNet')  # Default RNG
        return

    def init_post(self):
        self.add_pairs()
        return

    @staticmethod
    @nb.njit(cache=True)
    def get_source(inds, n_contacts):
        """ Optimized helper function for getting contacts """
        total_number_of_half_edges = np.sum(n_contacts)
        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=ss_int_)
        for i, person_id in enumerate(inds):
            n = n_contacts[i]
            source[count: count + n] = person_id
            count += n
        return source

    def get_contacts(self, inds, n_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target
        """
        source = self.get_source(inds, n_contacts)
        target = self.dist.rng.permutation(source)
        self.dist.jump()  # Reset the RNG manually # TODO, think if there's a better way
        return source, target

    def update(self):
        self.end_pairs()
        self.add_pairs()
        return

    def add_pairs(self):
        """ Generate contacts """
        people = self.sim.people
        born = people.alive & (people.age > 0)
        if isinstance(self.pars.n_contacts, ss.Dist):
            number_of_contacts = self.pars.n_contacts.rvs(
                born.uids)  # or people.uid?
        else:
            number_of_contacts = np.full(len(people), self.pars.n_contacts)

        number_of_contacts = sc.randround(number_of_contacts / 2).astype(ss_int_)  # One-way contacts

        p1, p2 = self.get_contacts(born.uids, number_of_contacts)
        beta = np.ones(len(p1), dtype=ss_float_)

        if isinstance(self.pars.dur, ss.Dist):
            dur = self.pars.dur.rvs(p1)
        else:
            dur = np.full(len(p1), self.pars.dur)

        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return

    def make_households(self):
       # we have n_households, n_households by size
       # we know who is a household reference
       # we need to generate_household_head_by_size_distribution() based on current distibution
       # Then for every household size:
       # we have to find size-1 contacts for the household heads
       # make adjustments as needed.
       pass

    def init_location_hh_data(self):
        # Set class attributes to have the corresponding data
        pass

    def get_households(self):
        """
        Given a population of size n_agents, and a household size distribution
        calculate the number of households of each size.

        Returns:
            An array with the count of households of size 's' at .
        """
        requested_n_agents = self.sim.pars.n_agents
        if self.pars.location is not None:
            hhs_dist = tyi.get_household_size_distribution(self.pars.location)
            hhs_bins = hhs_dist[:, 0]
            hhs_props = hhs_dist[:, 1] / 100.0

        else:
            # Default household size distribution
            hhs_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            hhs_props = [0.25, 0.125, 0.125, 0.1, 0.1, 0.06, 0.06, 0.06, 0.06, 0.06]

        household_size_dist = ss.choice(p=hhs_props, a=hhs_bins,
                                        name='Household size distribution',
                                        strict=False)

        av_household_size = np.sum(hhs_bins*hhs_bins)
        n_households = np.round(requested_n_agents / av_household_size)
        n_households_by_size = np.zeros(len(hhs_bins), dtype=ss_int_)

        for idx, prop in enumerate(hhs_props):
            n_households_by_size[idx] = int(prop * n_households)

        # Calculate number of agents we would have based on the household distributions
        resulting_n_agents = np.sum([n_households_by_size[s - 1] * s for s in hhs_bins], dtype=int)

        # Check difference between requested population size and expected based on household data
        n_agents_diff = resulting_n_agents - requested_n_agents

        while n_agents_diff != 0:
            new_household_size = household_size_dist.rvs(1)
            n_agents_diff, n_households_by_size = adjust_households(n_agents_diff,
                                                                    n_households_by_size,
                                                                    new_household_size)

        return n_households_by_size

    def select_household_heads(self, nhh_by_hs):
        # TODO: make this a function that returns probabilities, so it can be
        # used by household_head_dist = ss.bernoulli(), to determine whether
        # someone is a household head or not

        if self.pars.location is not None:
            hh_head_age_dist = tyi.get_household_head_age_distribution(self.pars.location)
        else:
            hh_head_age_dist = np.empty()
            hh_head_age_dist[:, 0] = np.arange(0, 101)
            hh_head_age_dist[:, 1] = np.random.dirichlet(np.ones(101), size=1)[0]

        head_age_probs = hh_head_age_dist[:, 1]
        n_households = np.sum(nhh_by_hs)

        # Assumes we have all ages represented in head_age_probs
        probs = [head_age_probs[age] for age in self.sim.people.age]
        probs /= np.sum(probs)

        # sample from array based on calculated probabilities
        head_uids = np.random.choice(self.sim.people.uids, size=n_households, p=probs)

        return head_uids


def adjust_households(delta_agents, nhh_by_hs, delta_hh_size):
    """
    Adjusts the distribution of households sizes (and the total number of
    households as a consequence) based on a specified number of people that need
    to be added or removed.

    Args:
        delta_agents (int): Number of agents to be added or removed. If
        negative, people are added. If is positive, people are removed.
        nhh_by_hs (np.array): number of household of a given size (current distribution/count) .
        delta_hh_size (int): The size of the household to be added or removed.

    Returns:
       delta_agents (int): number of remaining agents to be added or removed
       nhh_by_hs (np.array): adjusted number of household of a given size.
    """
    sign = -np.sign(delta_agents)
    delta_agents = delta_agents + sign*delta_hh_size
    nhh_by_hs[delta_hh_size - 1] = nhh_by_hs[delta_hh_size - 1] + sign
    return delta_agents, nhh_by_hs


def generate_household_head_by_size_distribution(nhh_by_hs, hh_head_age_dist):
    """
    Args:
        nhh_by_hs (np.array): number of household of a given size (current distribution/count) .
        hh_head_age_dist (int): current distribution of household head ages.

    Returns:

    """
    import pandas as pd
    # Total number of households
    n_households = np.sum(nhh_by_hs)
    # Transform to probabilities
    hh_size_dist = nhh_by_hs / n_households

    # Counts of unique combinations of household heads of a given age and household sie
    df = pd.DataFrame(np.outer(hh_size_dist[:, 1], hh_head_age_dist[:, 1]) * n_households,
                      columns=hh_head_age_dist[:, 0], index=hh_size_dist[:, 0])

    return df
