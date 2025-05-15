import numpy as np
import numba as nb
import scipy.stats as spst

import sciris as sc
import starsim as ss

from . import defaults as tyd
from . import ingest as tyi
from . import utils as tyu

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int

__all__ = ["AgeGroup", "CommunityNet", "HouseholdNet"]


class AgeGroup(sc.prettyobj):
    """ A simple age-based filter that returns uids of agents that match the criteria """
    def __init__(self, low, high, do_cache=True, cache_dur_ti=None):
        self.low = low
        self.high = high

        self.do_cache = do_cache
        if (not do_cache) or (cache_dur_ti is None):
            self.cache_dur_ti = 0
        else:
            self.cache_dur_ti = cache_dur_ti # cache duration in timesteps
            
        self.uids = None # Cached
        self.ti_cache = -1 - self.cache_dur_ti # remove cache duration so groups are recached on first step

        self.name = repr(self)
        return

    def __call__(self, sim):
        if (not self.do_cache) or ((self.ti_cache + self.cache_dur_ti) <= sim.ti):
            in_group = sim.people.age >= self.low
            if self.high is not None:
                in_group = in_group & (sim.people.age < self.high)
            self.uids = ss.uids(in_group)
            self.ti_cache = sim.ti
        return self.uids

    def __repr__(self):
        return f'age({self.low}-{self.high})'
    
class CommunityNet(ss.DynamicNetwork):
    """ Create an age-assortative network based on a 2D age-mixing pattern."""
    def __init__(self, pars=None, key_dict=None, **kwargs):
        super().__init__(key_dict=key_dict, **kwargs)
        self.define_pars(
            age_mixing=None,
            location='Chile',
            dur=ss.dur(0, unit='day'),  # Duration ensures new random edges each time step
        )
        self.update_pars(pars, **kwargs)

        if self.pars.age_mixing is None:
            self.pars.age_mixing = tyi.get_age_mix_distribution(self.pars.location)

        # Get and track some useful variables
        self.contact_rate_num_by_ag_gr, self.age_mix_matrix_probs = self.get_contact_rates()
        self.num_age_groups = len(self.pars.age_mixing['age_lb'])

        self.define_states(
            ss.Arr('age_group', default=0, dtype=ss_int_, nan=-999_999, label='Age group') # CK: TODO: consider ss.dtypes.int_nan value
        )
        # Store the size of each age group
        self.age_group_size = None

        return

    def get_contact_rates(self):
         """
         Get average number of total number of contacts per day, per age group
         (num age groups x 1), and a matrix of the average proportion of
         contacts per day of each age group (num age groups x num age groups).
         """
         contact_rate_matrix = self.pars.age_mixing['matrix']  # in average contacts per day
         # Transform number of daily contacts into proportion of contacts in each age bin
         total_rate = contact_rate_matrix.sum(axis=1)
         total_rate[total_rate == 0.0] = 1.0  # avoid division by zero
         contact_rate_probs = contact_rate_matrix / total_rate.reshape(-1, 1)
         # Get integer number of contacts per age group
         contact_rate_num = sc.randround(contact_rate_matrix.sum(axis=1))
         return contact_rate_num, contact_rate_probs
 
    
    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def init_post(self, add_pairs=True):
        self.get_age_groups()
        if add_pairs:
            self.add_pairs()
        return

    def get_age_groups(self):
        """ Find the age group each person belongs to, and the size of each group"""
        self.age_group[:] = tyu.digitize_ages(self.sim.people.age[:],
                                              np.concatenate([self.pars.age_mixing['age_lb'],
                                                              self.pars.age_mixing['age_ub'][-1:]]))

        people = self.sim.people
        born = people.alive & (people.age > 0)

        # Size in fraction of total population size
        self.age_group_size = np.bincount(self.age_group[born.uids],
                                          weights=np.ones(len(born.uids))/len(born.uids))
        return

    def get_contacts(self, born):
        uids = born.uids
        ag = self.age_group[uids]
    
        # get index for each person by age group
        ag_indices = [np.where(ag == i)[0] for i in range(self.num_age_groups)]
        
        p1_list, p2_list = [], []
    
        for ag1 in range(self.num_age_groups):
            # pull out individuals of selected age group
            idx1 = ag_indices[ag1]
            if len(idx1) == 0:
                continue
    
            for ag2 in range(self.num_age_groups):
                # pull out the individuals of the contact age group
                idx2 = ag_indices[ag2]
                if len(idx2) == 0:
                    continue
    
                # given the mixing rate for these two age groups, and the number of people
                # in the ego age group, how many contacts in total should be drawn between
                # these groups?
                n_edges = np.round(self.pars.age_mixing['matrix'][ag1, ag2] * len(idx1)).astype(int)

                # sample the pairs for this number of edges
                # NOTE: this does not default preserve the contact rate per person,
                # but on average should
                if ag1 == ag2:
                    # Sample without replacement within same group
                    if len(idx1) < 2 or n_edges == 0:
                        continue
                    pairs = np.random.choice(len(idx1), size=(n_edges, 2))
                    valid = pairs[:, 0] != pairs[:, 1] # remove self pairs
                    pairs = pairs[valid]
                    p1 = uids[idx1[pairs[:, 0]]]
                    p2 = uids[idx1[pairs[:, 1]]]
                else:
                    # Between groups: random pairings
                    i1 = np.random.choice(len(idx1), size=n_edges)
                    i2 = np.random.choice(len(idx2), size=n_edges)

                    p1 = uids[idx1[i1]]
                    p2 = uids[idx2[i2]]
                    
                p1_list.append(p1)
                p2_list.append(p2)
        return np.concatenate(p1_list), np.concatenate(p2_list)

    def add_pairs(self):
        """ Generate contacts using a specific age mixing pattern """
        people = self.sim.people
        born = people.alive & (people.age > 0)

        p1, p2 = self.get_contacts(born)

        beta = np.ones(len(p1), dtype=ss_float_)
        dur = np.full(len(p1), self.pars.dur.values)
        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return

    def step(self):
        self.end_pairs()
        self.get_age_groups()
        self.add_pairs()
        return

    def estimate_age_mixing_density(self, to_plot=None):
        """Perform a 2d KDE on rither ages or age groups of p1 and p2"""

        if to_plot in ['age', 'ages']:
            X, Y = np.mgrid[tyd.min_age:tyd.max_age, tyd.min_age:tyd.max_age]
            p1a = self.sim.people.age[self.p1]
            p2a = self.sim.people.age[self.p2]
        elif to_plot in ['age_group', 'age_groups', 'age_bins']:
            X, Y = np.mgrid[0:self.num_age_groups, 0:self.num_age_groups]
            p1a = self.age_group[self.p1]
            p2a = self.age_group[self.p2]
        else:
            raise ValueError(f"Unknown option to_plot={to_plot}")

        ages = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([p1a, p2a])
        kernel = spst.gaussian_kde(values)
        kde = np.reshape(kernel(ages).T, X.shape)
        return kde

    def plot_age_mixing_density(self, to_plot=None):
        """ Plot the age-group by age-group density matrix """
        import matplotlib.pyplot as plt

        if to_plot is None:
            to_plot = "age_group"

        if to_plot in ['age', 'ages']:
            p1a = self.sim.people.age[self.p1]
            p2a = self.sim.people.age[self.p2]
        elif to_plot in ['age_group', 'age_groups', 'age_bins']:
            p1a = self.age_group[self.p1]
            p2a = self.age_group[self.p2]
        else:
            raise ValueError(f"Unknown option to_plot={to_plot}")

        kde = self.estimate_age_mixing_density(to_plot=to_plot) / self.age_group_size.reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(18, 11))
        vmin, vmax = 0.0, 0.5

        if to_plot in ['age', 'ages']:
            ax.plot(p1a, p2a, '.',
                    markerfacecolor='dodgerblue', alpha=0.1,
                    markersize=1)
            xmin, xmax = p1a.min(), p1a.max()
            ymin, ymax = p2a.min(), p2a.max()
            # Plot density
            im = ax.imshow(np.rot90(kde), cmap=plt.cm.Blues,
                      extent=[xmin, xmax, ymin, ymax],
                      vmin=vmin, vmax=vmax)

        if to_plot in ['age_group', 'age_groups', 'age_bins']:
            im = ax.imshow(np.rot90(kde), cmap=plt.cm.Blues,
                      vmin=vmin, vmax=vmax)
            plt.xticks(np.arange(self.num_age_groups),
                       self.pars.age_mixing['age_group'],
                       rotation=30)
            plt.yticks(np.arange(self.num_age_groups),
                       self.pars.age_mixing['age_group'][::-1],
                       rotation=30)
        cb = plt.colorbar(im)
        plt.suptitle('Community Network age mixing density (single time step)')
        ax.set_xlabel('Age of individual (years)')
        ax.set_ylabel('Age of contact (years)')
        return fig

        
class HouseholdNet(ss.DynamicNetwork):
    """
    [WIP]: Microstructured connectivity between agents. Households can change in size
    if one of their members die.
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):
        """ Initialize """
        super().__init__(key_dict=key_dict)
        self.define_pars(
            n_contacts=ss.constant(10),
            location=None,
            dur=20, # TODO SOON: ss.years(20) # average duration of a household in years
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

    def step(self):
        self.end_pairs()
        self.add_pairs()
        return

    def add_pairs(self):
        """ Generate contacts """
        people = self.sim.people
        born = people.alive & (people.age > 0)
        if isinstance(self.pars.n_contacts, ss.Dist):
            number_of_contacts = self.pars.n_contacts.rvs(born.uids)  # or people.uid?
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

        av_household_size = np.sum(hhs_bins * hhs_props) # CK: is this correct? Change made by LLM
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
            hh_head_age_dist = np.empty((101, 2))
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
    delta_agents = delta_agents + sign * delta_hh_size
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
