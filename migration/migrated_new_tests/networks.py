
import numpy as np
import numba as nb
import scipy.stats as spst

import sciris as sc
import starsim as ss

from . import defaults as tyd
from . import ingest as tyi
from . import utils as tyu

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int

__all__ = ["CommunityNet"]

class CommunityNet(ss.DynamicNetwork):
    """ Create an age-assortative network based on a 2D age-mixing pattern."""

    def __init__(self, pars=None, key_dict=None, **kwargs):
        super().__init__(key_dict=key_dict, **kwargs)
        self.define_pars(
            age_mixing=None,
            location='Chile',
            dur=0  # Duration of zero ensures that new random edges are formed on each time step
        )
        self.update_pars(pars, **kwargs)

        if self.pars.age_mixing is None:
            self.pars.age_mixing = tyi.get_age_mix_distribution(self.pars.location)

        # Get and track some useful variables
        self.contact_rate_num_by_ag_gr, self.age_mix_matrix_probs = self.get_contact_rates()
        self.num_age_groups = len(self.pars.age_mixing['age_lb'])

        self.define_states(
            ss.IndexArr('age_group', default=0, dtype=ss_int, label='Age group')
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

    def get_contacts(self, born, n_contacts):
        """ Generate contacts based on age mixing"""
        available_uids = born.uids

        # Get all possible connections in the networks (upper triangle)
        idx1, idx2 = np.triu_indices(n=len(available_uids), k=1)

        # Weight age-group probabilities by the proportion of each age group in this specific population
        probs = self.age_mix_matrix_probs * self.age_group_size.reshape(-1, 1)

        edge_probs = probs[self.age_group[available_uids[idx1]],
                           self.age_group[available_uids[idx2]]]
        connected = np.random.rand(len(edge_probs)) <= edge_probs

        source = idx1[connected]
        target = idx2[connected]
        return source, target

    def add_pairs(self):
        """ Generate contacts using a specific age mixing pattern """
        people = self.sim.people
        born = people.alive & (people.age > 0)

        # Convert age into age group
        born_age_group = self.age_group[born.uids]

        # Total (integer) number of average contacts **per day** for each available age group
        n_contacts_by_age_grp = sc.randround(self.contact_rate_num_by_ag_gr)

        # Get the total number of contacts each person will have in one time step (1 day)
        n_contacts = n_contacts_by_age_grp[born_age_group]

        p1, p2 = self.get_contacts(born, n_contacts)

        beta = np.ones(len(p1), dtype=ss_float)
        dur = np.full(len(p1),  self.pars.dur)
        self.append(p1=p1, p2=p2, beta=beta, dur=dur)
        return

    def step(self):
        self.end_pairs()
        self.get_age_groups()
        self.add_pairs()
        return

    def estimate_age_mixing_density(self, to_plot=None):
        """Perform a 2d KDE on either ages or age groups of p1 and p2"""

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
