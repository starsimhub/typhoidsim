
"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd


__all__ = ['Births']


class Births(ss.Demographics):
    """ Create births based on rates, rather than based on pregnancy """
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            birth_rate = ss.peryear(20),  # Updated to use ss.peryear for consistency
            rel_birth = 1,
            rate_units = 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Year', value='CBR')),
            metadata,
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.n_births = 0 # For results tracking
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        if isinstance(self.pars.birth_rate, pd.DataFrame):
            br_year = self.pars.birth_rate[self.metadata.data_cols['year']]
            br_val = self.pars.birth_rate[self.metadata.data_cols['value']]
            all_birth_rates = np.interp(self.timevec, br_year, br_val)  # Use timevec from self for flexibility
            self.pars.birth_rate = all_birth_rates
        return

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        if isinstance(birth_rate, (pd.Series, pd.DataFrame)):
            return birth_rate.xs(0, level='age')
        return birth_rate

    def init_results(self):
        self.define_results(
            ss.Result('new', dtype=int, scale=True, label='New births'),
            ss.Result('cumulative', dtype=int, scale=True, label='Cumulative births'),
            ss.Result('cbr', dtype=float, scale=False, label='Crude birth rate'),
        )
        return

    def get_births(self):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        sim = self.sim
        p = self.pars

        if isinstance(p.birth_rate, (pd.Series, pd.DataFrame)):
            available_years = p.birth_rate.index
            year_ind = sc.findnearest(available_years, sim.t.now('year'))
            nearest_year = available_years[year_ind]
            this_birth_rate = p.birth_rate.loc[nearest_year]
        else:
            this_birth_rate = p.birth_rate

        factor = ss.time_ratio(unit1=self.t.unit, dt1=self.t.dt, unit2='year', dt2=1.0)  # Use time_ratio for consistency
        scaled_birth_prob = this_birth_rate * p.rate_units * p.rel_birth * factor
        scaled_birth_prob = np.clip(scaled_birth_prob, a_min=0, a_max=1)
        n_new = np.random.binomial(n=sim.people.alive.count(), p=scaled_birth_prob)  # Not CRN safe, see issue #404
        return n_new

    def step(self):
        new_uids = self.add_births()
        self.n_births = len(new_uids)
        return new_uids

    def add_births(self):
        """ Add n_new births to each state in the sim """
        people = self.sim.people
        n_new = self.get_births()
        new_uids = people.grow(n_new)
        people.age[new_uids] = 0
        return new_uids

    def update_results(self):
        self.results.new[self.ti] = self.n_births

        # Calculate crude birth rate (CBR)
        inv_rate_units = 1.0 / self.pars.rate_units
        births_per_year = self.n_births / self.sim.t.dt_year
        denom = self.sim.people.alive.sum()
        self.results.cbr[self.ti] = inv_rate_units * births_per_year / denom
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return
