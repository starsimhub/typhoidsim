"""
Test counts by age, from the sim's own results (stratified by age), and by vaccination
status, from an analyzer, against what the agents say
"""

import numpy as np
import sciris as sc
import starsim as ss
import typhoidsim as ty

age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60]
pars = sc.objdict(start=2000, dur=ss.years(1), dt=ss.days(1), verbose=0, rand_seed=2)


class infections_by_vaccination(ss.Analyzer):
    """ Count the new infections each step among those aged 2-5 who have been vaccinated, and those who have not """
    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('vaccinated', dtype=int), ss.Result('unvaccinated', dtype=int), ss.Result('all', dtype=int))
        return

    def step(self):
        sim = self.sim
        typhoid = sim.diseases.typhoid
        new = ss.uids.concatenate(typhoid.contact.uids, typhoid.environment.uids)
        age = sim.people.age[new]
        new = new[(age >= 2) & (age < 5)]
        vaccinated = np.isfinite(sim.interventions.vaccination.product.ti_delivered[new])
        self.results.vaccinated[self.ti] = vaccinated.sum()
        self.results.unvaccinated[self.ti] = (~vaccinated).sum()
        self.results.all[self.ti] = len(new)
        return


def make_sim():
    typhoid = ty.Typhoid(init_prev=0.05, p_death=0.0)
    vaccine = ss.Vaccine(efficacy=0.8)
    vaccination = ss.vaccinate(vaccine, at=[2000], who=(ss.ppl.age >= 2) & (ss.ppl.age < 5), coverage=0.5)
    sim = ss.Sim(pars=pars, people=ss.People(10_000, age_data=ss.uniform(0, 60)), diseases=typhoid, networks=ss.RandomNet(n_contacts=5),
                 interventions=vaccination, analyzers=infections_by_vaccination(), stratify=dict(age=age_bin_edges), use_aging=False)
    return sim


def test_vaccinated_counts():
    """ Infections among the vaccinated and the unvaccinated add up, and the vaccinated are infected less """
    sim = make_sim().run()
    res = sim.results.infections_by_vaccination
    assert (res.vaccinated + res.unvaccinated == res.all).all()
    n_vx = sim.results.vaccination.delivered.sum()
    n_group = np.count_nonzero((sim.people.age >= 2) & (sim.people.age < 5))
    rate_vx = res.vaccinated.sum()/n_vx
    rate_unvx = res.unvaccinated.sum()/(n_group - n_vx)
    assert rate_vx < rate_unvx, f'Infections per person: {rate_vx:.3f} vaccinated, {rate_unvx:.3f} unvaccinated'
    return sim


def test_by_age_counts():
    """ With no aging, births, or deaths, the people in typhoid's states by age group are those of each age at the start """
    sim = make_sim()
    sim.init()
    init_ages = sim.people.age.values.copy()
    sim.run()
    typhoid = sim.results.typhoid
    states = ['unexposed', 'susceptible', 'prepatent', 'acute', 'subclinical', 'chronic', 'recovered']
    by_age = sum(typhoid[state].by('age').values[-1] for state in states)
    expected = np.histogram(init_ages, bins=age_bin_edges + [np.inf])[0]
    assert np.allclose(by_age, expected)
    return sim


if __name__ == "__main__":
    test_vaccinated_counts()
    test_by_age_counts()
