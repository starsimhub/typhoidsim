"""
Test the natural history of typhoid: that the infections counted in the results and
in each person add up, the initial conditions, and the environmental reservoir
"""
import numpy as np

import sciris as sc
import starsim as ss
import typhoidsim as ty


# Define the parameters
pars = sc.objdict(
    n_agents=10e3,             # Number of agents
    start=2000,                # Starting year
    dur=ss.days(10),           # Number of days to simulate
    dt=ss.days(1),             # Timestep of 1 day
    verbose=0,                 # Don't print details of the run
    rand_seed=2,               # Set a non-default seed
)


def make_sim_no_deaths(run=False):
    """ A sim with transmission by contact, in which nobody dies """
    # With deaths, an agent's n_infections is lost when they die, so would not add up to the infections counted
    diseases = [ty.Typhoid(p_death=0.0)]
    networks = [ss.RandomNet(n_contacts=5)]
    sim = ss.Sim(pars=pars, networks=networks, diseases=diseases)
    if run:
        sim.run()
    return sim


def make_sim_no_transmission(run=False):
    """ A sim with no transmission by contact or through the environment """
    typhoid = ty.Typhoid(init_prev=0.1, exposure2contact_rate=0, teer=0) # With no network, contact would be homogeneous mixing
    sim = ss.Sim(pars=pars, diseases=typhoid)
    if run:
        sim.run()
    return sim


def test_n_infections():
    """ The infections counted by the results are those counted in each agent """
    sim = make_sim_no_deaths(run=True)
    typhoid = sim.diseases.typhoid
    res = typhoid.results
    n_seeds = res.prepatent[0]
    n_infections = np.nansum(typhoid['n_infections'].values)
    assert res.infections[0] == 0 # The results at the start are before any step
    assert res.infections.sum() > 0
    assert res.infections.sum() + n_seeds == n_infections, f'{res.infections.sum()} infections and {n_seeds} seeds, but the agents count {n_infections}'
    return sim


def test_init_prevalence():
    """ The share infected at the start is init_prev, and nobody else is infected """
    sim = make_sim_no_transmission(run=True)
    typhoid = sim.diseases.typhoid
    expected = sim.pars.n_agents*0.1
    assert np.isclose(typhoid.results.prepatent[0], expected, rtol=0.05)
    assert typhoid.results.infections.sum() == 0
    return sim


def test_init_susceptibility():
    """ At the start, the share of each age who are susceptible follows the curve of Gauld et al. (2018), and everyone over 20 is """
    sim = ss.Sim(pars=pars, diseases=ty.Typhoid(init_prev=0, exposure2contact_rate=0, teer=0), people=ss.People(20_000, age_data=ss.uniform(0, 40)))
    sim.init()
    typhoid = sim.diseases.typhoid
    age = sim.people.age.values
    susceptible = typhoid.susceptible.values
    assert susceptible[age >= 20].all()
    young = age < 20
    expected = ty.sigmoid(age[young], 20.0, 1.0).mean()
    assert np.isclose(susceptible[young].mean(), expected, atol=0.02)
    return sim


def test_environment():
    """ With a fixed number of chronic carriers and no new infections, the CFUs in the environment approach shedding/decay """
    n_carriers = 100
    typhoid = ty.Typhoid(init_prev=0, init_n=dict(chronic=n_carriers), exposure2contact_rate=0, teer=0)
    sim = ss.Sim(n_agents=1_000, start=2000, dur=ss.days(60), dt=ss.days(0.25), diseases=typhoid, verbose=0)
    sim.run()
    p = sim.diseases.typhoid.pars
    expected = p.shedding_rate.value*p.tai.value*p.tcri.value*n_carriers/p.decay_rate.value
    cfu = sim.results.typhoid.cfu
    assert np.isclose(cfu[-1], expected, rtol=0.01), f'CFUs {cfu[-1]:n}, expected {expected:n}'
    return sim


if __name__ == "__main__":
    test_n_infections()
    test_init_prevalence()
    test_init_susceptibility()
    test_environment()
