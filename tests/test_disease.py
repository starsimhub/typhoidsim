"""
Test that the number of infections tracked in results and as internal state add up
"""
import numpy as np

import sciris as sc
import starsim as ss
import typhoidsim as ty


# Define the parameters
pars = sc.objdict(
    n_agents=10e3,             # Number of agents
    start=2000,                # Starting year
    dur=10.0/365.0,             # Number of days to simulate
    dt=1.0/ty.days_per_year,   # Timestep of 1 day, expressed in years
    verbose=0,                 # Don't print details of the run
    rand_seed=2,               # Set a non-default seed
)


def make_sim_no_deaths(run=False):
    """
    """
    # If p_death > 0, cum_infections[ti_end] not necessarily equal to sim.diseases.typhoid.n_infections.sum().
    #     if p_death > 0, then sim.diseases.typhoid.n_infections.sum() <= cum_infections[ti_end]
    # Results account for people who died on a given timestep before agents are removed but at the end of a simulation
    # sim.diseases.typhoid.n_infections.sum() will have the "total" number of infections experienced by agents currently alive,
    # while cum_infections will have the total number of infections experienced by all agents who ever lived throughout the simulation
    diseases = [ty.Typhoid(pars={"p_death": ss.bernoulli(p=0.00)})]
    networks = [ss.RandomNet({'n_contacts': 5})]
    sim = ss.Sim(pars=pars, networks=networks, diseases=diseases)

    # Optionally run and plot
    if run:
        sim.run()
    return sim


def make_sim_no_transmission(run=False):
    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.1)})
    network = ss.RandomNet({'n_contacts': 0})

    sim = ss.Sim(
        pars=pars,
        diseases=typhoid,
        networks=network
    )
    # Optionally run and plot
    if run:
        sim.run()
    return sim


def test_n_infections():
    sim = make_sim_no_deaths()
    sim.run()
    ti_start = 0
    ti_end = -1
    assert sim.diseases.typhoid.results.cum_infections[ti_start] == sim.diseases.typhoid.results.new_infections[ti_start]
    assert sim.diseases.typhoid.results.new_infections.sum()  == (sim.diseases.typhoid.n_infections.sum() - sim.diseases.typhoid.n_infections_historical)
    assert sim.diseases.typhoid.results.cum_infections[ti_end] == (sim.diseases.typhoid.n_infections.sum() - sim.diseases.typhoid.n_infections_historical)
    return


def test_init_prevalence():
    sim = make_sim_no_transmission()
    sim.run()

    expected_init_prev = sim.pars["n_agents"] * sim.diseases.typhoid.pars.init_prev.pars.p
    simulated_init_prev = sim.diseases.typhoid.n_initial_cases
    assert np.isclose(np.log10(expected_init_prev), np.log10(simulated_init_prev), atol=1e-1)

    assert sim.diseases.typhoid.results.new_infections[0] == sim.diseases.typhoid.n_initial_cases
    return


if __name__ == "__main__":
    test_n_infections()
    test_init_prevalence()
