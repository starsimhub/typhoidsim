"""
Test that the number of infections tracked in results and as internal state add up
"""

import sciris as sc
import starsim as ss
import typhoidsim as ty


# Define the parameters
pars = sc.objdict(
    n_agents=10e3,             # Number of agents
    start=2000,                # Starting year
    dur=2,                     # Number of years to simulate
    dt=1.0/ty.days_per_year,   # Timestep of 1 day, expressed in years
    verbose=0,                 # Don't print details of the run
    rand_seed=2,               # Set a non-default seed
)


def make_sim(run=False):
    """
    """
    # If p_death > 0, cum_infections[ti_end] not necessarily equal to sim.diseases.typhoid.n_infections.sum().
    #     if p_death > 0, then sim.diseases.typhoid.n_infections.sum() <= cum_infections[ti_end]
    # Results account for people who died on a given timestep before agents are removed but at the end of a simulation
    # sim.diseases.typhoid.n_infections.sum() will have the "total" number of infections experienced by agents currently alive,
    # while cum_infections will have the totsl number of infections experienced by all agents who ever lived throughout the simulation
    diseases = [ty.Typhoid(pars={"p_death": ss.bernoulli(p=0.00)})]
    networks = [ ss.RandomNet({'n_contacts': 5})]
    sim = ss.Sim(pars=pars, networks=networks, diseases=diseases)

    # Optionally run and plot
    if run:
        sim.run()
    return sim

def test_n_infections():
    sim = make_sim()
    sim.run()
    ti_start = 0
    ti_end = -1
    assert sim.diseases.typhoid.results.cum_infections[ti_start] == sim.diseases.typhoid.results.new_infections[ti_start]
    assert sim.diseases.typhoid.results.cum_infections[ti_end] == sim.diseases.typhoid.n_infections.sum()
    return




if __name__ == "__main__":
    test_n_infections()
