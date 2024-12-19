"""
Test that the number of infections tracked in results and as internal state adds up
"""

import sciris as sc
import starsim as ss
import typhoidsim as ty


# Define the parameters
pars = sc.objdict(
    n_agents=10e3,             # Number of agents
    start=2000,                # Starting year
    n_years=2,                 # Number of years to simulate
    dt=1.0/ty.days_per_year,   # Timestep of 1 day, expressed in years
    verbose=0,                 # Don't print details of the run
    rand_seed=2,               # Set a non-default seed
)


def make_sim(run=False):
    """
    """
    diseases = [ty.Typhoid()]
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
