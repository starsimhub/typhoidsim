"""
Showcase how we can visually inspect the age distribution of a given population
"""
import matplotlib.pyplot as plt
import sciris as sc

import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,          # Starting year
    dur=ss.days(90),     # Duration of the simulation
    dt=ss.days(1),       # Timestep of 1 day
    verbose=1,           # Print details of the run
    rand_seed=2,         # Set a non-default seed
)

ppl_chile    = ss.people('Chile', n_agents=10_000) # The country's age structure, from Starsim's own data (was synthpops)
ppl_pakistan = ss.people('Pakistan', n_agents=10_000)

sim_chile = ss.Sim(pars=pars, people=ppl_chile)
sim_chile.run()

sim_pakistan = ss.Sim(pars=pars, people=ppl_pakistan)
sim_pakistan.run()

ty.plot_age_histogram(sim_chile.people)
ty.plot_age_histogram(sim_pakistan.people)

plt.show()
