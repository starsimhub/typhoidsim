"""
Showcase how we can visually inspect the age distribution of a given population
"""
import matplotlib.pyplot as plt
import sciris as sc

import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=0.25,     # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

ppl_chile    = ss.People(10_000, age_data=ty.get_age_distribution('Chile'))
ppl_pakistan = ss.People(10_000, age_data=ty.get_age_distribution('Pakistan'))

sim_chile = ss.Sim(pars=pars, people=ppl_chile)
sim_chile.run()

sim_pakistan = ss.Sim(pars=pars, people=ppl_pakistan)
sim_pakistan.run()

ty.plot_age_histogram(sim_chile.people)
ty.plot_age_histogram(sim_pakistan.people)

plt.show()
