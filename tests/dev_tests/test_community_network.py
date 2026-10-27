"""
Use and visualise the age-mixing of a commmunity network.
"""
import numpy as np

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=ss.days(11),  # Number of days to simulate
    dt=ss.days(1),    # Timestep of 1 day
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

location = 'Chile'
# Who
ppl = ss.people(location, n_agents=10_000) # The country's age structure, from Starsim's own data (was synthpops)

# What
typhoid = ty.Typhoid()

# How
network = ty.CommunityNet(location=location)

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    networks=network
    )

sc.tic()
sim.run()
sc.toc()
net = sim.networks['communitynet']
ty.plot_age_mixing(net)
plt.show()
