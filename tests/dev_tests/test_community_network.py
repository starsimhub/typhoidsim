"""
Use and visualise the age-mixing of a commmunity network.
"""
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    n_years=0.125/4,      # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

location = 'Chile'
# Who
ppl = ss.People(10_000, age_data=ty.get_age_distribution(location))

# What
typhoid = ty.Typhoid()

# How
network = ty.CommunityNet(pars={'location': location})
#network = ss.RandomNet({'n_contacts': 10})

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    networks=network
    )

sc.tic()
sim.run()
sc.toc()
net = sim.networks['communitynet']
ty.plot_age_mixing(net)
plt.show()
