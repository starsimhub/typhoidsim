"""

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
    dur=1.0,      # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Who
ppl = ss.People(100_000)

# What
typhoid = ty.Typhoid()

# How
environment = ty.EnvironmentalPool()

#
sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    demographics=environment,
    )

sc.tic()
sim.run()
sc.toc()
sim.plot()
plt.show()
