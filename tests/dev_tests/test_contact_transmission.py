"""

"""
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
n_years = 2  # Number of actual years to simulate
pars = sc.objdict(
    start=2000,  # Starting day/year
    n_years=n_years*ty.days_per_year,  # Number of days to simulate
    dt=1.0,       # Timestep expressed in days
    verbose=0,    # Don't print details of the run
    rand_seed=2,  # Set a non-default seed
)

# Who
ppl = ss.People(10000)

# What
typhoid = ty.Typhoid()

#
network = ss.RandomNet({'n_contacts': 5})

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    networks=network
    )

sim.run()
sim.plot()
plt.show()
