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

typhoid = ty.Typhoid()

ppl = ss.People(10000)

# This example runs on one static networks + the maternal network
network = ss.RandomNet({'n_contacts': ss.poisson(lam=0.18)})

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    )

sim.run()
sim.plot()
plt.show()
