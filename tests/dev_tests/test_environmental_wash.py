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
    n_years=1.0,      # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Who
ppl = ss.People(10_000)

# What
typhoid = ty.Typhoid()

# How
environment = ty.EnvironmentalPool()


sanitation_efficacy = ty.Pattern("efficacy", pars={'efficacy': 0.5})

sanitation = ty.behavioral_change(efficacy=0.5)

#
sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    demographics=environment,
    interventions=sanitation
    )

sim.run()
sim.plot()
plt.show()
