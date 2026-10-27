"""

"""
import numpy as np

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=ss.years(1),  # Duration of the simulation
    dt=ss.days(1),    # Timestep of 1 day
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Who
ppl = ss.People(10_000)

# What
typhoid = ty.Typhoid(teer=0) # Contact transmission only

#
network = ss.RandomNet(n_contacts=5)

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    networks=network
    )

sim.run()
sim.plot()
plt.show()
