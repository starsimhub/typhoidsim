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

typhoid = ty.Typhoid()

ppl = ss.People(10_000)

network = ss.RandomNet(n_contacts=5)

sim = ss.Sim(
    people=ppl,
    pars=pars,
    networks=network,
    diseases=typhoid,
    )
sc.tic()
sim.run()
sc.toc()
sim.plot()
plt.show()
