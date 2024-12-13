"""
Test simulation using vital dynamics
"""
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

typhoid = ty.Typhoid()

ppl = ss.People(10_000)


demographics = [
    ty.Births(birth_rate=20),
    ss.Deaths(death_rate=15)
]

network = ss.RandomNet({'n_contacts': 5})

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    demographics=demographics
    )
sc.tic()
sim.run()
sc.toc()
sim.plot()
plt.show()
