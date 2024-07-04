"""
Run a basic Typhoid simulation without any interventions.
"""
import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
n_years = 0.5  # Number of years to simulate
pars = sc.objdict(
    start=2000,  # Starting year
    n_years=n_years*ty.days_per_year,
    dt=1.0,       # Timestep (assumed to be in days for typhoid)
    verbose=0,    # Don't print details of the run
    rand_seed=21,  # Set a non-default seed
)

# Explicitly set demographics to be constant (ie, no births or deaths)
births = ss.Births({'birth_rate': 0.0})
deaths = ss.Deaths({'death_rate': 0.0})

# Diseas
typhoid = ty.TyphoidSimple()

# Population
ppl = ss.People(10000)

# Person-to-person interactions
network = ss.RandomNet()

sim = ss.Sim(
    pars=pars,
    demographics=[births, deaths],
    networks=network,
    diseases=typhoid,
    analyzers=ty.analyzers.states_consistency()
    )

sim.run()
sim.plot()
plt.show()