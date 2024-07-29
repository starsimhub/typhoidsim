"""
Run a basic Typhoid simulation without any interventions.
"""
import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
n_years = 2  # Number of actual years to simulate
pars = sc.objdict(
    start=2000,  # Starting year
    n_years=n_years*ty.days_per_year,
    dt=1.0,       # Timestep (assumed to be in days for typhoid)
    verbose=0,    # Don't print details of the run
    rand_seed=21,  # Set a non-default seed
)

# Disease
typhoid = ty.Typhoid()

# Population
ppl = ss.People(10000)

# Person-to-person interactions
network = ss.RandomNet({'n_contacts': ss.poisson(lam=0.18)})   # lam represents the average number of daily exposures

# Intervention, reduces the average number of exposures
efficacy_pattern = ty.Pattern("average_efficacy + amp * cos((2*pi/period)*var)",
                              pars={'average_efficacy': 0.9,
                                    'amp': 0.1,
                                    'period': ty.days_per_year/4,
                                    'pi': 3.141592653589793})

my_intervention = ty.environmental_intervention(pattern=efficacy_pattern,
                                                target_factor="env2ppl_exposure_rate")

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    interventions=my_intervention
    )

sim.run()
sim.plot()
plt.show()