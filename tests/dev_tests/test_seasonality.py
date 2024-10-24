"""
Run a basic Typhoid simulation with seasonal fluctuations of environmental CFUs
"""
import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    n_years=2.0,      # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Disease
typhoid = ty.Typhoid(pars={'tppi': 0.1})

environment = ty.EnvironmentalPool()

# Population
ppl = ss.People(10_000)


# Intervention, reduces the average number of exposures
seasonal_pattern = ty.Pattern("baseline_cfu + amp_cfu * cos((2*pi/period)*var)",
                              pars={'baseline_cfu': 4_000_000,
                                    'amp_cfu': 1_000_000,
                                    'period': 0.25,  # in years
                                    'pi': 3.141592653589793})

my_intervention = ty.environmental_seasonality(seasonal_pattern=seasonal_pattern)
sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    demographics=environment,
    interventions=my_intervention
    )

sim.run()
sim.plot()
plt.show()