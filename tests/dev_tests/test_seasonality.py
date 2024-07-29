"""
Run a basic Typhoid simulation with seasonal fluctuations of environmental CFUs
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
typhoid = ty.Typhoid(pars={'tppi': 0.1})

# Population
ppl = ss.People(10000)


# Intervention, reduces the average number of exposures
seasonal_pattern = ty.Pattern("baseline_cfu + amp_cfu * cos((2*pi/period)*var)",
                              pars={'baseline_cfu': 4_000_000,
                                    'amp_cfu': 1_000_000,
                                    'period': ty.days_per_year,
                                    'pi': 3.141592653589793})

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    interventions=ty.environmental_seasonality(pattern=seasonal_pattern)
    )

sim.run()
sim.plot()
plt.show()