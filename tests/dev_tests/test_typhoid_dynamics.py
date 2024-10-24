"""
Run a basic Typhoid simulation without any interventions.
"""
import numpy as np
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

# Disease
typhoid = ty.Typhoid()

# Population
ppl = ss.People(10_000)

# Person-to-person interactions
network = ss.RandomNet({'n_contacts': ss.poisson(lam=0.18)})   # lam represents the average number of daily exposures

# Intervention, reduces the average number of exposures
efficacy_pattern = ty.Pattern("average_efficacy + amp * cos((2*pi/period)*var)",
                              pars={'average_efficacy': 0.9,
                                    'amp': 0.1,
                                    'period': 0.25,
                                    'pi': 3.141592653589793})

#Square "pulse" intervention
# efficacy_pattern = ty.Pattern("where((var >= start) & (var < start+dur), average_efficacy, 0.0)",
#                               pars={
#                                   'start': 0,          # expressed in number of years relative to the start of the simulation that is considered to be 0
#                                   'dur': 2.0/365.0,    # in years
#                                   'average_efficacy': 0.9})
#
# # "Delta" intervention
# efficacy_pattern = ty.Pattern("where((var == 0.0), average_efficacy, 0.0)",
#                               pars={'average_efficacy': 0.9})

my_intervention = ty.environmental_cleanup(efficacy=efficacy_pattern, start=2500, dur=1)

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    interventions=my_intervention
    )

sim.run()
sim.plot()
plt.show()