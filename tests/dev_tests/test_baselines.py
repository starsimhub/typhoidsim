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


# my_product = ty.infectiousness_redux(multiplier=0.5)
#
# my_intervention = ty.acute_treatment(
#     prob=1.0,             # probability of seeking treatment when acute
#     product=my_product,  # use basic treatment that reduces infectiousness
# )

#my_intervention = ty.base_test()

efficacy_pattern = ty.Pattern("average_efficacy + amp * cos((2*pi/period)*var)",
                              pars={'average_efficacy': 0.9, 'amp': 0.1, 'period': ty.days_per_year/4, 'pi': 3.141592653589793})

my_intervention = ty.environmental_intervention(pattern=efficacy_pattern,
                                                target_factor="env2ppl_exposure_rate")

typhoid = ty.Typhoid()


ppl = ss.People(10000)

# This example runs on one static networks + the maternal network
network = ss.RandomNet({'n_contacts': ss.poisson(lam=0.18)})

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    interventions=my_intervention,
    )

sim.run()
sim.plot()
plt.show()
