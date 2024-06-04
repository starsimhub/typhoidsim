"""

"""
import numpy as np
import networkx as nx

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,  # Starting year
    n_years=2*ty.days_per_year,  # Number of years to simulate
    dt=1.0,       # Timestep in days
    verbose=0,    # Don't print details of the run
    rand_seed=2,  # Set a non-default seed
)


my_product = ty.infectiousness_redux(multiplier=0.5)

my_intervention = ss.BaseTreatment(
    prob=1.0,             # probability of seeking treatment when acute
    product=my_product  # use basic treatment that reduces infectiousness
)

typhoid = ty.TyphoidSimple()


ppl = ss.People(10000)

# This example runs on one static networks + the maternal network
network = ss.RandomNet()

sim = ss.Sim(
    pars=pars,
    networks=network,
    diseases=typhoid,
    )

sim.run()
sim.plot()
