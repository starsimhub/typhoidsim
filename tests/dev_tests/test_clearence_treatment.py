"""
Test the infection clearaence treatment on a simulation with environmental transmission only
"""
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
n_years = 1  # Number of actual years to simulate
pars = sc.objdict(
    start=2000,  # Starting day/year
    n_years=n_years*ty.days_per_year,  # Number of days to simulate
    dt=1.0,       # Timestep expressed in days
    verbose=0,    # Don't print details of the run
    rand_seed=2,  # Set a non-default seed
)

# Disease
typhoid = ty.Typhoid(pars={'tppi': 0.1})

# Population
ppl = ss.People(10000)

# Treatment
my_product = ty.infectiousness_clearence(clearence_rate=0.05)
my_intervention = ty.infection_clearence(
    product=my_product,   # use basic treatment that reduces infectiousness by product mutiplier
)

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    interventions=my_intervention,
    )

sim.run()
sim.plot()
plt.show()
