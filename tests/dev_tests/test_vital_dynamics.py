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
    dur=ss.years(1),  # Duration of the simulation
    dt=ss.days(1),    # Timestep of 1 day
    rand_seed=2,      # Set a non-default seed
)

typhoid = ty.Typhoid()

ppl = ss.People(10_000)


demographics = [
    ss.Births(birth_rate=ss.peryear(20)), # Per 1000 people
    ss.Deaths(death_rate=ss.peryear(15))
]

network = ss.RandomNet(n_contacts=5)

sim = ss.Sim(
    pars=pars,
    people=ppl,
    networks=network,
    diseases=typhoid,
    demographics=demographics
    )

sim.run()
sim.plot()
plt.show()
