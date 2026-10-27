"""
Run a basic Typhoid simulation with seasonal fluctuations of environmental CFUs
"""
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=ss.years(2),  # Duration of the simulation
    dt=ss.days(1),    # Timestep of 1 day
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Disease, with the environment as its only route
typhoid = ty.Typhoid(tppi=0.1, exposure2contact_rate=0)

# Population
ppl = ss.People(10_000)


# Seasonality: CFUs added to the environment from outside the model, 4 million ± 1 million a day, with a period of a quarter year
seasonality = ss.seasonal(mean=ss.perday(4e6), amplitude=0.25, period=ss.years(0.25))
my_intervention = ss.modify(typhoid.shedding, add=seasonality, name='seasonality')
sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    interventions=my_intervention
    )

sim.run()
sim.plot()
plt.show()
