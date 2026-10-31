"""
Test the infection clearance treatment on a simulation with environmental transmission only
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
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Disease
typhoid = ty.Typhoid(tppi=0.1, exposure2contact_rate=0) # Environmental transmission only

# Population
ppl = ss.People(10_000)

# Treatment: each day, 5% of those infected (of any type: prepatent, acute, subclinical, chronic) are cured
# (was ty.infection_clearence(product=ty.infectiousness_clearence(clearance_rate=0.05)), which reduced infectiousness by 5% a day, but never cleared anyone)
cure = ss.Treatment(cure=1.0, to='recovered')
clearance = ss.treat(cure, who=typhoid.infected, coverage=ss.probperday(0.05), name='clearance')

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    interventions=clearance,
    )

sim.run()
sim.plot()
plt.show()
