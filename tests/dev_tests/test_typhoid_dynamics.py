"""
Run a basic Typhoid simulation, with a cleanup of the environment from the middle of the year.
"""
import numpy as np
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
typhoid = ty.Typhoid()

# Population
ppl = ss.People(10_000)

# Person-to-person interactions
network = ss.RandomNet(n_contacts=ss.poisson(lam=0.18))   # lam represents the average number of daily exposures

# Intervention, cleans up the environment: each day, it removes a share of the CFUs (the efficacy), which varies in time
start_year = 2000.5 # Was 2500, after the end of the sim, so it was never active (v4 refuses a start outside the sim)
def efficacy(sim):
    """ The share of CFUs removed each day: a cosine with a period of a quarter year, from the start of the cleanup (was a ty.Pattern) """
    t = sim.t.now('year') - start_year
    return np.clip(0.9 + 0.1*np.cos(2*np.pi*t/0.25), 0, 0.999)

# A square "pulse" intervention, for the first two days: return np.where(t < 2/365, 0.9, 0.0)

# Removing a share e of the CFUs each day adds -log(1 - e) per day to their decay rate (was ty.environmental_cleanup(efficacy=efficacy_pattern, start_year=start_year, dur=1))
my_intervention = ss.modify('typhoid.pars.decay_rate', add=lambda sim: ss.perday(-np.log(1 - efficacy(sim))), start=start_year, stop=start_year + 1, name='cleanup')

sim = ss.Sim(
    pars=pars,
    people=ppl,
    networks=network,
    diseases=typhoid,
    interventions=my_intervention
    )

sim.run()
sim.plot()
plt.show()
