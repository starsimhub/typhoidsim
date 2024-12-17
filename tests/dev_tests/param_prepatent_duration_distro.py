"""
The duration of the prepatent stage of an individual’s infection is
calculated at the beginning of the infection as a draw from a log-normal
distribution, parameterised by two parameters: mu and sigma.

The mu-sigma pair depends on the exposure dose.

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

typhoid = ty.Typhoid()

ppl = ss.People(10_000)

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    )
sim.run()


uids = sim.people.auids
prepatent_dur = sim.diseases.typhoid.pars.dur_prep_dist.rvs(uids).astype(float)  # prepatent duration in days
prepatent_dur_ti = sc.randround((prepatent_dur * ty.day2year) / sim.dt)              # prepatent duration in number of timesteps (equivalent to 1 day if dt=1/365)

fig, axs = plt.subplots(1, 1)

bin_centers = sc.inclusiverange(0, 8) - 0.5
axs.hist(prepatent_dur, bins=bin_centers, label="Prepatent duration in days (continuous time)", facecolor="tab:blue", alpha=0.5)
axs.hist(prepatent_dur_ti, bins=bin_centers, label="Prepatent duration in timesteps (discrete time)", facecolor="grey", alpha=0.5)
axs.set_xlabel("Duration (days)")
axs.set_ylabel("Counts")
axs.legend()
plt.show()
