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
    dur=ss.years(1),  # Duration of the simulation
    dt=ss.days(1),    # Timestep of 1 day
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

typhoid = ty.Typhoid()

ppl = ss.People(10_000)

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    )
sim.run()


# The prepatent duration in days, at the dose received from the environment at the end of the run (by contact, it is the high dose's)
p = sim.diseases.typhoid.pars
dose = sim.results.typhoid.dose[-1]
mean, sigma = [ty.typhoid.by_dose(ty.typhoid.prepatent_levels[key], dose, p.cfu_lo_me.value, p.cfu_me_hi.value) for key in ['mean_dur', 'std_dur']]
prepatent_dur = ss.lognorm_im(mean=mean, sigma=sigma, strict=False).rvs(10_000)
prepatent_dur_ti = sc.randround(prepatent_dur/sim.t.dt.days)          # prepatent duration in number of timesteps (equivalent to 1 day if dt=1 day)

fig, axs = plt.subplots(1, 1)

bin_centers = sc.inclusiverange(0, 40) - 0.5
axs.hist(prepatent_dur, bins=bin_centers, label="Prepatent duration in days (continuous time)", facecolor="tab:blue", alpha=0.5)
axs.hist(prepatent_dur_ti, bins=bin_centers, label="Prepatent duration in timesteps (discrete time)", facecolor="grey", alpha=0.5)
axs.set_xlabel("Duration (days)")
axs.set_ylabel("Counts")
axs.legend()
plt.show()
