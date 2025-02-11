"""

"""
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=1.0,          # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=0,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Who
ppl = ss.People(10_000)

# What
typhoid = ty.Typhoid(pars={"p_death": 0.0})

demographics = [
    ss.Births(birth_rate=0),
    ss.Deaths(death_rate=0)
]

# How
environment = ty.EnvironmentalPool()

## Interventions
sanitation_efficacy = ty.Pattern("efficacy", pars={'efficacy': 0.5})
sanitation = ty.behavioral_change(efficacy=0.5, start_year=2000.0+180.0/365)

campaign_vax_2_5_yo = ty.vaccination_with_waning(
    start_year=pars["start"],
    prob=0.66,
    booster1_prob=0.0,
    booster2_prob=0.0,
    prob_type="annual",
    debug=True,
    # only use for this example to keep track of each individual's acquired immunity level over time
    age_pars={'min_age': 0.0,
              'max_age': 2.0}
)

to_record = dict(immunity_acquired=dict(path=("diseases", "typhoid"), label="immune_acquired"),
                 rel_sus=dict(path=("diseases", "typhoid"), label="rel_sus"),
                 eff_sus=dict(path=("diseases", "typhoid"), label="eff_sus"))


tracker = ty.track_individuals_monitor(to_record=to_record)

#
sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    demographics=demographics+[environment],
    analyzers=tracker,
    interventions=[sanitation, campaign_vax_2_5_yo]
    )

sim.run()


children_uids = ty.eligibility_by_age(sim, age_min=0.75, age_max=1.5)
vaccinated_uids = (sim.interventions[1].vaccinated).uids
selected_uids = np.intersect1d(children_uids, vaccinated_uids)

sim.analyzers[0].plot_ridge(uids=selected_uids, y_scaling=1.2)

fig, ax = sc.getrowscols(n=1, nrows=1, make=True)
ax.plot(sim.timevec, sim.analyzers[0].results["eff_sus"][:, selected_uids])

plt.show()
