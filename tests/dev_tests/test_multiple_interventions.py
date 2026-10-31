"""

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
    verbose=0,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

# Who
ppl = ss.People(10_000)

# What, and how: the environment is part of the disease
typhoid = ty.Typhoid(p_death=0.0)

## Interventions
# Behavioral change that halves susceptibility, from day 180 (was ty.behavioral_change(efficacy=0.5, start_year=2000.0+180.0/365))
sanitation = ss.modify('typhoid.rel_sus', multiply=1 - 0.5, start=2000.0+180.0/365, name='sanitation')

campaign_vax_0_2_yo = ty.vaccination_with_waning(
    start_year=pars["start"],
    prob=0.66,
    booster1_prob=0.0,
    booster2_prob=0.0,
    prob_type="annual",
    age_pars={'min_age': 0.0,
              'max_age': 2.0}
)

# Keep track of each individual's susceptibility over time: the base (rel_sus), and with the sanitation and the vaccine (eff_sus)
n_agents = len(ppl)
tracked = sc.objdict(rel_sus=[], eff_sus=[])
def tracker(sim):
    disease = sim.diseases.typhoid
    tracked.rel_sus.append(disease.rel_sus.raw[:n_agents].copy()) # By UID; there are no births
    tracked.eff_sus.append(disease.current_value('rel_sus').raw[:n_agents].copy())

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    analyzers=tracker,
    interventions=[sanitation] + campaign_vax_0_2_yo,
    )

sim.run()


age = sim.people.age.raw[:n_agents]
children_uids = ((age >= 0.75) & (age < 1.5)).nonzero()[0]
vaccinated_uids = np.isfinite(sim.interventions.vaccination.product.ti_delivered.raw[:n_agents]).nonzero()[0]
selected_uids = np.intersect1d(children_uids, vaccinated_uids)

fig, ax = sc.getrowscols(n=1, nrows=1, make=True)
ax.plot(sim.t.yearvec, np.array(tracked.eff_sus)[:, selected_uids])
ax.set_xlabel('Year')
ax.set_ylabel('Effective susceptibility')
ax.set_title('Vaccinated children aged 0.75-1.5 at the end')

plt.show()
