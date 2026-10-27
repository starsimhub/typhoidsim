"""
At the end of the prepatent stage individuals move to either acute or subclinical stage.
The proportion of individuals who move to acute infections is determined by p_acute parameter.
"""

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    dur=ss.days(30),  # Number of days to simulate
    dt=ss.days(1),    # Timestep of 1 day
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

typhoid = ty.Typhoid(init_prev=0.99, p_acute=0.234)

ppl = ss.People(10_000)

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    )
sim.run()


res = sim.results.typhoid
acute = res['onset.acute'].sum()
subclinical = res['onset.subclinical'].sum()

fig, ax = plt.subplots(1, 1)

bars = ax.bar(["Prepatent->Acute", "Prepatent->Subclinical"], [acute, subclinical])
ax.set_ylabel("Occurrences")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment
plt.suptitle(f"p_acute={typhoid.pars.p_acute.value}")

plt.show()
