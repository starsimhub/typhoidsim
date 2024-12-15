"""
Test how simulations with environmental transmission scales with respect to n_agents
"""

import matplotlib.pyplot as plt
import numpy as np

import sciris as sc
import starsim as ss
import typhoidsim as ty


def make_sim(n_agents=10_000):

    # Define the parameters
    pars = sc.objdict(
        start=2000,        # Starting year
        n_years=10.0,       # Number of days to simulate
        total_pop=50e6,    # Total population size
        pop_scale=None,    #
        n_agents=n_agents, #
        dt=1.0/365.0,      # Timestep of 1 day, expressed in years
        verbose=0,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )



    # Who
    ppl = ss.People(pars["n_agents"])

    demographics = [
        ty.Births(birth_rate=0),
        ss.Deaths(death_rate=0)
    ]

    typhoid = ty.Typhoid(pars={"init_prev":ss.bernoulli(p=0.0)})

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        start_year=2000.0,
        prob=0.66,
        dose_interval=5.0, # interval between receiving first dose and booster
        booster_prob=1.0,
        annual_prob=True,
        debug=True,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 0.0,
                  'max_age': 2.0}
        )


    #
    sim = ss.Sim(
        pars=pars,
        demographics=demographics,
        diseases=typhoid,
        interventions=campaign_vax_2_5_yo,
        people=ppl,
        label=f"n_agents={pars['n_agents']}"
        )
    return sim



sim = make_sim()
sim.run()
flat = sim.results.flatten()
vax = flat["vaccination_with_waning_immunity"]
are_kids  = (sim.people.age >= 10.0) & (sim.people.age < 15)
are_vaccinated = sim.interventions[0].vaccinated
data_imm = vax[:, (are_kids & are_vaccinated)]

idx = 0
plt.plot(flat["yearvec"], data_imm[:, 0:10])

plt.show()
