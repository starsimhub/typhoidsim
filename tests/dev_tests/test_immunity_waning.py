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
        dur=2,           # Number of years to simulate
        pop_scale=None,    #
        n_agents=n_agents, #
        dt=1.0/365.0,      # Timestep of 1 day, expressed in years
        verbose=0,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )



    # Who
    ppl = ss.People(pars["n_agents"])

    demographics = [
        ss.Births(birth_rate=10),
        ss.Deaths(death_rate=10)  # Needed to use debug=True with the vax intervention, tracks immunity level of of every agent
    ]

    typhoid = ty.Typhoid(pars={"init_prev":ss.bernoulli(p=0.05),
                               "immunity_age_bins": [0.75, 2.0, 5.0, 15.0, 125.0],
                               # Define age bins to represent different immunity waning dynamics for each agent based on their age when receive a vaccination
                               "immunity_fixed_dur": [940.4, 240.91, 0.0, 0.0],
                               # Duration of fixed immunity in days, one value per age bin of interest
                               "immunity_decay": [505.27, 505.27, 505.27, 505.27],
                               # Decay time constant, in days, one value per age bin of interest
                               "immunity_max_acq_response":[0.0, 0.9, 0.3, 0.3],
                               # Maximum protection at t=0 of receiving a vaccine
                               })

    network = ss.RandomNet({'n_contacts': 5})

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        prob=0.4,
        booster1_interval=5.0,  # interval between receiving first dose and first booster
        booster1_prob=0.0,
        start_year=2000.0,
        end_year=2000.0+0.125,
        prob_type="interval",
        debug=False,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 2.0,
                  'max_age': 5.0},
        name = "campaign_1"
        )


    campaign_vax_5_10_yo = ty.vaccination_with_waning(
        prob=0.8,
        booster1_interval=5.0,  # interval between receiving first dose and first booster
        booster1_prob=0.0,
        start_year=2000.0,
        end_year=2000.5,
        prob_type="interval",
        debug=False,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 5.0,
                  'max_age': 10.0},
        name="campaign_2"
        )

    age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15-19', '20-39', '40-59',
                      '60+']

    record_cases = dict(ti_infected=dict(path=("diseases", "typhoid"), label="infected"))


    monitor_cases = ty.histograms_by_age_sex_monitor(
        # age_bins=age_bin_edges,
        # age_bin_labels=age_bin_labels,
        to_record=record_cases,
        resampling_period=1.0,
        # Record data on a montly basis, so we can aggregate later
        aggregate_sex=True,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=2000.0,
        name="monitor_cases")

    monitor_cases_vax = ty.histogram_by_vaccination_status(
        track_vaccinated=True,
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=record_cases,
        resampling_period=1.0,
        # Record data on a montly basis, so we can aggregate later
        aggregate_sex=True,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=2000.0,
        name="monitor_vax")

    monitor_cases_unvax = ty.histogram_by_vaccination_status(
        track_vaccinated=False,   # tracks unvaccinated
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=record_cases,
        resampling_period=1.0,
        # Record data on a montly basis, so we can aggregate later
        aggregate_sex=True,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=2000.0,
        name="monitor_unvax")


    #
    sim = ss.Sim(
        pars=pars,
        demographics=demographics,
        diseases=typhoid,
        networks=network,
        interventions=[campaign_vax_2_5_yo, campaign_vax_5_10_yo],
        analyzers=[monitor_cases, monitor_cases_vax, monitor_cases_unvax],
        people=ppl,
        label=f"n_agents={pars['n_agents']}"
        )
    return sim


sim = make_sim()
sim.run()
sim.plot()
plt.show()
