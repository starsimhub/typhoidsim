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
        ss.Births(birth_rate=0),
        ss.Deaths(death_rate=0)  # Needed to use debug=True with the vax intervention, tracks immunity level of of every agent
    ]

    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.05), "p_death":0})

    network = ss.RandomNet({'n_contacts': 5})

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        prob=0.4,
        booster1_interval=1.0,  # interval between receiving first dose and first booster
        booster1_prob=0.5,
        start_year=2000.0,
        end_year=2002.0,
        prob_type="interval",
        imm_ve0=ss.constant(v=ty.imm_ve0_by_age(age_bins=[0.75, 2.0, 5.0, 15.0, 125.0],
                                                        vals=[1.0, 0.9, 0.3, 0.3])),
        debug=True,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 2.0,
                  'max_age': 5.0},
        name = "campaign_1"
        )


    campaign_vax_5_10_yo = ty.vaccination_with_waning(
        prob=0.8,
        booster1_interval=1.0,  # interval between receiving first dose and first booster
        booster1_prob=0.5,
        start_year=2000.0,
        end_year=2002.0,
        prob_type="interval",
        imm_ve0=ss.constant(v=ty.imm_ve0_by_age(age_bins=[0.75, 2.0, 5.0, 15.0, 125.0],
                                                        vals=[1.0, 0.9, 0.3, 0.3])),
        debug=True,  # only use for this example to keep track of each individual's acquired immunity level over time
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
plt.plot(sim.results.campaign_1.immunity[:, 0:100])
plt.show()
