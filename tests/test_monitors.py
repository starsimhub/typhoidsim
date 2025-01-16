"""
Test that the histogram monitors are counting correctly
"""

import matplotlib.pyplot as plt
import numpy as np
import functools

import sciris as sc
import starsim as ss
import typhoidsim as ty


def run_sim_with_vaccination():
    # Define the parameters
    pars = sc.objdict(
        start=2000,        # Starting year
        dur=1.0,       # Number of days to simulate
        dt=1.0/365.0,      # Timestep of 1 day, expressed in years
        verbose=0,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )


    demographics = [
        ss.Births(birth_rate=0),
        ss.Deaths(death_rate=0)
    ]

    ppl = ss.People(10_000)
    typhoid = ty.Typhoid(pars={"init_prev":ss.bernoulli(p=0.05)})
    random_p2p = ss.RandomNet({'n_contacts': 5})

    vax1 = ty.blocking_vaccine()
    elgibility = functools.partial(ty.eligibility_by_age, age_min=2.0, age_max=5.0)

    # Create the interventions
    my_intervention_vax1 = ss.routine_vx(
        start_year=2000,  # Begin vaccination in 2000
        prob=0.2,         # 20% coverage
        product=vax1,      # Use vax 1
        eligibility=elgibility
    )

    age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15-19', '20-39', '40-59',
                      '60+']

    record_cases = dict(ti_infected=dict(path=("diseases", "typhoid"), label="infected"))

    monitor_cases = ty.histograms_by_age_sex_monitor(
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
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

    sim = ss.Sim(pars=pars,
                 people=ppl,
                 diseases=typhoid,
                 networks=random_p2p,
                 interventions=[my_intervention_vax1],
                 analyzers=[monitor_cases, monitor_cases_vax, monitor_cases_unvax],
                 use_aging=False)

    sim.run()
    return sim

if __name__ == "__main__":
    run_sim_with_vaccination()