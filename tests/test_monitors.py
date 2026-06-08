"""
Test that the histogram monitors are counting correctly
"""

import matplotlib.pyplot as plt
import numpy as np
import functools

import sciris as sc
import starsim as ss
import typhoidsim as ty

import pytest


def make_sim_with_histogram_monitor():
    # Define the parameters
    pars = sc.objdict(
        start=2000,
        dur=1.0,
        dt=1.0/365.0,
        verbose=0,
        rand_seed=2,
    )

    ppl = ss.People(10_000)
    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.05), "p_death":0.0})
    random_p2p = ss.RandomNet({'n_contacts': 5})

    vax1 = ty.blocking_vaccine(efficacy=0.2)
    elgibility = functools.partial(ty.eligibility_by_age, age_min=2.0, age_max=5.0)

    # Create the interventions
    my_intervention_vax1 = ss.routine_vx(
        start_year=pars["start"],
        prob=0.2,
        product=vax1,
        eligibility=elgibility
    )

    age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15-19', '20-39', '40-59',
                      '60+']

    to_record = dict(ti_infected=dict(path=("diseases", "typhoid"), label="infected"),
                     alive=dict(path=("people",)))

    m1_name = "monitor_cases"
    agg_sex = True
    monitor_cases = ty.histograms_by_age_sex_monitor(
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=to_record,
        resampling_period=1.0,
        aggregate_sex=agg_sex,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=pars["start"],
        name=m1_name)

    m2_name = "monitor_cases_vax"
    monitor_cases_vax = ty.histogram_by_vaccination_status(
        track_vaccinated=True,
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=to_record,
        resampling_period=1.0,
        aggregate_sex=agg_sex,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=pars["start"],
        name=m2_name)

    m3_name = "monitor_cases_unvax"
    monitor_cases_unvax = ty.histogram_by_vaccination_status(
        track_vaccinated=False,   # tracks unvaccinated
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=to_record,
        resampling_period=1.0,
        aggregate_sex=agg_sex,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=pars["start"],
        name=m3_name)

    m4_name = "monitor_people"
    monitor_people = ty.histograms_by_age_sex_monitor(
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=to_record,
        aggregate_time="mean",
        resampling_period=1.0,
        aggregate_sex=agg_sex,
        record_from=pars["start"],
        name=m4_name)

    sim = ss.Sim(pars=pars,
                 people=ppl,
                 diseases=typhoid,
                 networks=random_p2p,
                 interventions=[my_intervention_vax1],
                 analyzers=[monitor_cases, monitor_cases_vax, monitor_cases_unvax, monitor_people],
                 use_aging=False)

    return sim


def test_vaccinated_counts():
    sim = make_sim_with_histogram_monitor()
    sim.run()

    flat = sim.results.flatten()
    m1_name = "monitor_cases"
    m2_name = "monitor_cases_vax"
    m3_name = "monitor_cases_unvax"
    quantity = "_b_new_infected"

    age_bin_idx = 1 ## Check number for age_bin that received vaccines
    m1_infected = flat[f"{m1_name}{quantity}"]
    m2_infected = flat[f"{m2_name}{quantity}"]
    m3_infected = flat[f"{m3_name}{quantity}"]
    assert m1_infected[0, age_bin_idx] == (m2_infected[0, age_bin_idx] + m3_infected[0, age_bin_idx])
    return


@pytest.mark.parametrize("dt,rs", [
    (1.0/365, 0.2),
    (1.0/365, 1.0),
    (1.0/365, 1.0/52),
    (1.0/365, 1.0/12),    # monthly
    (1.0/365, 7.0/365),   # weekly
    (1.0/365, 15.0/365),  # fortnightly
    (1.0/365, 30.41/365)])
def test_histogram_monitor_aggregate_resampling(dt, rs):
    pars = sc.objdict(
        start=2000,
        dur=1.0,
        dt=dt,
        verbose=0,
        rand_seed=2,
    )

    ppl = ss.People(1_000)
    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.05), "p_death": 0.0})
    random_p2p = ss.RandomNet({'n_contacts': 5})
    to_record = dict(alive=dict(path=("people",)))

    m1_name = "monitor_people"
    agg_sex = True
    monitor_people = ty.histograms_by_age_sex_monitor(
        to_record=to_record,
        resampling_period=rs,
        aggregate_sex=agg_sex,
        aggregate_time="sum",
        # Sum over the resampling period
        record_from=pars["start"],
        name=m1_name)

    sim = ss.Sim(pars=pars,
                 people=ppl,
                 diseases=typhoid,
                 networks=random_p2p,
                 analyzers=[monitor_people],
                 use_aging=False)

    sim.run()

    assert sim.complete
    return


@pytest.mark.parametrize("dt,rs", [
    (1.0/365, 0.2),
    (1.0/365, 1.0),
    (1.0/365, 1.0/52),
    (1.0/365, 1.0/12),    # monthly
    (1.0/365, 7.0/365),   # weekly
    (1.0/365, 15.0/365),  # fortnightly
    (1.0/365, 30.41/365)])
def test_histogram_monitor_subsampling(dt, rs):
    pars = sc.objdict(
        start=2000,
        dur=1.0,
        dt=dt,
        verbose=0,
        rand_seed=2,
    )

    ppl = ss.People(1_000)
    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.05), "p_death": 0.0})
    random_p2p = ss.RandomNet({'n_contacts': 5})
    to_record = dict(alive=dict(path=("people",)))

    m1_name = "monitor_people"
    agg_sex = True
    monitor_people = ty.histograms_by_age_sex_monitor(
        to_record=to_record,
        resampling_period=rs,
        aggregate_sex=agg_sex,
        aggregate_time="subsample",
        # Sum over the resampling period
        record_from=pars["start"],
        name=m1_name)

    sim = ss.Sim(pars=pars,
                 people=ppl,
                 diseases=typhoid,
                 networks=random_p2p,
                 analyzers=[monitor_people],
                 use_aging=False)

    sim.run()

    assert sim.complete
    return


def test_by_age_counts():
    sim = make_sim_with_histogram_monitor()
    sim.init()
    init_ages = sim.people.age.raw.copy() # this population does not change and does not age
    sim.run()

    age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
    n_alive =  np.reshape(np.histogram(init_ages, bins=age_bin_edges)[0],(1, -1))

    flat = sim.results.flatten()
    m4_name = "monitor_people"
    quantity = "_b_n_alive"
    m4_alive = flat[f"{m4_name}{quantity}"]
    assert (m4_alive == n_alive).all()
    return


if __name__ == "__main__":
    test_vaccinated_counts()
    test_by_age_counts()
    test_histogram_monitor_aggregate_resampling(1.0/365, 7.0/365)
    test_histogram_monitor_subsampling(1.0/365, 7.0/365)
