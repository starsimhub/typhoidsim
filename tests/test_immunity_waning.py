"""
Test vaccination with waning immunity and boosters.
"""
import numpy as np
import sciris as sc
import starsim as ss
import typhoidsim as ty


def make_sim(n_agents=5_000, **kwargs):
    """ Vaccinate under-2s (66% a year), with two boosters, in a population with no infection """
    pars = sc.objdict(start=2000, dur=ss.years(1), total_pop=50e6, n_agents=n_agents, dt=ss.days(1), rand_seed=2, verbose=0)
    typhoid = ty.Typhoid(init_prev=0.0)
    vaccination = ty.vaccination_with_waning(
        start_year=2000.0,
        prob=0.66,
        booster1_interval=0.33, # Years between the first dose and the first booster
        booster1_prob=1.0,
        booster2_interval=0.66, # Years between the first dose and the second booster
        booster2_prob=1.0,
        prob_type='annual',
        age_pars=dict(min_age=0.0, max_age=2.0),
        **kwargs,
    )
    sim = ss.Sim(pars=pars, diseases=typhoid, interventions=vaccination, people=ss.People(n_agents, age_data=ss.uniform(0, 60)), use_aging=False)
    return sim


def run_sim(sim):
    sim.run()
    res = sim.results
    vx = sim.interventions.vaccination
    n_group = np.count_nonzero(sim.people.age < 2)
    assert sim.complete
    first = res.vaccination.delivered.sum()/sim.current_scale # In agents, since the results count people
    assert np.isclose(first/n_group, 0.66, atol=0.05), f'{first} of {n_group} vaccinated'
    b1, b2 = [res[f"vaccination_booster{i}"].delivered.sum()/sim.current_scale for i in [1, 2]]
    assert 0 < b2 < b1 <= first # Fewer have had time for the later boosters
    held = vx.product.held.uids
    t = ss.years((sim.ti - vx.product.ti_delivered[held])*sim.t.dt_year) # Years since the last dose
    protection = vx.product.waning(t, sim, held)
    assert len(held) and protection.max() > 0, 'No protection after vaccination'
    assert np.all((protection >= 0) & (protection <= 1))
    return sim


def test_waning_defaults():
    """ vaccination_with_waning, with the default immunity by age at vaccination """
    return run_sim(make_sim())


def test_waning_with_constant_dist():
    """ vaccination_with_waning, with immunity in other age bins """
    sim = make_sim(
        imm_constant_dur=ty.imm_constant_dur_by_age(age_bins=[0, 2, 5, 125], vals=[0.0, 1000.0, 0.0]),
        imm_decay_rate=ty.imm_decay_rate_by_age(age_bins=[0, 5, 10, 125], vals=ty.days_per_year/np.array([300.0, 200.0, 100.0])),
        imm_ve0=ty.imm_ve0_by_age(age_bins=[0, 2, 5, 125], vals=[1.0, 0.7, 0.5]),
    )
    return run_sim(sim)


def test_box_gamma_waning():
    """ Protection is full for the box, then decays as an exponential with the rate given """
    waning = ty.box_gamma_waning(ve0=0.9, constant_dur=0.5, rate=2.0)
    t = np.array([0.0, 0.25, 0.5, 1.0])
    expected = 0.9*np.array([1, 1, 1, np.exp(-2*0.5)])
    assert np.allclose(0.9*ty.box_gamma(t, 0.0, 0.5, 1.0, 2.0), expected)
    return waning


if __name__ == '__main__':
    test_waning_defaults()
    test_waning_with_constant_dist()
    test_box_gamma_waning()
