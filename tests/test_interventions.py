"""
Run tests of interventions
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import typhoidsim as ty


# def run_sim_vaccine(efficacy, leaky=True, do_plot=False):
#     # parameters
#     v_frac = 0.5  # fraction of population vaccinated
#     total_cases = 500  # total cases at which point we check results
#     tol = 3  # tolerance in standard deviations for simulated checks
#
#     # Define high-level simulation parameters
#     pars = dict(
#         start=2000,  # Starting year
#         dur=1.0,  # Duration of the simulation in years
#         dt=1.0/365.0,  # Timestep of 1 day, expressed in years
#     )
#
#     ppl = ss.People(10_000)
#
#     typhoid = ty.Typhoid()
#
#     random_p2p = ss.RandomNet({'n_contacts': 5})
#
#     sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=random_p2p)
#
#     sim.init(verbose=False)
#     sim.diseases.typhoid.init_pre(sim)
#
#     # work out who to vaccinate
#     eligible = sim.people.typhoid.susceptible.uids
#     n_vax = round(len(eligible) * v_frac)
#
#     is_vaxxed  = np.random.choice(eligible, n_vax, replace=False)
#     is_placebo = np.setdiff1d(eligible, is_vaxxed)
#     uids = ss.uids(is_vaxxed)
#
#     # create and apply the vaccination
#     vax = ty.typhoid_vaccine(efficacy=efficacy, leaky=leaky)
#     vax.init_pre(sim)
#     vax.administer(sim.people, uids)
#
#     # check the relative susceptibility at the start of the simulation
#     rel_sus = sim.people.typhoid.rel_sus.values
#     assert min(rel_sus[is_placebo]) == 1.0, 'Placebo arm is not fully susceptible'
#     if not leaky:
#         assert min(rel_sus[is_vaxxed]) == 0.0, 'Nobody fully vaccinated (all_or_nothing)'
#         assert max(rel_sus[is_vaxxed]) == 1.0, 'Vaccine effective in everyone (all_or_nothing)'
#         mean = (1.0 - efficacy)
#         sd = np.sqrt(efficacy * (1 - efficacy))
#         assert (np.mean(rel_sus[is_vaxxed]) - mean) / sd < tol, 'Incorrect mean susceptibility in vaccinated (all_or_nothing)'
#     else:
#         assert max(abs(rel_sus[is_vaxxed] - (1 - efficacy))) < 0.0001, 'Relative susceptibility not 1-efficacy (leaky)'
#
#     # run the simulation until sufficient cases
#     old_cases = []
#     for idx in range(1000):
#         sim.run_one_step()
#         susc  = sim.people.typhoid.infected.uids
#         cases = np.setdiff1d(is_vaxxed, susc)
#         if len(cases) > total_cases:
#             break
#         old_cases = cases
#
#     if len(cases) > total_cases:
#         cases = np.concatenate([old_cases, np.random.choice(np.setdiff1d(cases, old_cases),
#                                                             total_cases - len(old_cases),
#                                                             replace=False)])
#     vac_cases = np.intersect1d(cases, is_vaxxed)
#
#     # check to see whether the number of cases are as expected
#     p = v_frac * (1 - efficacy) / (1 - efficacy * v_frac)
#     mean = total_cases * p
#     sd = np.sqrt(total_cases * p * (1 - p))
#     #assert (len(vac_cases) - mean) / sd < tol, 'Incorrect proportion of vaccincated infected'
#
#     # for all or nothing check that fully vaccinated did not get infected
#     if not leaky:
#         assert len(
#             np.intersect1d(vac_cases, is_vaxxed[rel_sus[is_vaxxed] == 1.0])) == len(
#             vac_cases), 'Not all vaccine cases amongst vaccine failures (all or nothing)'
#         assert len(np.intersect1d(vac_cases, is_vaxxed[rel_sus[is_vaxxed] == 0.0])) == 0, 'Vaccine cases amongst fully vaccincated (all or nothing)'
#
#     if do_plot:
#         sim.plot()
#
#     return sim


def run_sim_base_test(prob_test, prob_test_positive, do_plot=False):

    def acute_eligibility(sim, **kwargs):
        """ Select individuals who just became acute and have not been screened"""
        # By default only test acute and people can be tested more than once
        acute_uids = ((sim.people.typhoid.ti_acute == sim.ti) & ~sim.interventions.routine_acute_screening.screened).uids
        return acute_uids

    # Define high-level simulation parameters
    pars = dict(
        start=2000,  # Starting year
        dur=1.0,  # Duration of the simulation in years
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        verbose=False
    )

    ppl = ss.People(20_000)
    init_prev = 0.5
    typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(p=init_prev), "p_death": 0.0})

    # create and apply the test intervention
    dx_product = ty.typhoid_test(pars=dict(sensitivity=ss.bernoulli(p=prob_test_positive)))
    screen_acute = ty.routine_acute_screening(product=dx_product, prob=prob_test, annual_prob=False, eligibility=acute_eligibility)  # Screen 30% of acute

    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, interventions=screen_acute)

    msim = ss.MultiSim(sims=sim, n_runs=10)
    msim.run()
    msim.reduce()

    actual_cum_screened = np.cumsum(msim.results['routine_acute_screening_new_screened'])[-1]
    actual_cum_positive = np.cumsum(msim.results['routine_acute_screening_new_positive'])[-1]

    assert np.isclose(actual_cum_screened / msim.results['n_alive'][-1],
                      init_prev * msim.sims[0].diseases.typhoid.pars.p_acute.pars.p * prob_test,
                      rtol=0.1)

    assert np.isclose(actual_cum_positive / msim.results['n_alive'][-1],
                      init_prev * msim.sims[0].diseases.typhoid.pars.p_acute.pars.p * prob_test * prob_test_positive,
                      rtol=0.1)

    if do_plot:
        sim.plot()

    return sim


def run_sim_with_wash(efficacy):
    # Define the parameters
    pars = sc.objdict(
        start=2000,  # Starting year
        dur=1.0,  # Number of days to simulate
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        rand_seed=2,  # Set a non-default seed
        verbose=0,
    )
    typhoid = ty.Typhoid()
    environment = ty.EnvironmentalPool()
    sanitation_efficacy = ty.Pattern("efficacy", pars={'efficacy': 0.5})
    sanitation = ty.behavioral_change(efficacy=efficacy)
    sim = ss.Sim(
        pars=pars,
        diseases=typhoid,
        demographics=environment,
        interventions=sanitation
    )

    sim.run()
    assert (sim.diseases.typhoid.rel_sus == efficacy).all()
    return sim


def make_sim_with_acute_screening(screen_coverage=1.0, test_sensitivity=1.0):

    def acute_eligibility(sim, **kwargs):
        """ Select individuals who just became acute and have not been screened"""
        # By default only test acute and people can be tested more than once
        acute_uids = ((sim.people.typhoid.ti_acute == sim.ti) & ~sim.interventions.routine_acute_screening.screened).uids
        return acute_uids

    # Define the parameters
    pars = sc.objdict(
        start=2000,
        dur=1.0,
        dt=1.0/365.0,
        n_agents=10_000,
        rand_seed=2,
        verbose=False,
    )
    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.1), "p_death": 0.0})

    # create and apply the test/screen intervention
    blood_test = ty.typhoid_test(pars=dict(sensitivity=ss.bernoulli(p=test_sensitivity)))
    screen_all_acute = ty.routine_acute_screening(product=blood_test,
                                                  prob=screen_coverage,
                                                  annual_prob=False,
                                                  eligibility=acute_eligibility)  # Screen 30% of all eligible population at each time step

    age_bin_edges = [0, ty.max_age]
    age_bin_labels = ['all']

    to_record = dict(
        ti_acute=dict(path=("diseases", "typhoid"), label="acute"),
        ti_positive=dict(path=("interventions", "routine_acute_screening"), label="tested_positive"),
        ti_screened=dict(path=("interventions", "routine_acute_screening"), label="screened"))

    monitor_cases = ty.histograms_by_age_sex_monitor(
        age_bins=age_bin_edges,
        age_bin_labels=age_bin_labels,
        to_record=to_record,
        aggregate_sex=True,
        record_from=2000.0)

    sim = ss.Sim(
        pars=pars,
        diseases=typhoid,
        interventions=screen_all_acute,
        analyzers=monitor_cases,
    )
    return sim


def test_screening_with_monitor():
    sim0 = make_sim_with_acute_screening()
    sim0.run()
    flat = sim0.results.flatten()
    res_acute = 'monitor_by_age_sex_b_new_acute'
    res_tested = 'monitor_by_age_sex_b_new_screened'
    res_positive = 'monitor_by_age_sex_b_new_positive'
    assert (flat[res_acute].sum() == flat[res_tested].sum() == flat[res_positive].sum())

    coverage = 0.5
    sim1 = make_sim_with_acute_screening(screen_coverage=coverage)
    msim = ss.MultiSim(sims=sim1, n_runs=10)
    msim.run()

    val = 0.0
    target_val = 0.0
    for sim in msim.sims:
        flat = sim.results.flatten()
        val += flat[res_tested].sum()
        target_val += flat[res_acute].sum()*coverage
    val /= len(msim.sims)
    target_val /= len(msim.sims)
    assert np.isclose(val, target_val, atol=1e-1)

    coverage = 0.5
    sensitivity = 0.6
    sim2 = make_sim_with_acute_screening(screen_coverage=coverage,
                                        test_sensitivity=sensitivity)
    msim = ss.MultiSim(sims=sim2, n_runs=10)
    msim.run()

    val = 0.0
    target_val = 0.0
    for sim in msim.sims:
        flat = sim.results.flatten()
        val += flat[res_positive].sum()
        target_val += flat[res_acute].sum()*coverage*sensitivity
    val /= len(msim.sims)
    target_val /= len(msim.sims)
    assert np.isclose(val, target_val, atol=1)
    return


def test_base_test(do_plot=False):
    return run_sim_base_test(0.3, 0.3, do_plot=do_plot)


def test_base_test_leaky(do_plot=False):
    return run_sim_base_test(0.3, 0.5, do_plot=do_plot)


def test_wash_behavior_change():
    return run_sim_with_wash(efficacy=0.5)


# def test_vaccine_leaky(do_plot=False):
#     return run_sim_vaccine(0.3, False, do_plot=do_plot)


# def test_vaccine_all_or_nothing(do_plot=False):
#     return run_sim_vaccine(0.3, True, do_plot=do_plot)


if __name__ == '__main__':
    T = sc.timer()
    do_plot = False
    test_base_test(do_plot=do_plot)
    test_base_test_leaky(do_plot=do_plot)
    #test_vaccine_leaky()
    test_wash_behavior_change()
    test_screening_with_monitor()
    T.toc()
