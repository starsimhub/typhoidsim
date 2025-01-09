"""
Run tests of interventions
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import typhoidsim as ty


def run_sim_vaccine(efficacy, leaky=True, do_plot=False):
    # parameters
    v_frac = 0.5  # fraction of population vaccinated
    total_cases = 500  # total cases at which point we check results
    tol = 3  # tolerance in standard deviations for simulated checks

    # Define high-level simulation parameters
    pars = dict(
        start=2000,  # Starting year
        dur=1.0,  # Duration of the simulation in years
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        verbose=0,  # Do not print details of the run
    )

    ppl = ss.People(10_000)

    typhoid = ty.Typhoid()

    random_p2p = ss.RandomNet({'n_contacts': 5})

    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=random_p2p)

    sim.init(verbose=False)
    sim.diseases.typhoid.update_pre()

    # work out who to vaccinate
    eligible = sim.people.typhoid.susceptible.uids
    n_vax = round(len(eligible) * v_frac)

    is_vaxxed  = np.random.choice(eligible, n_vax, replace=False)
    is_placebo = np.setdiff1d(eligible, is_vaxxed)
    uids = ss.uids(is_vaxxed)

    # create and apply the vaccination
    vax = ty.typhoid_vaccine(efficacy=efficacy, leaky=leaky)
    vax.init_pre(sim)
    vax.administer(sim.people, uids)

    # check the relative susceptibility at the start of the simulation
    rel_sus = sim.people.typhoid.rel_sus.values
    assert min(rel_sus[is_placebo]) == 1.0, 'Placebo arm is not fully susceptible'
    if not leaky:
        assert min(rel_sus[is_vaxxed]) == 0.0, 'Nobody fully vaccinated (all_or_nothing)'
        assert max(rel_sus[is_vaxxed]) == 1.0, 'Vaccine effective in everyone (all_or_nothing)'
        mean = (1.0 - efficacy)
        sd = np.sqrt(efficacy * (1 - efficacy))
        assert (np.mean(rel_sus[is_vaxxed]) - mean) / sd < tol, 'Incorrect mean susceptibility in vaccinated (all_or_nothing)'
    else:
        assert max(abs(rel_sus[is_vaxxed] - (1 - efficacy))) < 0.0001, 'Relative susceptibility not 1-efficacy (leaky)'

    # run the simulation until sufficient cases
    old_cases = []
    for idx in range(1000):
        sim.step()
        susc  = sim.people.typhoid.infected.uids
        cases = np.setdiff1d(is_vaxxed, susc)
        if len(cases) > total_cases:
            break
        old_cases = cases

    if len(cases) > total_cases:
        cases = np.concatenate([old_cases, np.random.choice(np.setdiff1d(cases, old_cases),
                                                            total_cases - len(old_cases),
                                                            replace=False)])
    vac_cases = np.intersect1d(cases, is_vaxxed)

    # check to see whether the number of cases are as expected
    p = v_frac * (1 - efficacy) / (1 - efficacy * v_frac)
    mean = total_cases * p
    sd = np.sqrt(total_cases * p * (1 - p))
    #assert (len(vac_cases) - mean) / sd < tol, 'Incorrect proportion of vaccincated infected'

    # for all or nothing check that fully vaccinated did not get infected
    if not leaky:
        assert len(
            np.intersect1d(vac_cases, is_vaxxed[rel_sus[is_vaxxed] == 1.0])) == len(
            vac_cases), 'Not all vaccine cases amongst vaccine failures (all or nothing)'
        assert len(np.intersect1d(vac_cases, is_vaxxed[rel_sus[is_vaxxed] == 0.0])) == 0, 'Vaccine cases amongst fully vaccincated (all or nothing)'

    if do_plot:
        sim.plot()

    return sim


def run_sim_base_test(prob_test, prob_test_positive, do_plot=False):
    # Define high-level simulation parameters
    pars = dict(
        start=2000,  # Starting year
        dur=2.0/365.0,  # Duration of the simulation in years
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        verbose=0,  # Do not print details of the run
    )

    ppl = ss.People(20_000)
    init_prev = 0.5
    typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(p=init_prev)})
    # create and apply the test intervention
    dx_product = ty.typhoid_test(pars=dict(sensitivity=ss.bernoulli(p=prob_test_positive)))
    screen_acute = ty.routine_acute_screening(product=dx_product, prob=prob_test)  # Screen 30% of acute
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, interventions=screen_acute)
    sim.init(verbose=False)
    sim.run()
    # check the number of infected cases
    sim_prev = sum(sim.people.typhoid.infected)/sum(sim.people.alive)
    assert np.isclose(sim_prev, init_prev, atol=1e-1)

    mean_tests = sim.interventions.routine_acute_screening.results.new_screened.mean() / sum(sim.diseases.typhoid.infected)
    prob = init_prev * sim.diseases.typhoid.pars.p_acute.pars.p * prob_test
    assert np.isclose(mean_tests, prob, atol=1e-1)

    mean_positive = sim.interventions.routine_acute_screening.results.new_positive.mean() / (sum(sim.diseases.typhoid.acute) * prob_test)
    prob = init_prev * sim.diseases.typhoid.pars.p_acute.pars.p * prob_test_positive
    assert np.isclose(mean_positive, prob, atol=1e-1)

    if do_plot:
        sim.plot()

    return sim


def run_sim_with_wash(efficacy):
    # Define the parameters
    pars = sc.objdict(
        start=2000,  # Starting year
        dur=1.0,  # Number of days to simulate
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        verbose=1,  # Print details of the run
        rand_seed=2,  # Set a non-default seed
    )
    ppl = ss.People(10_000)
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
    assert (efficacy == sim.diseases.typhoid.rel_sus).all()
    return sim


def test_base_test(do_plot=False):
    return run_sim_base_test(0.3, 1.0, do_plot=do_plot)


def test_base_test_leaky(do_plot=False):
    return run_sim_base_test(0.3, 0.5, do_plot=do_plot)


def test_wash_behavior_change():
    return run_sim_with_wash()


# def test_vaccine_leaky(do_plot=False):
#     return run_sim_vaccine(0.3, False, do_plot=do_plot)


# def test_vaccine_all_or_nothing(do_plot=False):
#     return run_sim_vaccine(0.3, True, do_plot=do_plot)


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True
    test_base_test(do_plot=do_plot)
    test_base_test_leaky(do_plot=do_plot)
    test_base_test()
    #test_vaccine_all_or_nothing(do_plot=do_plot)

    T.toc()
