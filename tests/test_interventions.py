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
        n_years=1.0,  # Duration of the simulation in years
        dt=1.0/365.0,  # Timestep of 1 day, expressed in years
        verbose=0,  # Do not print details of the run
    )

    ppl = ss.People(10_000)

    typhoid = ty.Typhoid()

    random_p2p = ss.RandomNet({'n_contacts': 5})

    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=random_p2p)

    sim.initialize(verbose=False)

    # work out who to vaccinate
    in_trial = sim.people.typhoid.susceptible.uids
    n_vac = round(len(in_trial) * v_frac)
    in_vac = np.random.choice(in_trial, n_vac, replace=False)
    in_pla = np.setdiff1d(in_trial, in_vac)
    uids = ss.uids(in_vac)

    # create and apply the vaccination
    vac = ty.typhoid_vaccine(efficacy=efficacy, leaky=leaky)
    vac.init_pre(sim)
    vac.administer(sim.people, uids)

    # check the relative susceptibility at the start of the simulation
    rel_susc = sim.people.typhoid.rel_sus.values
    assert min(rel_susc[in_pla]) == 1.0, 'Placebo arm is not fully susceptible'
    if not leaky:
        assert min(
            rel_susc[in_vac]) == 0.0, 'Nobody fully vaccinated (all_or_nothing)'
        assert max(rel_susc[
                       in_vac]) == 1.0, 'Vaccine effective in everyone (all_or_nothing)'
        mean = n_vac * (1 - efficacy)
        sd = np.sqrt(n_vac * efficacy * (1 - efficacy))
        assert (np.mean(rel_susc[
                            in_vac]) - mean) / sd < tol, 'Incorrect mean susceptibility in vaccinated (all_or_nothing)'
    else:
        assert max(abs(rel_susc[in_vac] - (
                    1 - efficacy))) < 0.0001, 'Relative susceptibility not 1-efficacy (leaky)'

    # run the simulation until sufficient cases
    old_cases = []
    for idx in range(1000):
        sim.step()
        susc  = sim.people.typhoid.susceptible.uids
        cases = np.setdiff1d(in_trial, susc)
        if len(cases) > total_cases:
            break
        old_cases = cases

    if len(cases) > total_cases:
        cases = np.concatenate([old_cases,
                                np.random.choice(np.setdiff1d(cases, old_cases),
                                                 total_cases - len(old_cases),
                                                 replace=False)])
    vac_cases = np.intersect1d(cases, in_vac)

    # check to see whether the number of cases are as expected
    p = v_frac * (1 - efficacy) / (1 - efficacy * v_frac)
    mean = total_cases * p
    sd = np.sqrt(total_cases * p * (1 - p))
    assert (
                       len(vac_cases) - mean) / sd < tol, 'Incorrect proportion of vaccincated infected'

    # for all or nothing check that fully vaccinated did not get infected
    if not leaky:
        assert len(
            np.intersect1d(vac_cases, in_vac[rel_susc[in_vac] == 1.0])) == len(
            vac_cases), 'Not all vaccine cases amongst vaccine failures (all or nothing)'
        assert len(np.intersect1d(vac_cases, in_vac[rel_susc[
                                                        in_vac] == 0.0])) == 0, 'Vaccine cases amongst fully vaccincated (all or nothing)'

    if do_plot:
        sim.plot()

    return sim


def test_vaccine_leaky(do_plot=False):
    return run_sim_vaccine(0.3, False, do_plot=do_plot)


def test_vaccine_all_or_nothing(do_plot=False):
    return run_sim_vaccine(0.3, True, do_plot=do_plot)


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    leaky = test_vaccine_leaky(do_plot=do_plot)
    a_or_n = test_vaccine_all_or_nothing(do_plot=do_plot)

    T.toc()
