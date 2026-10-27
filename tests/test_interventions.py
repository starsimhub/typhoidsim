"""
Run tests of interventions
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import typhoidsim as ty


def make_screening(coverage, sensitivity, typhoid):
    """ Screen people as they become acute, with a test of this sensitivity """
    test = ss.Diagnostic(sensitivity=sensitivity)
    return ss.screen(test, on=typhoid.onset.to(typhoid.acute), coverage=coverage, name='screening')


def run_sim_base_test(prob_test, prob_test_positive, do_plot=False):
    """ Screen a share of people as they become acute: the share screened and testing positive are as given """
    pars = dict(start=2000, dur=ss.years(1), dt=ss.days(1), verbose=False)
    init_prev = 0.5
    typhoid = ty.Typhoid(init_prev=init_prev, p_death=0.0, exposure2contact_rate=0, teer=0) # Only the infections at the start
    screening = make_screening(prob_test, prob_test_positive, typhoid)
    sim = ss.Sim(pars=pars, people=ss.People(20_000), diseases=typhoid, interventions=screening)
    sim.run()

    res = sim.results.screening
    n = sim.results.alive[-1]
    p_acute = sim.diseases.typhoid.pars.p_acute.value
    assert np.isclose(res.delivered.sum()/n, init_prev*p_acute*prob_test, rtol=0.1)
    assert np.isclose(res.positive.sum()/n, init_prev*p_acute*prob_test*prob_test_positive, rtol=0.1)
    if do_plot:
        sim.plot()
    return sim


def make_env_sim(*interventions, **kwargs):
    """ A sim with only the environmental route, strong enough to spread """
    pars = sc.objdict(start=2000, dur=ss.years(1), dt=ss.days(1), rand_seed=2, verbose=0, n_agents=5_000)
    typhoid = ty.Typhoid(exposure2contact_rate=0, env_rel_trans=1e2, **kwargs)
    return ss.Sim(pars=pars, diseases=typhoid, interventions=list(interventions))


def test_wash_behavior_change():
    """ Behavioral change (washing vegetables, handwashing) that removes the risk from the environment stops infection """
    sim0 = make_env_sim().run()
    behavior = ss.modify('typhoid.rel_sus', multiply=0.0, start=2000.5) # From the middle of the year
    sim1 = make_env_sim(behavior).run()
    half = sim0.t.npts//2 + 1
    inf0, inf1 = sim0.results.typhoid.environment, sim1.results.typhoid.environment
    assert inf0[half:].sum() > 0
    assert inf1[half:].sum() == 0
    assert np.array_equal(inf0[:half], inf1[:half]) # The same until it starts
    return sim1


def test_shedding_reduction():
    """ Sanitation that halves the shedding into the environment halves the CFUs in it, if nobody is infected through it """
    efficacy = 0.5
    kw = dict(teer=0, init_prev=0.05) # No exposure to the environment, so the people are the same in both
    sim0 = make_env_sim(**kw).run()
    sanitation = ss.modify('typhoid.pars.shedding_rate', multiply=1 - efficacy)
    sim1 = make_env_sim(sanitation, **kw).run()
    cfu0, cfu1 = sim0.results.typhoid.cfu, sim1.results.typhoid.cfu
    assert cfu0[-1] > 0
    assert np.allclose(cfu1, (1 - efficacy)*cfu0)
    return sim1


def make_sim_with_acute_screening(screen_coverage=1.0, test_sensitivity=1.0, seed=2):
    """ Screen everyone (or a share) as they become acute, in a sim with transmission """
    pars = sc.objdict(start=2000, dur=ss.years(2), dt=ss.days(1), n_agents=10_000, rand_seed=seed, verbose=False)
    typhoid = ty.Typhoid(init_prev=0.1, p_death=0.0)
    screening = make_screening(screen_coverage, test_sensitivity, typhoid)
    return ss.Sim(pars=pars, diseases=typhoid, interventions=screening, networks=ss.RandomNet(n_contacts=1))


def test_screening():
    """ Everyone who becomes acute is screened and tests positive; and with coverage and sensitivity below 1, the shares are as given """
    sim0 = make_sim_with_acute_screening().run()
    acute = sim0.results.typhoid['onset.acute'].sum()
    res = sim0.results.screening
    assert acute > 0
    assert acute == res.delivered.sum() == res.positive.sum()

    # Averaged over several seeds, to reduce the noise
    n_runs = 8
    coverage = 0.5
    sensitivity = 0.6
    tested, positive, acute = 0.0, 0.0, 0.0
    for seed in range(n_runs):
        sim = make_sim_with_acute_screening(screen_coverage=coverage, test_sensitivity=sensitivity, seed=seed).run()
        tested += sim.results.screening.delivered.sum()
        positive += sim.results.screening.positive.sum()
        acute += sim.results.typhoid['onset.acute'].sum()
    assert np.isclose(tested, acute*coverage, rtol=0.1)
    assert np.isclose(positive, acute*coverage*sensitivity, rtol=0.1)
    return sim


def test_acute_treatment():
    """ Treating acute cases, which makes them less infectious while acute, reduces transmission by contact, and what they shed """
    def run(*interventions):
        pars = sc.objdict(start=2000, dur=ss.years(1), dt=ss.days(1), n_agents=5_000, verbose=0)
        typhoid = ty.Typhoid(p_acute=0.9, tsri=0.2, tpri=0.1, tcri=0.1, teer=0, init_prev=0.002) # Mostly acute, which is where treatment acts; the environment infects nobody, but is still shed into
        sim = ss.Sim(pars=pars, diseases=typhoid, interventions=list(interventions), networks=ss.RandomNet(n_contacts=1))
        res = sim.run(n_runs=6, verbose=0).results.typhoid
        return res.infections.sum(axis=1).mean(), (res.shedding.sum(axis=1)/res.acute.sum(axis=1)).mean()
    base, base_shed = run()
    treated, treated_shed = run(ty.acute_treatment(efficacy=1.0, coverage=1.0, delay=ss.constant(ss.days(1))))
    assert treated < 0.8*base, f'Treatment should reduce infections: {treated:n} vs {base:n}'
    assert treated_shed < 0.5*base_shed, f'Treatment should reduce shedding per acute case: {treated_shed:n} vs {base_shed:n}'
    return


def test_base_test(do_plot=False):
    return run_sim_base_test(0.3, 0.3, do_plot=do_plot)


def test_base_test_leaky(do_plot=False):
    return run_sim_base_test(0.3, 0.5, do_plot=do_plot)


if __name__ == '__main__':
    T = sc.timer()
    do_plot = False
    test_base_test(do_plot=do_plot)
    test_base_test_leaky(do_plot=do_plot)
    test_wash_behavior_change()
    test_shedding_reduction()
    test_screening()
    test_acute_treatment()
    T.toc()
