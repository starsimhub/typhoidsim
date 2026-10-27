"""
Test calibration with sim.fit(): the initial prevalence, from the number infected
"""

import numpy as np
import sciris as sc
import starsim as ss
import typhoidsim as ty

debug = False # If true, will run in serial
do_plot = 1


def make_sim(init_prev=ss.par(0.005, bounds=[0.001, 0.1]), data=None, rand_seed=1):
    """ No transmission, so the number infected is the infections at the start, until they recover (chronic carriers stay) """
    typhoid = ty.Typhoid(init_prev=init_prev, exposure2contact_rate=0, teer=0)
    sim = ss.Sim(n_agents=10e3, start=2000, dur=ss.years(2), dt=ss.weeks(1), diseases=typhoid, data=data, rand_seed=rand_seed, verbose=0)
    return sim


def test_calibration(do_plot=False):
    sc.heading('Testing calibration')
    truth = make_sim(init_prev=0.02, rand_seed=2).run()
    data = truth.to_data('typhoid.infected')
    sim = make_sim(data=data)
    fit = sim.fit(method='optuna', n_trials=30, n_runs=4, n_workers=1 if debug else 4, verbose=0)
    print(fit.summarize())
    trials = fit.to_df()
    best = trials.init_prev[trials.objective.idxmin()] # The best-scoring trial; fit.pars smooths over the trials (see TY10 in the port notes)
    assert np.isclose(best, 0.02, rtol=0.25), f'Fitted init_prev {best:n}, not 0.02'
    assert fit.mismatch < fit.mismatch_before
    if do_plot:
        fit.plot()
    return sim, fit


if __name__ == '__main__':
    T = sc.timer()
    sim, fit = test_calibration(do_plot=do_plot)
    T.toc()
