"""
Test calibration - migrated from Starsim 2.2.0
"""

# %% Imports and settings
import pandas as pd
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss

import typhoidsim as ty

debug = False  # If true, will run in serial
do_plot = 1
do_save = 0
n_agents = 1e4


# %% Helper functions

def instantiate_sim():

    typhoid = ty.Typhoid()

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000.0,
        dur=5,
        dt=1.0,
        diseases=typhoid,
    )

    return sim


def parse_update_sim_pars(sim, calib_pars):
    """
    Modify the base unintialised simulation by applying calib_pars.
    This allows us to modify the parameters before initialising
    and running the sim.
    """
    typhoid = sim.pars.diseases  # There is only one disease in the sim
    # Capture any parameters that need special handling here
    for k, par in calib_pars.items():
        v = par['value']
        if k == 'init_prev':
            typhoid.pars.init_prev = ss.bernoulli(v)
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    return sim


# %% Define the tests
def test_calibration(do_plot=False):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = dict(
        # Log scale and no "path", will be handled by build_sim (ablve)
        init_prev=dict(low=0.001, high=0.1, guess=0.005,
                       path=('diseases', 'typhoid', 'init_prev')),
        )

    # Make the sim and data
    sim = instantiate_sim()

    infectious = ty.CalibComponent220(
        name='Infectious',

        # "expected" data generated from a simulation with pars
        #   init_prev=0.02
        expected=pd.DataFrame({
            'n': [9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
            'x': [6.0, 1.0, 0.0, 0.0, 0.0, 0.0],     # Number of individuals found to be infectious
        }, index=pd.Index(
            [2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0],
            name='t')),  # On these dates

        extract_fn=lambda sim: pd.DataFrame({
            'n': sim.results.n_alive,
            'x': sim.results.typhoid.n_infected,
        }, index=pd.Index(sim.results.yearvec, name='t')),

        conform='prevalent',
        nll_fn='beta',

        weight=1,
    )

    # Make the calibration
    calib = ty.Calibration220(
        calib_pars=calib_pars,
        sim=sim,
        build_fn=parse_update_sim_pars,  # If None, uses default builder, Calibration.translate_pars
        components=infectious,
        total_trials=20,
        n_workers=4,
        die=True,
        debug=debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    sc.printcyan('\nChecking fit...')
    calib.check_fit()
    print(f'Fit with original pars: {calib.before_fits}')
    print(f'Fit with best-fit pars: {calib.after_fits}')
    if calib.after_fits.mean() <= calib.before_fits.mean():
        print('✓ Calibration improved fit')
    else:
        print(
            '✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')

    if do_plot:
        calib.plot_sims()
        calib.plot_trend()

    return sim, calib


# %% Run as a script
if __name__ == '__main__':
    generate_surrogate_data = False
    # Useful for generating surrogate "expected" data
    if generate_surrogate_data:
        sim = instantiate_sim()
        pars = {
            'init_prev': dict(value=0.02),
        }
        sim = parse_update_sim_pars(sim, pars)
        ms = ss.MultiSim(sim)  # NOTE: if we define n_runs here, the multisim receives two differen values for n_runs, probably a clash between starsim 1.0.3 and 2.2.0
        ms.run().plot()

    T = sc.timer()
    do_plot = True
    sim, calib = test_calibration(do_plot=do_plot)
    T.toc()
    plt.show()
