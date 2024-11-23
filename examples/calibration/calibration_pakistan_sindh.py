"""
This script contains step 1 of the calibration workflow to reproduce results from
Kraay et al. 2024 https://www.medrxiv.org/content/10.1101/2024.08.30.24312839v1

Country: Pakistan
Province: Sindh

Transmission routes:
- Environmental only

Period:
- Prior to vaccine introduction
- January 2018 - December 2019 (2 years)

Free parameters:
- typhoid acute infectiousness (TAI) expressed in CFUs
- lambda environment: average exposure rate to the environment (TEER), expressed in (in num exposures * volume / day)

Target:
- age distribution of blood culture confirmed typhoid cases
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

import sciris as sc
import starsim as ss
import typhoidsim as ty

import calibration_pakistan_utils as utils

calib_debug = True # If true, calibration will run in serial


def run_starsim_calibration_step_1(do_plot=True, option=1):
    """"
    Calibration, step 1, we use data across the whole population
    """

    # Define the calibration parameters
    calib_pars = dict(
        # typhoid acute infectiousness
        tai=dict(low=1e-4, high=1e5, guess=42_808, log=True),
        # typhoid environmental exposure rate
        teer=dict(low=0.0, high=10.0, guess=1.99)  # The path always consists of three components/steps.
    )

    # Make the sim and data
    sim = make_sim()

    #Option 1: with custom-made goodnes-of-fit function
    if option == 1:
        calib = ty.Calibration220(
            calib_pars=calib_pars,
            sim=sim,
            build_fn=build_sim,
            eval_fn=my_gof_fun,
            eval_kwargs=dict(expected_data=utils.load_empirical_data_pakistan()),  # loads all the data from TahirData_0928.csv without any preprocessing
            total_trials=16,
            n_workers=4,
            die=True,
            debug=calib_debug
        )
    elif option == 2:
        # Option 2: with calib components
        components = make_calib_components()
        calib = ty.Calibration220(
            calib_pars=calib_pars,
            sim=sim,
            build_fn=build_sim,
            components=components,
            total_trials=16,
            n_workers=4,
            die=True,
            debug=calib_debug
        )
    else:
        raise ValueError(f"Uknown calibration option: {option}")

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)

    calib.best_pars

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.check_fit(n_runs=5)

    # if do_plot:
    # NOTE: these plotting functions fail to work properly with the result arrays that hold the histograms by age and sex
    # This is a bug with the starsim framework (https://github.com/starsimhub/typhoidsim/issues/120)
    #     calib.plot_sims(key="typhoid")
    #     calib.plot_trend()
    plt.show()
    return calib



if __name__ == '__main__':
    #run_starsim_calibration_step_1(do_plot=True, option=2)
