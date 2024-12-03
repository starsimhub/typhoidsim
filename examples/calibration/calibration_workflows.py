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
# import standard python libraries and numeric libraries if appropriate
import matplotlib.pyplot as plt

import sciris as sc
import typhoidsim as ty

#Import modules specific to this project
import models
import calibration_components as cbcomp

calib_debug = False  # If true, calibration will run in serial


def run_starsim_calibration_step_1(do_plot=True):
    # Make the sim, get the right reference data (in components), get
    # the correct calibration parameters
    sim = models.make_sim(scenario="baseline")
    components = cbcomp.get_calib_components(calibration_targets="cases_prevax")
    calib_pars = cbcomp.get_calib_pars(calibration_step="step_1")

    calib = ty.Calibration220(
            calib_pars=calib_pars,
            sim=sim,
            build_fn=cbcomp.parse_update_sim_pars,  # Tell the calibration how to update parameters
            components=components,
            total_trials=4,
            n_workers=8,
            die=True,
            debug=calib_debug
        )

    # Perform the calibration
    sc.printgreen('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)
    print(calib.best_pars)
    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.check_fit(n_runs=10)
    # Save fig
    if do_plot:
        #ty.plot_calib(calib)
        calib.plot_trend()
    plt.show()
    return calib


if __name__ == '__main__':
    run_starsim_calibration_step_1(do_plot=True)
