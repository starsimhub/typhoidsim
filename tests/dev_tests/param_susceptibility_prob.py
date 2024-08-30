"""
Showcase age-based mechanism to move agents from an 'unsusceptible' state
to 'susceptible'.
"""

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import typhoidsim as ty

typhoid = ty.Typhoid()
age = np.linspace(0, 101, 100)

# This is the function called by typhoid.susceptibilty_prob_function() during a simulation
p_sus = ty.utils_math.sigmoid(age, typhoid.pars['sus_saturation_age'],
                              typhoid.pars['sus_age_exposure_slope'])

delta_slope = np.linspace(-1.0, 1.0, 21)

with sc.options.with_style('fancy'):

    for slope in delta_slope:
            p_sus_shifted = ty.utils_math.sigmoid(age, typhoid.pars['sus_saturation_age'],
                                                       typhoid.pars['sus_age_exposure_slope'] + slope)

            if slope > 0:
                color = [0.7, 0.0, 0.0]
            else:
                color = [0.0, 0.0, 0.7]
            plt.plot(age, p_sus_shifted, color=color, alpha=np.abs(slope))
    plt.plot(age, p_sus, color='black')

    plt.vlines(typhoid.pars['sus_saturation_age'], 0, 2, linestyle='dashed', color='black')
    plt.hlines(1, age[0], age[-1], linestyle='dashed', color='black')
    plt.xlabel('Age (years)')
    plt.ylabel('p_sus (or fraction of susceptible population aged X years)')
    plt.ylim([0, 1.01])
    plt.xlim([age[0], typhoid.pars['sus_saturation_age']])
    plt.show()
