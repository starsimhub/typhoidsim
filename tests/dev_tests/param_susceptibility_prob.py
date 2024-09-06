"""
Showcase age-based mechanism to move agents from an 'unsusceptible' state
to 'susceptible'.
"""

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import typhoidsim as ty

age = np.linspace(0, 101, 100)

# This is the function called by typhoid.susceptibilty_prob_function() during a simulation
age_thr = 20.0 # years
slope = 1.0    #
p_sus = ty.utils_math.sigmoid(age, 20, slope)

delta_slope = np.linspace(-1.0, 1.0, 21)

with sc.options.with_style('fancy'):

    for ds in delta_slope:
            p_sus_shifted = ty.utils_math.sigmoid(age, age_thr, slope + ds)

            if ds > 0:
                color = [0.7, 0.0, 0.0]
            else:
                color = [0.0, 0.0, 0.7]
            plt.plot(age, p_sus_shifted, color=color, alpha=np.abs(slope))
    plt.plot(age, p_sus, color='black')

    plt.vlines(age_thr, 0, 2, linestyle='dashed', color='black')
    plt.hlines(1, age[0], age[-1], linestyle='dashed', color='black')
    plt.xlabel('Age (years)')
    plt.ylabel('p_sus (or fraction of susceptible population aged X years)')
    plt.ylim([0, 1.01])
    plt.xlim([age[0], age_thr+0.1])
    plt.show()
