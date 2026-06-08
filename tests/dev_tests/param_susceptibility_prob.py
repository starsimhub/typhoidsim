"""
Showcase age-based mechanism to move agents from an 'unsusceptible' state
to 'susceptible'.
"""

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import typhoidsim as ty

ages = np.linspace(0, 101, 100)

# This is the function called by typhoid.susceptibilty_prob_function() during a simulation
age_thr = 20.0  # years
slope = 0.0    #
p_sus = ty.utils_math.sigmoid(ages, age_thr, slope)

delta_slope =[-2.0, -1.0, -0.5, -0.25, 0.0, 0.5, 2.0]

neg = 0
pos = 0
with sc.options.with_style('fancy'):

    for ds in delta_slope:
            p_sus_shifted = ty.utils_math.sigmoid(ages, age_thr, ds)

            if ds > 0:
                color = [0.7, 0.0, pos*0.1]
                pos +=1
            else:
                color = [neg*0.1, 0.0, 0.7]
                neg += 1
            plt.plot(ages, p_sus_shifted, color=color, alpha=0.5, label=f"slope={ds}")
    plt.plot(ages, p_sus, color='black', label="slope=0")

    plt.vlines(age_thr, 0, 2, linestyle='dashed', color='black')
    plt.hlines(1, ages[0], ages[-1], linestyle='dashed', color='black')
    plt.xlabel('Age (years)')
    plt.ylabel('p_sus (or fraction of susceptible population aged X years)')
    plt.ylim([-0.1, 1.1])
    plt.xlim([ages[0], ages[-1]//2])
    plt.legend()
    plt.show()
