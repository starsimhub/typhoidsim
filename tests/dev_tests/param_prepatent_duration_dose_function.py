"""
The duration of the prepatent stage of an individual’s infection is
calculated at the beginning of the infection as a draw from a log-normal
distribution, parameterised by two parameters: mu and sigma.

In EMOD one of three mu-sigma parameter pairs are uased depending on the
 "quantization" of the exposure amount into one of three
buckets using various thresholds. (Glynn et al., 1995).

In Typhoidsim, as in EMOD, the parameters are those of a low, medium, or high
dose, in two steps at cfu_lo_me and cfu_me_hi (`ty.typhoid.by_dose()`), by the
dose of each infection (`infection_dose`: by contact, a high dose).
"""

import numpy as np
import matplotlib.pyplot as plt

import typhoidsim as ty

typhoid = ty.Typhoid()
p = typhoid.pars
levels = ty.typhoid.prepatent_levels # The parameters at a low, medium, and high dose

fig, axs = plt.subplots(1, 2)
exposure_dose = np.linspace(0, 65000000, 1024)

# As the disease's prepatent_mean and prepatent_sigma, by the dose of each infection
mean_dur, std_dur = [ty.typhoid.by_dose(levels[key], exposure_dose, p.cfu_lo_me.value, p.cfu_me_hi.value) for key in ['mean_dur', 'std_dur']]
axs[0].plot(exposure_dose, mean_dur)
axs[0].set_xlabel("Exposure dose (CFUs)")
axs[0].set_ylabel("Mean of the log of the duration of prepatent stage (days)")

axs[1].plot(exposure_dose, std_dur)
axs[1].set_xlabel("Exposure dose (CFUs)")
axs[1].set_ylabel("Standard deviation of the log of the duration of prepatent stage (days)")

plt.show()
