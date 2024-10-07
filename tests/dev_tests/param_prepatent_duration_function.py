"""
The duration of the prepatent stage of an individual’s infection is
calculated at the beginning of the infection as a draw from a log-normal
distribution, parameterised by two parameters: mu and sigma.

In EMOD one of three mu-sigma parameter pairs are uased depending on the
 "quantization" of the exposure amount into one of three
buckets using various thresholds. (Glynn et al., 1995).

In typhoidsim we use a double sigmoid function to achieve the "quantization"
effect, but the function allows for a continuous representation of the mean
and sigma parameters as a function of exposure dose. This is more efficient, as
we do not have to actually quantize the exposure dose of an individual.
"""

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty

# Define the parameters
pars = sc.objdict(
    start=2000,       # Starting year
    n_years=1.0,      # Number of days to simulate
    dt=1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose=1,        # Print details of the run
    rand_seed=2,      # Set a non-default seed
)

typhoid = ty.Typhoid()

ppl = ss.People(10_000)

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    )
sim.run()


fig, axs = plt.subplots(1, 2)
exposure_dose = np.linspace(0, 55000000, 1024)

mean_dur = sim.diseases.typhoid.partial_prep_dur_mean(exposure_dose)
std_dur  = sim.diseases.typhoid.partial_prep_dur_mean(exposure_dose)
axs[0].plot(exposure_dose, mean_dur)
axs[0].set_xlabel("Exposure dose (CFUs)")
axs[0].set_ylabel("Mean duration of prepatent stage (days)")

axs[1].plot(exposure_dose, std_dur)
axs[1].set_xlabel("Exposure dose (CFUs)")
axs[1].set_ylabel("Standard deviation of duration of prepatent stage (days)")

plt.show()
