"""
Run a basic Typhoid simulation with seasonal fluctuations of environmental CFUs
"""
import functools
import  numpy as np
import starsim as ss
import typhoidsim as ty


# Define high-level simulation parameters
pars = dict(
    start    = 2004,  # Starting year
    dur      = 3.0,   # Duration of the simulation in years
    dt       = 1.0/365.0,     # Timestep of 1 day, expressed in years
    verbose  = 0,             # Do not print details of the run
)


# Disease
typhoid = ty.Typhoid()
environment = ty.EnvironmentalPool()
ppl = ss.People(10_000)


# Modulates the level of CFUs in the environment by reducing the shedding rate from people to the environment,
# Use a periodic rectangular pulse
# Parameters
seasonal_env_pars = {
    'period': 365.0,  # in days
    'peak_start_doy': 275.0,  # day of the year
    'ramp_up_dur': 175.26,    # duration in days
    'ramp_dw_dur': 100.0,     # duration in days
    'cutoff_dur': 20.0,       # duration in days
    'max_amp': 1.0}


def trapezoid(env_pars):
    if env_pars['ramp_up_dur'] + env_pars['ramp_dw_dur'] + env_pars['cutoff_dur'] > env_pars['period']:
        raise ValueError(f"the duration of the pattern is longer than the period")
    elif env_pars['ramp_up_dur'] + env_pars['ramp_dw_dur'] + env_pars['cutoff_dur'] == env_pars['period']:
        raise ValueError(f"the duration of the pattern is exactly the period, will get a triangular waveform")
    else:
        return functools.partial(ty.asym_trapezoidal, **env_pars)

start_day = (seasonal_env_pars["peak_start_doy"] - seasonal_env_pars["ramp_up_dur"]) / ty.days_per_year
start_year = 2004.0 + start_day
sanitation = ty.environmental_trapezoidal_modulation(efficacy=trapezoid(seasonal_env_pars), start_year=start_year)

sim = ss.Sim(
    pars=pars,
    diseases=typhoid,
    demographics=environment,
    interventions=sanitation
    )

sim.run()

import matplotlib.pyplot as plt
time_yearvec = sim.timevec
data2 = sim.interventions[0].results['effective_value']
data3 = sim.interventions[0].results['efficacy']
data4 = sim.interventions[0].target_baseline ## baseline value of rel_trans

# Check these two match
fig, ax = plt.subplots()
axx = ax.twinx()
ayy = ax.twiny()

time = (time_yearvec - start_year) * ty.days_per_year
ax.plot(time, data2, label="effective rel trans (from intervention)", ls=":", lw=3)
ax.plot(time, data4*np.ones(len(time)), label="baseline rel_trans", marker=".", ms=1)
axx.plot(time, data3, label="modulation amplitude", ls="-.", color="black")
ayy.plot(time_yearvec, data2, label="modulation amplitude", ls="-.", color="black")
ayy.spines['bottom'].set_position(('outward', 20))
ayy.xaxis.tick_bottom()

ax.set_ylabel("env rel_trans")
axx.set_ylabel("modulation amplitude")
#ax.set_xlabel("Days relative to intervention start")
ax.set_title("effective transmissibility = rel_trans_baseline * modulation amplitude")
ax.legend()
axx.legend()
plt.show()

