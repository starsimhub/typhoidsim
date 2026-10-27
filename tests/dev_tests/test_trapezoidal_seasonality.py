"""
Run a basic Typhoid simulation with seasonal fluctuations of environmental CFUs
"""
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import typhoidsim as ty


# Define high-level simulation parameters
pars = dict(
    start    = 2004,            # Starting year
    dur      = ss.years(3),     # Duration of the simulation
    dt       = ss.days(1),      # Timestep of 1 day
    verbose  = 0,               # Do not print details of the run
)


# Disease, with the environment as its only route
typhoid = ty.Typhoid(exposure2contact_rate=0)
ppl = ss.People(10_000)


# Modulates exposure to the environment, by scaling the share of the CFUs received in one exposure (env_rel_trans)
# with a periodic asymmetric trapezoidal pulse
# Parameters
trapezoidal_pars = {
    'period': 365.0,  # in days
    'peak_start_doy': 275.0,  # day of the year
    'ramp_up_dur': 175.26,    # duration in days
    'ramp_dw_dur': 100.0,     # duration in days
    'cutoff_dur': 20.0,       # duration in days
    'max_amp': 1.0}

start_day = (trapezoidal_pars["peak_start_doy"] - trapezoidal_pars["ramp_up_dur"]) / ty.days_per_year
start_year = 2004.0 + start_day

def modulation(year):
    """ The modulation amplitude in a year (or an array of them) """
    return ty.asym_trapezoidal(year*ty.days_per_year, **trapezoidal_pars)

# Was ty.environmental_trapezoidal_modulation(), which also set env_rel_trans to 0 before start_year
sanitation = ss.modify('typhoid.pars.env_rel_trans', multiply=lambda sim: modulation(sim.t.now('year')), start=start_year, name='sanitation')

sim = ss.Sim(
    pars=pars,
    people=ppl,
    diseases=typhoid,
    interventions=sanitation
    )

sim.run()

time_yearvec = sim.t.yearvec
baseline = sim.diseases.typhoid.pars.env_rel_trans.value # Baseline value of env_rel_trans
amplitude = np.where(time_yearvec >= start_year, modulation(time_yearvec), 1.0)
res = sim.results.typhoid
# The dose per exposure is env_rel_trans*CFUs/people, so this is the effective env_rel_trans; the results at each time point
# are of the step that ends there, so the dose of each step, from the second time point on, is that of the step starting at the one before
effective = res.dose[1:]/res.cfu[1:]*sim.results.alive[1:]

# Check these two match
assert np.allclose(effective, baseline*amplitude[:-1], rtol=1e-6), 'The effective env_rel_trans is not the baseline times the modulation'

fig, ax = plt.subplots()
axx = ax.twinx()
ayy = ax.twiny()

time = (time_yearvec - start_year) * ty.days_per_year
ax.plot(time[:-1], effective, label="effective env_rel_trans (from the dose)", ls=":", lw=3)
ax.plot(time, baseline*np.ones(len(time)), label="baseline env_rel_trans", marker=".", ms=1)
axx.plot(time, amplitude, label="modulation amplitude", ls="-.", color="black")
ayy.plot(time_yearvec[:-1], effective, label="effective env_rel_trans", ls="-.", color="black")
ayy.spines['bottom'].set_position(('outward', 20))
ayy.xaxis.tick_bottom()

ax.set_ylabel("env rel_trans")
axx.set_ylabel("modulation amplitude")
ax.set_title("effective transmissibility = rel_trans_baseline * modulation amplitude")
ax.legend()
axx.legend()
plt.show()
