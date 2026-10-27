"""
The transition from unexposed to susceptible with age: the share of each age who are
still unexposed over time, for two saturation ages (the age by which everyone is susceptible)
"""
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import sciris as sc
import typhoidsim as ty

age_bin_edges = np.arange(0, 31, 1)
states = ['unexposed', 'susceptible', 'prepatent', 'acute', 'subclinical', 'chronic', 'recovered']


def make_sim(saturation_age=20.0):
    # Define the parameters
    pars = sc.objdict(
        start=1990,        # Starting year
        dur=ss.years(2),   # Number of years to simulate
        dt=ss.days(1),     # Timestep of 1 day
        use_aging=True,    # Use for debugging purposes, as a changing population may affect the results
        verbose=1,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )

    typhoid = ty.Typhoid(init_prev=0.0, unexp2sus_saturation_age=saturation_age, unexp2sus_slope=1.0) # Don't seed infections

    sim = ss.Sim(
        people=ss.People(10_000),
        pars=pars,
        diseases=typhoid,
        stratify=dict(age=age_bin_edges), # The results by age (was histograms_by_age_sex_monitor)
        label=f'Saturation age {saturation_age:n}',
        )
    return sim


sim1 = make_sim(saturation_age=20.0)
sim2 = make_sim(saturation_age=5.0)
sim1.run()
sim2.run()

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax, sim in zip(axs[:2], [sim1, sim2]):
    res = sim.results.typhoid
    unexposed = res.unexposed.by('age').values
    alive = sum(res[state].by('age').values for state in states) # Everyone is in one of the states
    share = unexposed/np.maximum(alive, 1)
    years = sim.t.yearvec
    im = ax.pcolormesh(years, age_bin_edges, share.T, vmin=0, vmax=1)
    ax.set_xlabel('Year')
    ax.set_ylabel('Age (years)')
    ax.set_title(f'Share unexposed, {sim.label.lower()}')
    fig.colorbar(im, ax=ax)

    # The share susceptible (or past it) at the end, against the curve it starts on
    axs[2].plot(age_bin_edges, 1 - share[-1], 'o', label=f'{sim.label} (end)')
    saturation_age = sim.diseases.typhoid.pars.unexp2sus_saturation_age.value
    axs[2].plot(age_bin_edges, ty.sigmoid(age_bin_edges, saturation_age, 1.0), label=f'{sim.label} (curve)')
axs[2].set_xlabel('Age (years)')
axs[2].set_ylabel('Share not unexposed')
axs[2].legend()
plt.show()
