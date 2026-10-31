"""
Infections and people by age group, over time: from the sim's own results, stratified by age
(was histograms_by_age_sex_monitor)
"""
import numpy as np
import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define high-level simulation parameters
pars = dict(
    start=2010,        # Starting year
    dur=ss.years(1),   # Duration of the simulation
    dt=ss.days(1),     # Timestep of 1 day
    verbose=1,         # Print details of the run
)

ppl = ss.People(10_000)
init_p = 0.05 # Large prevalence to see results
typhoid = ty.Typhoid(init_prev=init_p)
network = ss.RandomNet(n_contacts=4)

# Create and apply the test intervention: screen 30% of people as they become acute
blood_test = ss.Diagnostic(sensitivity=0.65)
screen_all = ss.screen(blood_test, who=typhoid.onset.to(typhoid.acute), coverage=0.3, name='screening')

age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60]
states = ['unexposed', 'susceptible', 'prepatent', 'acute', 'subclinical', 'chronic', 'recovered']

sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=network, interventions=screen_all, stratify=dict(age=age_bin_edges))
sim.run()
sim.plot()

# Cases by age, summed over each week; and the people in each age group, at the end of each week
res = sim.results.typhoid
cases = res.infections.by('age').resample('week')
population = sum(res[state].by('age') for state in states).resample('week') # Everyone is in one of the states

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, r, title in zip(axs, [cases, population], ['Infections per week', 'People']):
    for i, label in enumerate(r.columns):
        ax.plot(r.timevec, r.values[:, i], label=label)
    ax.set_title(f'{title}, by age')
    ax.legend()
plt.show()
