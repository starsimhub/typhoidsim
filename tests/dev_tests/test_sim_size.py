"""
Test continuing a sim that was saved part-way through, and how the size of a sim in
memory grows with the number of agents and the duration, before and after shrinking it
"""
import os
import tempfile

import sciris as sc
import starsim as ss
import typhoidsim as ty

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def make_sim(n_agents=1_000, dur=1.0, add_monitor=False):
    # Define the parameters
    pars = sc.objdict(
        start=2000,          # Starting year
        dur=ss.years(dur),   # Number of years to simulate
        dt=ss.days(1),       # Timestep of 1 day
        verbose=0,           # Print details of the run
        rand_seed=2,         # Set a non-default seed
    )

    # Who
    age_data = ty.utils.get_age_distribution_un(loc_type="Low-and-middle-income countries")
    ppl = ss.People(n_agents, age_data=age_data)

    # What, and how: the environment is part of the disease
    typhoid = ty.Typhoid()

    births = ss.Births(birth_rate=ss.peryear(10))
    deaths = ss.Deaths(death_rate=ss.peryear(10))
    demographics = [births, deaths]

    # The population by age group (was histograms_by_age_sex_monitor)
    stratify = dict(age=[0, 2, 5, 10, 15]) if add_monitor else None

    sim = ss.Sim(
        people=ppl,
        pars=pars,
        diseases=typhoid,
        demographics=demographics,
        stratify=stratify,
        label=f"n_agents_{n_agents}"
        )

    return sim


def test_continuation_saved(do_plot=False):
    """ A sim run part-way, saved, loaded, and run to the end gives the same results as one run straight through """
    sc.tic()
    # Start, run, and write the base sim to a file
    base_sim = make_sim(n_agents=2_000, dur=4.0, add_monitor=True)
    base_sim.run(until=2002)
    with tempfile.TemporaryDirectory() as folder:
        filename = os.path.join(folder, "base_sim.pkl.gz")
        base_sim.save(filename)
        sim = ss.load(filename)

    sim.label = "continuation"
    sim.run()
    sc.toc()

    ref = make_sim(n_agents=2_000, dur=4.0, add_monitor=True).run()
    for key in ['infections', 'infected', 'cfu']:
        assert np.allclose(sim.results.typhoid[key], ref.results.typhoid[key]), f'The continued sim differs from one run straight through in {key}'
    assert np.allclose(sim.results.typhoid.susceptible.by('age').values, ref.results.typhoid.susceptible.by('age').values)
    if do_plot:
        sim.plot()
        plt.show()
    return sim


def test_memory_size(do_plot=False):
    """ Shrinking a sim makes it smaller """
    y_full_size = []
    y_shrunk_size = []
    n_agents_lst = []
    dur_lst = []
    for n_agents in [1e3, 2e3, 5e3]:
        for dur in [1, 2, 4]: # Was up to 56 years and 50,000 agents
            sim = make_sim(n_agents=int(n_agents), dur=dur, add_monitor=True)
            sim.run()
            n_agents_lst.append(int(n_agents))
            dur_lst.append(np.round(dur, decimals=2))
            y_full_size.append(sc.checkmem(sim, descend=0).bytesize[0]/1e6)
            sim.shrink()
            y_shrunk_size.append(sc.checkmem(sim, descend=0).bytesize[0]/1e6)

    data = {"n_agents": n_agents_lst, "dur": dur_lst, "size_full": y_full_size, "size_shrunk": y_shrunk_size}
    df = pd.DataFrame(data)
    assert (df.size_shrunk < df.size_full).all(), f'Shrinking should make a sim smaller:\n{df}'
    if do_plot:
        print(df)
        tb_shrunk = pd.pivot_table(df, values="size_shrunk", columns="n_agents", index=["dur"])
        sns.heatmap(tb_shrunk, annot=True, cbar_kws={'label': 'size [MB]'})
        plt.show()
    return df


if __name__ == '__main__':
    do_plot = True
    test_memory_size(do_plot=do_plot)
    test_continuation_saved(do_plot=do_plot)
