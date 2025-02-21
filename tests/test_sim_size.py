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
        start=2000,       # Starting year
        dur=dur,      # Number of days to simulate
        dt=1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose=0,        # Print details of the run
        rand_seed=2,      # Set a non-default seed
    )

    # Who
    age_data = ty.utils.get_age_distribution_un(loc_type="Low-and-middle-income countries")
    ppl = ss.People(n_agents, age_data=age_data)


    # What
    typhoid = ty.Typhoid()


    births = ss.Births(pars={'birth_rate': 10})
    deaths = ss.Deaths(pars={'death_rate': 10})

    # How
    environment = ty.EnvironmentalPool()

    demographics = [births, deaths, environment]

    monitors = []
    if add_monitor:
        record_n = dict(alive=dict(path=("people",)))
        monitor_age_bin_edges = [0, 2, 5, 10, 15, 125]
        monitor_population = ty.histograms_by_age_sex_monitor(
            to_record=record_n,
            age_bins=monitor_age_bin_edges,
            resampling_period=0.333,
            aggregate_sex=True,
            aggregate_time="mean",
            scaling=1.0,
            record_from=pars["start"],
            record_until=pars["start"] + pars["dur"],
            name="monitor_population")
        monitors.append(monitor_population)

    sim = ss.Sim(
        people=ppl,
        pars=pars,
        diseases=typhoid,
        analyzers=monitors,
        label=f"n_agents_{n_agents}"
        )

    return sim


def test_continuation_saved():
    sc.tic()
    # start, run, and write base sims as individual files
    base_sim = make_sim(n_agents=10_000, dur=10.0, add_monitor=True)
    base_sim.run(until=2005.0)
    base_sim.save("base_sim.pkl.gz")
    sim = sc.loadobj("base_sim.pkl.gz")

    sim.analyzers["monitor_population"].scaling = 0.5
    sim.label = f"continuation"

    sim.run()
    sc.toc()
    sim.plot()
    plt.show()
    return

y_full_size = []
y_shrunk_size = []
n_agents_lst = []
dur_lst = []
for n_agents in [1e4, 2e4, 5e4]:
    for dur in np.logspace(0, 1.75 , base=10, num=5):
        sim = make_sim(n_agents=int(n_agents), dur=dur, add_monitor=True)
        sim.run()
        n_agents_lst.append(int(n_agents))
        dur_lst.append(np.round(dur, decimals=2))
        full_size = sc.checkmem(sim, descend=0).bytesize[0]/1e6
        y_full_size.append(sc.checkmem(sim, descend=0).bytesize[0]/1e6)
        sim.shrink()
        y_shrunk_size.append(sc.checkmem(sim, descend=0).bytesize[0]/1e6)


data = {"n_agents": n_agents_lst, "dur": dur_lst, "size_full": y_full_size, "size_shrunk": y_shrunk_size}
df = pd.DataFrame(data)
branch = "refactored_monitor"
df.to_csv(f"test_sim_size_{branch}")
tb_full = pd.pivot_table(df, values="size_full", columns="n_agents", index=["dur"])
tb_shrunk = pd.pivot_table(df, values="size_shrunk", columns="n_agents", index=["dur"])

sns.heatmap(tb_shrunk, annot=True, cbar_kws={'label': 'size [MB]'})
plt.show()

test_continuation_saved()
