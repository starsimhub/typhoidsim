import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import sciris as sc
import typhoidsim as ty



def make_sim(saturation_age=20.0):
    # Define the parameters
    pars = sc.objdict(
        start=1990.0,      # Starting year
        n_years=2.0,       # Number of years to simulate
        dt=1.0/365.0,      # Timestep of 1 day, expressed in years
        use_aging=True,    # Use for debugging purposes, as a changing population may affect the results
        verbose=1,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )

    typhoids_pars = {"init_prev": None,  # Don't seed infections
                     "unexp2sus_saturation_age": saturation_age,
                     "unexp2sus_slope": 1.0}

    # Overwrite default probability-based transition mechanism unexposed -> susceptible
    typhoid = ty.Typhoid(pars=typhoids_pars)

    ppl = ss.People(10_000)

    age_bin_edges = np.arange(0, 31, 1)

    to_record = dict(unexposed=dict(path=("diseases", "typhoid"), label="unexposed"),
                     susceptible=dict(path=("diseases", "typhoid"), label="susceptible"),
                     alive=dict(path=("people",), label="num_agents"))

    monitor_unexposed = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                         to_record=to_record,
                                                         aggregate_sex=True,
                                                         aggregate_time="sum",
                                                         resampling_period=0.5,
                                                         name="monitor_1")

    sim = ss.Sim(
        people=ppl,
        analyzers=monitor_unexposed,
        pars=pars,
        diseases=typhoid,
        )
    return sim


sim1 = make_sim(saturation_age=20.0)
sim2 = make_sim(saturation_age=5.0)
sim1.run()
sim2.run()

report1 = sim1.get_analyzers()[0]
report2 = sim2.get_analyzers()[0]
report1.plot()
report1.plot_waterfall()
report2.plot_waterfall()
plt.show()