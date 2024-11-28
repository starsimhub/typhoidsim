import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import sciris as sc
import typhoidsim as ty



# Define the parameters
pars = sc.objdict(
    start=1990.0,      # Starting year
    n_years=1.0,       # Number of years to simulate
    dt=1.0/365.0,      # Timestep of 1 day, expressed in years
    use_aging=False,   # Use for debugging purposes, as a changing population may affect the results
    verbose=1,         # Print details of the run
    rand_seed=2,       # Set a non-default seed
)

typhoids_pars = {"init_prev": None,  # Don't seed infections
                 "unexp2sus_saturation_age": 20.0,
                 "unexp2sus_slope": 1.0}

# Overwrite default probability-based transition mechanism unexposed -> susceptible
typhoid = ty.Typhoid(pars=typhoids_pars)

ppl = ss.People(10_000)


age_bin_edges = np.arange(0, 31, 1)
age_bin_centers = (age_bin_edges[0:-1] + age_bin_edges[1:])/2

to_record = dict(unexposed=dict(path=("diseases", "typhoid"), label="unexposed"),
                 susceptible=dict(path=("diseases", "typhoid"), label="susceptible"),
                 alive=dict(path=("people",), label="num_agents"))

monitor_unexposed = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                     to_record=to_record,
                                                     aggregate_sex=True,
                                                     name="monitor_1")


sim = ss.Sim(
    people=ppl,
    analyzers=monitor_unexposed,
    pars=pars,
    diseases=typhoid,
    )
sim.run()

report = sim.get_analyzers()[0]
yearvec = report.yearvec
tidx = -1  # Look at the last year of
n = report.results["hist_b_alive"][tidx, :]
plt.bar(age_bin_centers, (report.results["hist_b_susceptible"][tidx, :]/n).T)
plt.ylabel("Proportion of susceptible")
plt.xlabel("Age (years)")
plt.show()
