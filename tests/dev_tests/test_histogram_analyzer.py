import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define high-level simulation parameters
pars = dict(
    start=2010,    # Starting year
    n_years=1.0,   # Duration of the simulation in years
    dt=1.0/365.0,  # Timestep of 1 day, expressed in years
    verbose=1,     # Do not print details of the run
)

ppl = ss.People(10_000)
init_p = 0.5 # Large prevalence to see results
typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(p=init_p)})
# create and apply the test intervention
tst = ty.base_test(prob_t=0.3, prob_tp=1.0, eligibility=ty.eligibility_by_age)
age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15-19', '20-39', '40-59', '60+']

# Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.

# to_record = dict(ti_infected=dict(path=("diseases", "typhoid")),
#                  alive=dict(path=("people",)),
#                  ti_positive=dict(path=("interventions", "base_test")),
#                  ti_tested=dict(path=("interventions", "base_test")),
#                  )

# anz_1 = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
#                                          age_bin_labels=age_bin_labels,
#                                          aggregate_sex=True,
#                                          name="hist_by_counts")

age_bin_edges = [0, 2, 15, ty.max_age]
age_bin_labels = ['<2', '2-15', '15+']  # human readable labels

# Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.
record_cases = dict(ti_acute=dict(path=("diseases", "typhoid"), label="cases"))
record_population = dict(alive=dict(path=("people",)))

monitor_cases = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                 age_bin_labels=age_bin_labels,
                                                 to_record=record_cases,
                                                 resampling_period=30.44/365,
                                                 # Record data on a montly basis, so we can aggregate later
                                                 aggregate_sex=True,
                                                 aggregate_time="sum",
                                                 # Sum over the resampling period
                                                 record_from=2010.0,
                                                 name="monitor_1")

monitor_population = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                      age_bin_labels=age_bin_labels,
                                                      to_record=record_population,
                                                      resampling_period=30.44/365.0,
                                                      # Record data on a montly basis, so we can aggregate later
                                                      aggregate_sex=True,
                                                      aggregate_time="mean",
                                                      # Average the number of people over the resampling period
                                                      record_from=2010.0,
                                                      name="monitor_2")

sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, interventions=tst, analyzers=[monitor_cases, monitor_population])
sim.run()
timevec = sim.get_analyzers()[0].yearvec
ty.plot_sim(sim, key="monitor_", yearvec=timevec)
sim.plot(key="typhoid_new")
plt.show()
