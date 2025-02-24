import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define high-level simulation parameters
pars = dict(
    start=2010,    # Starting year
    dur=1.0,   # Duration of the simulation in years
    dt=1.0/365.0,  # Timestep of 1 day, expressed in years
    verbose=1,     # Do not print details of the run
)

ppl = ss.People(10_000)
init_p = 0.05 # Large prevalence to see results
typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(p=init_p)})
network = ss.RandomNet({'n_contacts': 4})

# create and apply the test intervention
blood_test   = ty.typhoid_test(pars=dict(sensitivity=ss.bernoulli(p=0.65)))
screen_all= ty.routine_acute_screening(product=blood_test, prob=0.3, annual_prob=False,
                                       eligibility=ty.eligibility_by_age)  # Screen 30% of all eligible population at each time step

age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60, ty.max_age]
age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15-19', '20-39', '40-59', '60+']


# Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.
record_cases = dict(ti_infected=dict(path=("diseases", "typhoid"), label="infected"))

record_population = dict(alive=dict(path=("people",)))

monitor_cases = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                 age_bin_labels=age_bin_labels,
                                                 to_record=record_cases,
                                                 resampling_period=7.0/365,
                                                 # Record data on a montly basis, so we can aggregate later
                                                 aggregate_sex=True,
                                                 aggregate_time="sum",
                                                 # Sum over the resampling period
                                                 record_from=2010.0,
                                                 name="monitor_1")

monitor_population = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                      age_bin_labels=age_bin_labels,
                                                      to_record=record_population,
                                                      resampling_period=7.0/365.0,
                                                      # Record data on a montly basis, so we can aggregate later
                                                      aggregate_sex=True,
                                                      aggregate_time="mean",
                                                      # Average the number of people over the resampling period
                                                      record_from=2010.0,
                                                      name="monitor_2")

sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=network,
             interventions=screen_all, analyzers=[monitor_cases, monitor_population])
sim.run()
sim.plot()
sim.analyzers[0].plot()
sim.analyzers[0].plot_waterfall()
plt.show()
