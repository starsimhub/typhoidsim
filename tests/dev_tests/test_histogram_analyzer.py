import starsim as ss
import typhoidsim as ty
import matplotlib.pyplot as plt

# Define high-level simulation parameters
pars = dict(
    start=2000,    # Starting year
    n_years=1.0,   # Duration of the simulation in years
    dt=1.0/365.0,  # Timestep of 1 day, expressed in years
    verbose=1,     # Do not print details of the run
)

ppl = ss.People(10_000)
init_p = 0.5
typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(p=init_p)})
# create and apply the test intervention
tst = ty.base_test(prob_t=0.3, prob_tp=1.0, eligibility=ty.eligibility_by_age)
age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']
# Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.

# to_record = dict(ti_infected=dict(path=("diseases", "typhoid")),
#                  alive=dict(path=("people",)),
#                  ti_positive=dict(path=("interventions", "base_test")),
#                  ti_tested=dict(path=("interventions", "base_test")),
#                  )

anz_1 = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                         age_bin_labels=age_bin_labels,
                                         modality="counts",
                                         name="hist_by_counts")

anz_2 = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                         age_bin_labels=age_bin_labels,
                                         modality="proportions",
                                         name="hist_by_props")

sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, interventions=tst, analyzers=[anz_1, anz_2])
sim.run()
sim.plot(key="hist_by_")
sim.plot(key="typhoid")
plt.show()
