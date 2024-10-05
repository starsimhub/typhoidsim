"""
This script shows how to perform kind of a parameter sweep, using a (random)
sampler from optuna (the library Starsim and typhoidsim) use on the backgrounds
to perform calibration.

The random samplers will always sample using whatever distribution are passed
 (i.e. a uniform or log-uniform distribution) and tell the trial to evaluate
 whatever comes out. We only need to pass a  "seed" argument to initialize the sampler.


Runtime: about 5 mins for 50 simulations.
"""
import matplotlib.pyplot as plt
import optuna

import starsim as ss
import typhoidsim as ty


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(vax_eff=1.0, vax_cov=1.0, start=2000.0):
    """
    Create a simulation instance of typhoid with a simple vaccination intervention.

    Args:
        vax_eff (float): The efficacy of the typhoid vaccine. A value between 0 and 1 where 1 represents 100% efficacy.
        vax_cov (float): The coverage of the vaccination. A value between 0 and 1 where 1 represents 100% of population coverage.
        start (float): The year when the vaccination intervention starts.

    Returns:
    sim (starsim.Sim): a starsim simulation configured with the same high level parameters for
    the populatioin and typhoid disease, but with slightly different intervention parameters.

    Example
    # Make a single simulation with a vaccine campaign where the vaccine has 70% efficacy and we cover 80% of the population
    sim = make_sim(vax_eff=0.7, vax_cov=0.8)
    sim.run()
    """

    # Define high-level simulation parameters
    pars = dict(
        start    =2000,          # Starting year
        n_years  =2.0,           # Duration of the simulation in years
        dt       =1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  =0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(1_000)

    # The disease
    typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(0.1)})

    # The contacts
    network = ss.RandomNet({'n_contacts': 10})

    # Create products
    vax1 = ty.typhoid_vaccine(efficacy=vax_eff)  # PARAMETER 1

    # Create the intervention
    my_intervention_vax1 = ss.routine_vx(
        start_year=start,    # PARAMETER 3
        prob=vax_cov,        # PARAMETER 2
        product=vax1         # Use vax 1
    )
    # Create the simulation with specific intervention parameters
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=network, interventions=my_intervention_vax1, label=f"Vax efficacy:{vax_eff}; coverage:{vax_cov}")
    return sim


# Why are we using hyperparameter optmization? It will sample in more detail
# interesting parts of the parameter space. This is useful if we want to get
# a quick idea of how the model behaves within a parameter space of 3 or
# more parameters, even if we are not trying to 'optimise' the parameters to
# match empirifcal data

# For instance if we did gridded search and had 2 parameters, with 10 values each,
# we would have to run 100 sims. If we then added another parameter and 10 values,
# we would have to run 1000 sims! There are probably lots of uninteresting parts of that space


def objective(trial):
    p1 = trial.suggest_float('vax_eff', 0.0, 1.0)
    p2 = trial.suggest_float('vax_cov', 0.0, 1.0)
    p3 = trial.suggest_float('start_year', 2000.0, 2001.0)

    # Create sim with parameter values chosen by the sampler, from the ranges specified above
    sim = make_sim(vax_eff=p1, vax_cov=p2, start=p3)
    sim.run()

    # For the sake of this example, we will use the total number of acute cases as the "cost" we need to minimise with
    # the vaccination intervention.
    res = sim.results.flatten()
    cost = res["typhoid_new_acute"].sum()
    return cost


# Run the sweep/optimisation
random_sweep = optuna.study.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42))
n_samples = 50  # number of combinations of parameters we are going to explore
random_sweep.optimize(objective, n_trials=n_samples, n_jobs=4)

# Plot the results
from optuna.visualization.matplotlib import plot_contour
fig = plot_contour(random_sweep, params=['vax_eff', 'vax_cov', 'start_year'])
plt.show()
