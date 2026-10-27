"""
This scripts shows how to perform a parameter sweep using Starsim's parallel
runs (`ss.parallel()`). This code is intended to be run on a latpop to do a super quick
exploration over 1 or 2 parameters.
"""
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(vax_eff=1.0, vax_cov=1.0):
    """
    Create a simulation instance of typhoid with a simple vaccination intervention.

    Args:
        vax_eff (float): The efficacy of the typhoid vaccine. A value between 0 and 1 where 1 represents 100% efficacy.
        vax_cov (float): The coverage of the vaccination. A value between 0 and 1 where 1 represents 100% of population coverage.

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
        start    = 2000,          # Starting year
        dur      = ss.years(2),   # Duration of the simulation
        dt       = ss.days(1),    # Timestep of 1 day
        verbose  = 0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(1_000)

    # The disease
    typhoid = ty.Typhoid(init_prev=0.1)

    # The contacts
    network = ss.RandomNet(n_contacts=10)

    # The vaccination: each person not yet vaccinated is vaccinated with a probability of vax_cov a year
    vaccination = ss.vaccinate(efficacy=vax_eff, coverage=ss.probperyear(vax_cov), start=2000) # PARAMETERS 1 and 2

    # Create the simulation with specific intervention parameters
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=network, interventions=vaccination, label=f"Vax efficacy:{vax_eff}; coverage:{vax_cov}")
    return sim


# Explore vaccination coverage, leave vaccination efficacy fixed at 70%
# Create list to hold the multiple simulation instances
sims = sc.autolist()

for vx_cov in [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]:
    sims.append(make_sim(vax_cov=vx_cov, vax_eff=0.7))


# Run them
msim = ss.parallel(sims)
msim.plot('typhoid')
plt.show()
