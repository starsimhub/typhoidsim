"""
This scripts shows how to perform a parameter sweep using stasrsim's Multisim
functionality. This code is intended to be run on a latpop to do a super quick
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
        start    =2000,          # Starting year
        dur      =2.0,           # Duration of the simulation in years
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
        start_year=2000.0,   # Beginning of vaccination campaign
        prob=vax_cov,        # PARAMETER 2
        product=vax1         # Use vax 1
    )
    # Create the simulation with specific intervention parameters
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, networks=network, interventions=my_intervention_vax1, label=f"Vax efficacy:{vax_eff}; coverage:{vax_cov}")
    return sim


# Explore vaccination coverage, leave vaccination efficacy fixed at 70%
# Create list to hold the multiple simulation instances
sims = sc.autolist()

for vx_cov in [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]:
    sims.append(make_sim(vax_cov=vx_cov, vax_eff=0.7))


# Run them
msim = ss.MultiSim(sims)
msim.run()
msim.plot(key="typhoid")
plt.show()
