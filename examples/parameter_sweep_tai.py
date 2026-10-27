"""
This script performs a parameter sweep over Typhoid acute infectiousness (TAI)
"""
import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(tai=42_000):
    """
    Create a simulation instance of typhoid with a simple vaccination intervention.
    """

    # Define high-level simulation parameters
    pars = dict(
        start    = 2000,          # Starting year
        dur      = ss.years(20),  # Duration of the simulation
        dt       = ss.days(1),    # Timestep of 1 day
        verbose  = 0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(10_000)

    # The disease, with only the environmental route
    typhoid = ty.Typhoid(
        init_prev = 0.001,
        tai = tai,
        exposure2contact_rate = 0,  # No transmission by contact
        teer = ss.perday(1.99),     # TEER: Typhoid environmental exposure rate
        env_rel_trans = 0.00001,    # Equivalent to the mEL parameter in Gauld et al. 2018
    )

    # Create the simulation with specific intervention parameters
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, label=f"TAI:{tai}")
    return sim


# Create list to hold the multiple simulation instances
sims = sc.autolist()

# Values of TAI
for val in [1_000, 5_000, 10_0000, 20_000, 42_000]:
    sims.append(make_sim(tai=val))

# Run them
msim = ss.parallel(sims)
msim.plot('typhoid')
plt.show()
