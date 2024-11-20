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
        start    =2000,          # Starting year
        n_years  =20.0,           # Duration of the simulation in years
        dt       =1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  =0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(10_000)

    # The disease
    typhoid = ty.Typhoid(pars={'init_prev': ss.bernoulli(0.001), 'tai': tai})


    # ENVIRONMENT
    environment = ty.EnvironmentalPool(pars={'teer_lam': 1.99,  # TEER: Typhoid environmental exposure rate
                                             'volume': 1,       # Set the volume to 1 if we want to reproduce EMOD-like results
                                             'transmission': ss.Pars({'rel_trans': 0.00001})})  # This parameter is equivalent to mEL parameter in Gauld etal 2018

    # Create the simulation with specific intervention parameters
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid,  demographics=environment, label=f"TAI:{tai}")
    return sim


# Create list to hold the multiple simulation instances
sims = sc.autolist()

# Values of TAI
for val in [1_000, 5_000, 10_0000, 20_000, 42_000]:
    sims.append(make_sim(tai=val))

# Run them
msim = ss.MultiSim(sims)
msim.run()
msim.plot(key="typhoid")
plt.show()
