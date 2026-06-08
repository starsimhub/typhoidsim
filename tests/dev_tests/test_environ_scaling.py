"""
Test how simulations with environmental transmission scales with respect to n_agents
"""

import matplotlib.pyplot as plt

import sciris as sc
import starsim as ss
import typhoidsim as ty


def make_sim(n_agents=1_000):

    # Define the parameters
    pars = sc.objdict(
        start=2000,       # Starting year
        dur=1.0,      # Number of days to simulate
        n_agents=n_agents,
        dt=1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose=0,        # Print details of the run
        rand_seed=2,      # Set a non-default seed
    )

    # Who
    ppl = ss.People(pars["n_agents"])

    # DISEASE CONFIGURATION
    typhoids_pars = {'tai': 42_000,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.05,
                     'p_cpg': 0.108,
                     'p_acute': ss.bernoulli(p=0.24),
                     'init_prev': ss.bernoulli(p=0.05),
                     "unexp2sus_saturation_age": 20.0,
                     "unexp2sus_slope": 7.0
                     }

    # What
    typhoid = ty.Typhoid(pars=typhoids_pars)

    # How
    environment = ty.EnvironmentalPool(
        pars={'volume': 1,
              # Set the volume to 1 if we want to reproduce EMOD-like results
              'transmission': ss.Pars({'rel_trans': 0.025,
                                       # This parameter is equivalent to mEL parameter in Gauld etal 2018
                                       'shedding_rate': 0.3})})

    #
    sim = ss.Sim(
        people=ppl,
        pars=pars,
        diseases=typhoid,
        demographics=environment,
        label=f"n_agents={pars['n_agents']}"
        )
    return sim


sims = sc.autolist()

for n_agents in [1_000, 10_000, 100_000, 100_000]:
    sims.append(make_sim(n_agents=n_agents))

msim = ss.MultiSim(sims)
msim.run()
msim.plot(key="typhoid_")
plt.show()
