#!/usr/bin/env python3
"""
Larger test of sim performance.
"""

import sciris as sc
import starsim as ss
import typhoidsim as ty


def make_run_sim(n_agents=100_000):
    # Define high-level simulation parameters
    pars = dict(
        n_agents = n_agents,
        start    = 2000,  # Starting year
        n_years  = 1.0,   # Duration of the simulation in years
        dt       = 1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  = 0,             # Do not print details of the run
    )

    # The environment the population is affected by, and can carry the pathogen
    environment = ty.EnvironmentalPool()

    # Person to person transmission
    random_p2p = ss.RandomNet({'n_contacts': 5})

    # The disease
    typhoid = ty.Typhoid()

    sim = ss.Sim(
        pars=pars,
        demographics=environment,
        networks=random_p2p,
        diseases=typhoid,
        )
    sim.run()
    return sim


if __name__ == '__main__':
    sc.tic()
    sim = make_run_sim()
    sc.toc()
