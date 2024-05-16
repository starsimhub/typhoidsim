"""
Define environments
"""

import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

__all__ = ['Environment']


class Environment(ss.Module):
    """
    A base class for environment modules.
    """
    def initialize(self, sim):
        super().initialize(sim)
        self.init_results()
        return

    def init_results(self):
        pass

    def update(self):
        pass

    def update_results(self):
        pass


class Climate(Environment):
    pass


class LongCycleCCVT(Environment):
    """
    Contaminated vehicles of transmission
    """

    def __init__(self, pars=None, *args, **kwargs):
        """ Initialize with parameters """
        super().__init__()
        self.default_pars(

            # Environmental parameters
            beta=1.0,  #  transmission from environment,
            half_sat_rate=1_000_000,
            # Infectious dose in water sufficient to produce infection in X% of exposed agent
            shedding_rate=10,
            # Rate at which infectious people shed bacteria to the environment (per day),
            decay_rate=0.033,
            # Rate at which bacteria in the environment dies (per day),
            p_env_transmit=ss.bernoulli(p=0),
            # Probability of environmental transmission - filled out later
        )
        self.update_pars(pars, **kwargs)

        # Boolean states
        self.add_states(
            # Susceptible & infected are added automatically, here we add the rest
            ss.BoolArr('prevalence'),
            ss.BoolArr('concentration'),
            ss.BoolArr('contaminated'),

            # Timepoint states
            ss.FloatArr('ti_contaminated'),
        )

    def update(self):
        pass