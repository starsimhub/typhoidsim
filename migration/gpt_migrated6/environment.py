
"""
Define environments
"""

import numpy as np

import starsim as ss
import sciris as sc

import typhoidsim.patterns as typ
import typhoidsim.defaults as tyd


__all__ = ['EnvironmentalPool']


class EnvironmentalPool(ss.Demographics):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_cfu=0,            # Initial level of CFUs in the environment.
            decay_rate=ss.perday(0.3),  # Decay rate of environmental in fraction of CFUs that decay in 1/day
            volume=1,              # Assumed volume of the environmental pool.
            acceptable_level=600,  # CFU/volume, usually expressed in CFU/ml (not used at the moment)
            transmission=ss.Pars(
                rel_trans=ss.rate(1e-3),  # Long-cycle exposure (to the environment) multiplier
                shedding_rate=ss.perday(1.0),  # Rate at which infectious people shed CFUs to the environment.
                env2ppl_exposure_rate=ss.poisson(lam=2.0),  # Poisson rate determining the daily amount of exposures.
            ),
        )
        self.update_pars(pars, **kwargs)

        # Track a variable that does not track the state of individual agents
        self.sv = typ.StateVariables(self.name)

        self.buffer_isteps = 2  # length of the concentration buffer in integer number of timesteps
        return

    def init_results(self):
        self.define_results(
            ss.Result('cfu_conc', dtype=float, scale=False, label='Current CFU concentration'),
            ss.Result('cfu_num', dtype=int, scale=False, label='Current number of CFUs'),
        )
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        self.init_svs()
        self.init_env_pool(sim)  # Initialise the environmental pool of contagion at t-1
        return

    def init_svs(self):
        """
        Initialise StateVariable objects
        """
        self.sv += [typ.StateVariable(self.name, "cfu_conc", self.sim.t.npts, dtype=float),]
        self.sv += [typ.StateVariable(self.name, "cfu_conc_buffer", self.buffer_isteps, dtype=float),]
        return

    def init_env_pool(self, sim):
        self.sv.cfu_conc_buffer[:] = self.pars.init_cfu   # Fill the history
        self.sv.cfu_conc[sim.ti] = self.pars.init_cfu     # Fill initial conditions
        return

    def step(self):
        sim = self.sim
        ti = self.sim.ti
        p = self.pars
        # For external changes that may promote bacterial growth
        change_rate = p.decay_rate
        effective_rate = change_rate.to_parent()  # transform to yearly rate

        prev_cfu = self.sv.cfu_conc_buffer[ti % self.buffer_isteps]
        self.sv.cfu_conc[ti] = prev_cfu * np.exp(-effective_rate * sim.t.dt)  # decay
        return

    def update_results(self):
        self.results['cfu_conc'][self.sim.ti] = self.sv.cfu_conc[self.sim.ti - 1]
        self.results['cfu_num'][self.sim.ti] = self.sv.cfu_conc[self.sim.ti - 1] * self.pars.volume
        return
