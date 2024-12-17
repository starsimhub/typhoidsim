
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
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_cfu=0,             # Initial level of CFUs in the environment.
            decay_rate=0.3,         # Decay rate of environmental CFUs (fraction of CFUs that decay per day)
            volume=1,               # Volume of the environmental pool. Units: to be defined.
            acceptable_level=600,   # CFU/volume, usually expressed in CFU/ml (not used at the moment)
            transmission=ss.Pars(
                rel_trans=1e-3,                             # Long-cycle exposure multiplier
                shedding_rate=ss.rate(1.0, unit='day'),     # Rate at which infectious people shed CFUs per day
                env2ppl_exposure_rate=ss.poisson(lam=2.0),  # Poisson rate for daily environmental exposures
            ),
        )
        self.update_pars(pars, **kwargs)

        # Track a variable that does not track the state of individual agents, ~and it's not a Result
        self.sv = typ.StateVariables(self.name)

        self.buffer_isteps = 2  # Length of the concentration buffer in integer number of timesteps
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
        self.init_env_pool()  # Initialise the environmental pool of contagion at t-1
        return

    def init_svs(self):
        """ Initialise StateVariable objects """
        npts = self.sim.npts
        self.sv += [typ.StateVariable(self.name, "cfu_conc", npts, dtype=float)]
        self.sv += [typ.StateVariable(self.name, "cfu_conc_buffer", self.buffer_isteps, dtype=float)]
        return

    def init_env_pool(self):
        self.sv.cfu_conc_buffer[:] = self.pars.init_cfu  # Fill the history
        self.sv.cfu_conc[self.sim.ti] = self.pars.init_cfu  # Fill initial conditions
        return

    def step(self):
        sim = self.sim
        ti = sim.ti
        p = self.pars
        change_rate = p.decay_rate
        effective_rate = change_rate / tyd.day2year  # Transform to yearly rate

        prev_cfu = self.sv.cfu_conc_buffer[ti % self.buffer_isteps]
        self.sv.cfu_conc[ti] = prev_cfu * np.exp(-effective_rate * sim.t.dt)  # + shedded into environment + decay
        return

    def update_results(self):
        ti = self.sim.ti
        self.results['cfu_conc'][ti] = self.sv.cfu_conc[ti-1]
        self.results['cfu_num'][ti] = self.sv.cfu_conc[ti-1] * self.pars.volume
        return
