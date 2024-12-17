
"""
Define environments
"""

import numpy as np
import starsim as ss
import sciris as sc
import typhoidsim.patterns as typ
import typhoidsim.defaults as tyd

__all__ = ['EnvironmentalPool']


class EnvironmentalPool(ss.Module):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_cfu=0,            # Initial level of CFUs in the environment.
            decay_rate=0.3,        # Decay rate of environmental in fraction of CFUs that decay in 1/day (init_cfu*exp(-decay_rate*t))
            volume=1,              # Assumed volume of the environmental pool. Units: to be defined: See https://www.pnas.org/doi/full/10.1073/pnas.1719579115
            acceptable_level=600,  # CFU/volume, usually expressed in CFU/ml (not used at the moment) #TODO: to be used with an environmental monitor intervention
            transmission=ss.Pars(
                rel_trans=1e-3,                           # Long-cycle exposure (to the environment) multiplier, targeted by interventions, mEL in Gauld et al 2018
                shedding_rate=1.0,                        # Rate at which infectious people shed colony-forming units to the environment (per day)
                env2ppl_exposure_rate=ss.poisson(lam=2.0),  # Poisson rate determining the daily amount of exposures for environment route (num exposures * volume)/day -- lam is equivalent to typhoid_environmental_exposure_rate
            ),
        )
        self.update_pars(pars, **kwargs)

        # Track a variable that does not track the state of individual agents, ~and it's not a Result
        self.sv = typ.StateVariables(self.name)

        self.buffer_isteps = 2  # length of the concentration buffer in integer number of timesteps
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
        npts = self.sim.t.npts

        self.sv += [typ.StateVariable(self.name, "cfu_conc", npts, dtype=float),]
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
        effective_rate = (change_rate / tyd.day2year)  # transform to yearly rate

        prev_cfu = self.sv.cfu_conc_buffer[ti % self.buffer_isteps]
        self.sv.cfu_conc[ti] = prev_cfu * np.exp(-effective_rate * sim.t.dt)  # + shedded into environment + decay
        return

    def init_results(self):
        npts = self.sim.t.npts
        self.define_results(
            ss.Result('cfu_conc', dtype=float, scale=False, label='Current CFU concentration', shape=npts),
            ss.Result('cfu_num', dtype=int, scale=False, label='Current number of CFUs', shape=npts),
        )
        return

    def update_results(self):
        self.results.cfu_conc[self.sim.ti] = self.sv.cfu_conc[self.sim.ti - 1]
        self.results.cfu_num[self.sim.ti] = self.sv.cfu_conc[self.sim.ti - 1] * self.pars.volume
        return
