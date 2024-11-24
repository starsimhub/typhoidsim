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
        self.default_pars(
            init_cfu=0,            # Initial level of CFUs in the environment.
            decay_rate=0.3,        # Decay rate of environmental in fraction of CFUs that decay in 1/day (init_cfu*exp(-decay_rate*t))
            volume=1e4,            # Assumed volume of the environmental pool. Units: to be defined: See https://www.pnas.org/doi/full/10.1073/pnas.1719579115
            acceptable_level=600,  # CFU/volume, usually expressed in CFU/ml (not used at the moment) #TODO: to be used with an environmental monitor intervention
            transmission=ss.Pars(
                rel_trans=1e-3,     # Long-cycle exposure (to the environment) multiplier, targeted by interventions, mEL in Gauld et al 2018
                shedding_rate=1.0,                           # Rate at which infectious people shed colony-forming units to the environment (per day)
                env2ppl_exposure_rate=ss.poisson(lam=2.0),   # Poisson rate determining the daily amount of exposures for environment route (num exposures * volume)/day -- lam is equivalent to typhoid_environmental_exposure_rate

            ),
            teer_lam=2.0,     # HACKY parameter, otherwise CAlibration class does not ru where the path to the parameter is longer than 3
            # Temperature-parameters, not used at the moment
            bs_temp=6.0,   # Baseline temperature at which bacteria would stop growing, in degree Celsius
            av_temp=14.0,  # typ.Pattern("av_temp", pars={'av_temp': 14.0}, pattern_name="Environmental Temperature"),
            b=0.0297,      # fraction of change (increase or decrease) in [growth rate/degree Celsius]
        )
        self.update_pars(pars, **kwargs)

        # Track a variable that does not track the state of individual agents, ~and it's not a Result
        self.sv = typ.StateVariables(self.name)

        return

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'temperature', npts, dtype=int, scale=True, label='Environmental temperature'),
            ss.Result(self.name, 'cfu_conc', npts, dtype=float, scale=True, label='Current CFU concentration'),
            ss.Result(self.name, 'cfu_num', npts, dtype=int, scale=True, label='Current number of CFUs'),
            ss.Result(self.name, 'rel_trans', npts, dtype=float, scale=True, label='Relative exposure to long-cycle CCTV'),
        ]
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        self.init_svs()
        self.init_env_pool(sim)  # Initialise the environmental pool of contagion at t-1
        if sc.isnumber(self.pars.av_temp):
            self.pars.av_temp = typ.Pattern("av_temp", pars={'av_temp': self.pars.av_temp})
        ##
        self.pars.transmission.env2ppl_exposure_rate.lam = self.pars.teer_lam
        return

    def init_svs(self):
        """
        Initialise StateVariable objects
        """
        npts = self.sim.npts
        self.sv += [typ.StateVariable(self.name, "cfu_conc",    npts, dtype=float),]
        self.sv += [typ.StateVariable(self.name, "temperature", npts, dtype=float),]
        return

    def init_env_pool(self, sim):
        ti = 0  # initial time step
        self.sv.cfu_conc[sim.ti-1] = self.pars.init_cfu
        return

    def get_growth_rate(self):
        sim = self.sim
        ti = sim.ti
        p = self.pars
        self.sv.temperature[ti] = p.av_temp.evaluate(ti)
        sqr_growth_rate = p.b * (self.sv.temperature[ti] - p.bs_temp)  # fraction of change in CFUs / per day
        return sqr_growth_rate**2

    def update(self):
        sim = self.sim
        ti = self.sim.ti
        p = self.pars

        # For external changes that may promote bacterial growth
        growth_rate = self.get_growth_rate()
        change_rate = (p.decay_rate)
        effective_rate = (change_rate / tyd.day2year)  # transform to yearly rate
        self.sv.cfu_conc[ti-1] = self.sv.cfu_conc[ti-2] * np.exp(-effective_rate*self.sim.dt)  # + shedded into environment + decay
        return

    def update_results(self):
        self.results['cfu_conc'][self.sim.ti] = self.sv.cfu_conc[self.sim.ti-1]
        self.results['cfu_num'][self.sim.ti]  = self.sv.cfu_conc[self.sim.ti-1] * self.pars.volume
        self.results['temperature'][self.sim.ti] = self.sv.temperature[self.sim.ti-1]
        self.results['rel_trans'][self.sim.ti] = self.pars.transmission.rel_trans
        return
