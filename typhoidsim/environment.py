"""
Define environments
"""

import numpy as np

import starsim as ss

import typhoidsim.patterns as typ


__all__ = ['EnvironmentalPool']


class Environment(ss.Demographics):
    """
    A base class for environment modules.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.init_results()
        return

    def init_results(self):
        pass

    def update(self):
        pass

    def update_results(self):
        pass


class EnvironmentalPool(Environment):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.default_pars(
            init_prev=ss.bernoulli(0.0),
            init_cfu=0,      # Initial level of CFUs in the environment.
            decay_rate=0.3,  # Decay rate of environmental in fraction of CFUs that decay in 1/day (init_cfu*exp(-decay_rate*t))
            acceptable_level=600,  # CFU/ml
            bs_temp=6,       # Baseline temperature at which bacteria would stop growing
            av_temp=typ.Pattern("av_temp", pars={'av_temp': 25.0}, pattern_name="Environmental Temperature"),
            b=0.0297,
            transmission=ss.Pars(
                ppl2env_shedding_rate=0.1,  # Rate at which infectious people shed colony-forming units to the environment (per day), scaled by individual rel_trans
                env2ppl_exposure_rate=ss.poisson(lam=0.5),  # Poisson rate determining the daily number of exposures for environment route (size ppl)
            ),

        )
        self.update_pars(pars, **kwargs)

        # Track a variable that does not track the state of individual agents, and it's not a Result
        self.sv = typ.StateVariables(self.name)

        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        self.init_svs()
        return

    def init_svs(self):
        """
        Initialise StateVariable objects
        """
        npts = self.sim.npts
        self.sv += [typ.StateVariable(self.name, "cfu_level", npts, dtype=float),]
        return

    def get_growth_rate(self):
        sim = self.sim
        ti = self.sim.ti
        p = self.pars
        sqr_growth_rate = p.b * (p.av_temp.evaluate(ti) - p.bs_temp)
        return sqr_growth_rate**2

    def update(self):
        sim = self.sim
        ti = self.sim.ti
        p = self.pars

        # For external changes that may promote bacterial growth
        growth_rate = self.get_growth_rate()
        change_rate = (p.decay_rate-growth_rate)
        self.sv.cfu_level[ti] = self.sv.cfu_level[ti-1] * np.exp(-change_rate*self.sim.dt)  # + shedded into environment + decay
        return
