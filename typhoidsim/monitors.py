"""
Monitors: analyzers that record what happens to each agent, for debugging.

Counts by age and sex, and over coarser periods than the sim's step, are the sim's own
results: stratify the sim (`ss.Sim(..., stratify=dict(age=[0, 5, 15]))`), and read
`sim.results.typhoid.acute.by('age')` or `res.resample('year')`.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['track_individuals_monitor']


class track_individuals_monitor(ss.Analyzer):
    """
    Record the values of some arrays for every agent at every time point (NaN for those not
    eligible): for debugging, with a population of fixed size

    Args:
        to_record (dict): the arrays to record, each with the path to the module holding it, e.g. `dict(rel_sus=dict(path=('diseases', 'typhoid')))` (the default)
        eligibility (func): a function of the sim returning a boolean array or the UIDs of the agents to record (default: those alive)
    """
    def __init__(self, to_record=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.to_record = sc.ifelse(to_record, dict(rel_sus=dict(path=('diseases', 'typhoid'))))
        self.eligibility = eligibility
        return

    def init_results(self):
        super().init_results()
        shape = (self.t.npts, len(self.sim.people))
        self.define_results(*[ss.Result(key, shape=shape, dtype=float, scale=False, label=key) for key in self.to_record])
        return

    def step(self):
        sim = self.sim
        eligible = sim.people.alive if self.eligibility is None else self.eligibility(sim)
        uids = eligible.uids if isinstance(eligible, ss.BoolArr) else ss.uids(eligible)
        for key, spec in self.to_record.items():
            obj = sim
            for attr in spec['path']:
                obj = getattr(obj, attr)
            vals = obj[key]
            row = np.full(self.results[key].shape[1], np.nan)
            uids = uids[uids < len(row)]
            row[uids] = vals[uids]
            self.results[key][self.ti, :] = row
        return
