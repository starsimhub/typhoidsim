"""
Define observation methods that do not interfere with the course of a disease
"""

import starsim as ss
import sciris as sc
import numpy as np

__all__ = ['Monitor']
class Monitor(ss.Module):
    """ Generic monitor implementation """

    def initialize(self, sim):
        if not self.initialized:
            super().initialize(sim)
        else:
            return

    def record(self, people, inds):
        """ Observe and record a snapshot in time"""
        raise NotImplementedError


class TyphoidTest(ss.Dx):
    # implement specificity and sensitivity
    pass


class EnviromentalMonitor(ss.Monitor):
    pass
