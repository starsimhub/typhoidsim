"""
Distributions
"""
import sciris as sc
import starsim as ss
import scipy.stats as sps


__all__ = ['truncnorm']


class truncnorm(ss.Dist):
    """
    Truncated normal distribution (specifically, scipy.stats.truncnorm)

    Args:
        a (float): the abscissa at which we wish to truncate the distribution to the left
        b (float): the abscissa at which we wish to truncate the distribution to the right
        loc (float): the center of the normal distribution (default 0.0)
        scale (float): the standard deviation of the distribution (default 1.0)


    """
    def __init__(self, a_trunc=0.0, b_trunc=1.0, loc=0.0, scale=1.0, **kwargs):
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        super().__init__(distname='truncnorm', dist=sps.truncnorm, a=a, b=b, loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs
