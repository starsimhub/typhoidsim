"""
Distributions, useful for vaccine waning parameterization
"""
import sciris as sc
import starsim as ss
import scipy.stats as sps


__all__ = ['truncnorm', 'gausshyper', 'gompertz', 'beta']


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


class gausshyper(ss.Dist):
    """
    A Gauss hypergeometric continuous random variable,
    also known was 2F1 function (scipy.special.hyp2f1)

    Args:
        a (float): a > 0, shape parameter
        b (float): b > 0, shape parameter
        c (float): c is a real number, shape parameter
        z (float): z > -1, shaper parameter
        loc (float): location parameter
        scale (float): scale parameter

    See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gausshyper.html
    """
    def __init__(self, a=13.8, b=3.12, c=2.51, z=5.18, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='gausshyper', dist=sps.gausshyper,
                         a=a, b=b, c=c, z=z, loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class gompertz(ss.Dist):
    """
    A Gompertz (or truncated Gumbel) continuous random variable.
    Args:
        c (float):
        loc (float):
        scale (float):
    """
    def __init__(self, c=1.0, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='gompertz', dist=sps.gompertz, c=c,
                         loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class beta(ss.Dist):
    """
    A beta continuous random variable.
    Args:
        c (float):
        loc (float):
        scale (float):
    """

    def __init__(self, a=1.0, b=1.0, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='beta', dist=sps.beta, a=a, b=b,
                         loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs
