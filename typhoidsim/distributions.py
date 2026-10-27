"""
Distributions, useful for vaccine waning parameterization
"""
import sciris as sc
import starsim as ss
import scipy.stats as sps


__all__ = ['gausshyper', 'gompertz'] # For truncated normal and beta distributions, see ss.truncnorm and ss.beta_dist


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
