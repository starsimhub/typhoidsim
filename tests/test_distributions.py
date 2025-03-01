"""
Test that the number of infections tracked in results and as internal state add up
"""
import sciris as sc
import typhoidsim as ty

import matplotlib.pyplot as plt


def test_truncnorm(n_samples=1000, a_trunc=0, b_trunc=2_000):
    sc.heading('Testing truncnorm')
    mu, sigma = 1_000, 1_000
    dist = ty.truncnorm(a_trunc=a_trunc, b_trunc=b_trunc, loc=mu, scale=sigma, name='test_truncnorm', strict=False)
    dist.init()
    rvs = dist(n_samples)
    print(rvs)
    assert rvs.min() >= a_trunc, f'Values should be above {a_trunc}'
    assert rvs.max() <= b_trunc,  f'Values should be below {b_trunc}'

    dist.plot_hist()
    return rvs


def test_gausshyper(n_samples=1000):
    sc.heading('Testing gausshyper')
    a = 13.8
    b = 3.12
    c = 2.51
    z = 5.18
    loc = 1.0
    dist = ty.gausshyper(a=a, b=b, c=c, z=z, loc=loc, strict=False)
    dist.init()
    rvs = dist(n_samples)
    dist.plot_hist()


def test_gompertz(n_samples=1000):
    sc.heading('Testing gompertz')
    dist = ty.gompertz(c=1.0, strict=False)
    dist.init()
    rvs = dist(n_samples)
    dist.plot_hist()


if __name__ == '__main__':
    test_truncnorm()
    test_gausshyper()
    test_gompertz()
    plt.show()
