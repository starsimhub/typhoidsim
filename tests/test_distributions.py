"""
Test that the number of infections tracked in results and as internal state add up
"""
import sciris as sc
import typhoidsim as ty

import matplotlib.pyplot as plt


def test_truncnorm(n_samples=1000, a_trunc=0, b_trunc=2_000):
    """ Test the Dist class """
    sc.heading('Testing the basic Dist call')
    mu, sigma = 1_000, 1_000
    dist = ty.truncnorm(a_trunc=a_trunc, b_trunc=b_trunc, loc=mu, scale=sigma, name='test_runcnorm', strict=False)
    dist.init()
    rvs = dist(n_samples)
    print(rvs)
    assert rvs.min() >= a_trunc, f'Values should be above {a_trunc}'
    assert rvs.max() <= b_trunc,  f'Values should be below {b_trunc}'

    dist.plot_hist()
    return rvs


if __name__ == '__main__':
    test_truncnorm()
    plt.show()
