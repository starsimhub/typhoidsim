"""
This script contains step 1 of the calibration workflow to reproduce results from
Kraay et al. 2024 https://www.medrxiv.org/content/10.1101/2024.08.30.24312839v1

Country: Pakistan
Province: Sindh

Transmission routes:
- Environmental only

Period:
- Prior to vaccine introduction
- January 2018 - December 2019 (2 years)

Free parameters:
- typhoid acute infectiousness (TAI) expressed in CFUs
- lambda environment: average exposure rate to the environment, expressed in (in num exposures * volume / day)

Target:
- age distribution of blood culture confirmed typhoid cases
"""

import matplotlib.pyplot as plt
import optuna

import starsim as ss
import typhoidsim as ty


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(tai=None, lam=None):
    """
    Create a simulation instance of typhoid with a simple vaccination intervention.

    Args:
        vax_eff (float): The efficacy of the typhoid vaccine. A value between 0 and 1 where 1 represents 100% efficacy.
        vax_cov (float): The coverage of the vaccination. A value between 0 and 1 where 1 represents 100% of population coverage.
        start (float): The year when the vaccination intervention starts.

    Returns:
    sim (starsim.Sim): a starsim simulation configured with the same high level parameters for
    the populatioin and typhoid disease, but with slightly different intervention parameters.

    Example
    # Make a single simulation with a vaccine campaign where the vaccine has 70% efficacy and we cover 80% of the population
    sim = make_sim(vax_eff=0.7, vax_cov=0.8)
    sim.run()
    """

    # Define high-level simulation parameters
    start_year = 2018
    pars = dict(
        start    =1990,          # Starting year
        n_years  =50.0,          # Duration of the simulation in years
        dt       =1.0/365.0,     # Timestep of 1 day, expressed in years
        verbose  =0,             # Do not print details of the run
    )

    # The population
    ppl = ss.People(10_000)

    # Demographics
    # TODO: add life expectancy 66.51 year
    # TODO: population growth rate 2.74%/year

    # The disease
    typhoids_pars = {'tai': tai, 'tpri': 0.5, 'tsri': 1.0, 'tcri': 0.241, 'tppi': 0.98,
                     'p_cpg': 0.108, 'init_prev': ss.bernoulli(0.1)}
    typhoid = ty.Typhoid(pars=typhoids_pars)

    environment = ty.EnvironmentalPool(pars={'transmission': ss.Pars(env2ppl_exposure_rate=ss.poisson(lam=lam))})

    # Create seasonal pattern ramp

    # Create intervention for reporting to get positive cases 'cases'

    # Create an analyzer that will provide the results we need to compare to target empirical data
    age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']

    # Create the simulation
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=environment)

    # Run multisim with 100 sims?

    return sim


def objective_step_1(trial):
    """ The cost function to optimise in step 1 of calibration"""
    p1 = trial.suggest_float('vax_eff', 0.0, 1.0)
    p2 = trial.suggest_float('vax_cov', 0.0, 1.0)
    p3 = trial.suggest_float('start_year', 2000.0, 2001.0)

    # include variance of cost function?
    cost = None
    return cost


def objective_step_2(trial):
    """ The cost function to optimise in step 2 of calibration"""
    pass


def objective_step_3(trial):
    """ The cost function to optimise in step 3 of calibration"""
    pass