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
import pandas as pd

import sciris as sc
import starsim as ss
import typhoidsim as ty

import calibration_pakistan_utils as utils


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameter we are exploring.
def make_sim(tai=None, teer=None):
    """
    Specify the complete model, and create a simulation instance of typhoid
    with a simple vaccination intervention.

    Args:


    Returns:
    sim (starsim.Sim): a starsim simulation object, configured and ready to run.
    """

    # HIGH-LEVEL SIM PARAMETERS
    pars = dict(
        start    =1990.0,         # Starting year
        n_years  =10.0,           # Duration of the simulation in years
        dt       =1.0/365.0,      # Timestep of 1 day, expressed in years
        verbose  =1,              # Pint details of the run
    )

    age_data = utils.get_age_distribution_pakistan()  # Loads age distribution data from the json file used in EMOD simulations

    # POPULATION
    ppl = ss.People(10_000, age_data=age_data)

    # DEMOGRAPHICS
    # Load age-specific mortality rate, expressed in deaths per 1000 people
    ty_data = ty.get_data_home()  # Get's the directory of the data directory within typhoidsim
    death_rates = pd.read_csv(ty_data + '/pakistan_2020_deaths.csv')

    # Crude birth rate in Pakistan 2020 per 1000 people
    cbr = 27
    vital_dynamics = [
        ss.Births(birth_rate=cbr, units=1e-3),         # units=1e-3 mean rates are expressed per 1000 people
        ss.Deaths(death_rate=death_rates, units=1e-3)  # units=1e-3 mean rates are expressed per 1000 people
    ]

    # DISEASE CONFIGURATION

    # Define the susceptible introduction curve
    # This curve defines an age-based transition from completeley immune to susceptible.
    sus_saturation_age = 20.0       # in years
    sus_age_exposure_slope = 6.94   # ? not sure about the units

    # Function that defines the probability of becoming susceptible
    p_unexp2sus_parc_fun = utils.partial_unexp2susc(
        sus_saturation_age=sus_saturation_age,
        sus_age_exposure_slope=sus_age_exposure_slope)

    typhoids_pars = {'tai': tai,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.98,
                     'p_cpg': 0.108,
                     'p_acute': ss.bernoulli(p=0.16),
                     'init_prev': ss.bernoulli(0.1),
                     'p_unexp2sus': ss.bernoulli(p=p_unexp2sus_parc_fun)}

    typhoid = ty.Typhoid(pars=typhoids_pars)

    # ENVIRONMENT
    environment = ty.EnvironmentalPool(pars={'transmission': ss.Pars(env2ppl_exposure_rate=ss.poisson(lam=teer)),  #TEER: Typhoid environmental exposure rate
                                             'volume': 1})  # Set the volume to 1 if we want to reproduce EMOD results

    # INTERVENTIONS: Vaccination campaigns

    # Parameters of seasonal pattern
    seasonal_env_pars = {
        'period': 365.0,           # in days
        'peak_start_doy': 275.85,  # day of the year
        'ramp_up_dur': 175.26,     # duration in days
        'ramp_dw_dur': 100.0,      # duration in days
        'cutoff_dur': 20.0,        # duration in days
        'max_amp': 1.0}
    # Seasonal pattern
    trapezoidal_pattern = utils.partial_env_trapezoidal(seasonal_env_pars)

    # Intervention that uses the seasonal pattern
    exposure_modulation = ty.environmental_trapezoidal_modulation(efficacy=trapezoidal_pattern, start_year=2005.0)

    # Intervention with vaccination
    campaign_start_year = 1991.0
    campaign_vax_2_5_yo = ty.vaccination_wih_waning(
        start_year=campaign_start_year,
        end_year=1999.0,
        prob=0.1/365.0,  # coverage
        waning_pars={'efficacy': 0.95,
                     'decay_time_constant': 505.0 / ty.days_per_year,
                     'box_duration': 1.0},
        age_pars={'min_age': 2.0,
                  'max_age': 4.0}
        )

    # OBSERVATIONS AND REPORTING

    # Create an analyzer that will provide the results we need to compare to target empirical data
    age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']

    age_histogram_report = ty.age_histogram(age_bins=age_bin_edges, age_bin_labels=age_bin_labels, to_record="ti_infected")

    # PUT EVERYTHING TOGETHER IN A SIMULATION
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=vital_dynamics + [environment],
                 interventions=[exposure_modulation, campaign_vax_2_5_yo],
                 analyzers=age_histogram_report)


    # Export parameterss

    return sim


# def objective_step_1(trial):
#     """ The cost function to optimise in step 1 of calibration"""
#     p1 = trial.suggest_float('vax_eff', 0.0, 1.0)
#     p2 = trial.suggest_float('vax_cov', 0.0, 1.0)
#     p3 = trial.suggest_float('start_year', 2000.0, 2001.0)
#
#     # include variance of cost function?
#     cost = None
#     return cost


sim = make_sim(tai=42808.0, teer=6.94)
sim.run()

utils.save_outputs(sim)


