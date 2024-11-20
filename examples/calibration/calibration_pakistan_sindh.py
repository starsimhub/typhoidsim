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
- lambda environment: average exposure rate to the environment (TEER), expressed in (in num exposures * volume / day)

Target:
- age distribution of blood culture confirmed typhoid cases
"""
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import typhoidsim as ty

import calibration_pakistan_utils as utils


# We will make a function that will return an instance of Sim(). All instances
# will be identical except for the the value of the parameters we are calibrating.
def make_sim():
    """
    Specify the complete model, and create a simulation instance of typhoid
    with a simple vaccination intervention.

    Args:
        None

    Returns:
    sim (starsim.Sim): a starsim simulation object, configured and ready to run.
    """

    # HIGH-LEVEL SIM PARAMETERS
    pars = dict(
        start    =2017.0,         # Start year
        n_years  =6.0,            # Duration of the simulation in years
        dt       =1.0/365.0,      # Timestep of 1 day, expressed in years
        verbose  =0,              # Pint details of the run
    )

    # Loads age distribution data from the json file used in EMOD simulations
    age_data = utils.get_age_distribution_pakistan()   # NOTE: Any adjustments to the age distribution (ie, to scale it to the demographics of the specific province) can be done here

    # POPULATION
    ppl = ss.People(10_000, age_data=age_data)

    # DEMOGRAPHICS
    # Load age-specific mortality rate, expressed in age-group proportions, in units of 1/year
    death_rates_df = utils.get_mortality_rates_pakistan()

    # Crude birth rate in Pakistan 2020 per 1000 people
    cbr = 27
    vital_dynamics = [
        ss.Births(birth_rate=cbr, units=1e-3),         # units=1e-3 mean rates are expressed per 1000 people
        ss.Deaths(death_rate=death_rates_df, units=1)  # units=1 mean rates are expressed as proportions/percentages in 1/year
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

    typhoids_pars = {'tai': 42_000,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.98,
                     'p_cpg': 0.108,
                     'p_acute': ss.bernoulli(p=0.16),
                     'init_prev': ss.bernoulli(p=0.05),                   # Initial prevalence: This is how we seed infections at the start of a simulation. In this case approx 5% of the total population of agents, will be infected at t=0
                     'p_unexp2sus': ss.bernoulli(p=p_unexp2sus_parc_fun)}

    typhoid = ty.Typhoid(pars=typhoids_pars)

    # ENVIRONMENT
    environment = ty.EnvironmentalPool(pars={'teer_lam': 1.99,  # TEER: Typhoid environmental exposure rate
                                             'volume': 1,       # Set the volume to 1 if we want to reproduce EMOD-like results
                                             'transmission': ss.Pars({'rel_trans': 0.00001})})  # This parameter is equivalent to mEL parameter in Gauld etal 2018

    # INTERVENTIONS: Vaccination campaigns

    # Parameters of seasonal pattern
    seasonal_env_pars = {
        'period': 365.0,           # in days
        'peak_start_doy': 0.0,  # day of the year
        'ramp_up_dur': 175.26,     # duration in days
        'ramp_dw_dur': 100.0,      # duration in days
        'cutoff_dur': 20.0,        # duration in days
        'max_amp': 1.0}
    # Seasonal pattern
    trapezoidal_pattern = utils.partial_env_trapezoidal(seasonal_env_pars)

    # Intervention that uses the seasonal pattern
    exposure_modulation = ty.environmental_trapezoidal_modulation(efficacy=trapezoidal_pattern, start_year=2005.0)

    # Intervention with vaccination
    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        start_year=2020.0,
        end_year=2022.0,
        prob=0.1/365.0,  # coverage
        waning_pars={'efficacy': 0.95,
                     'decay_time_constant': 505.0 / ty.days_per_year,
                     'box_duration': 1.0},
        age_pars={'min_age': 2.0,
                  'max_age': 4.0}
        )

    # Intervention base test. This mimics the imperfect/incomplete nbature of empirical data through the lens of testing a fraction of the population.
    test = ty.base_test(prob_t=0.3, prob_tp=1.0, eligibility=ty.eligibility_by_age)

    # OBSERVATIONS AND REPORTING

    # Create an analyzer that will provide the results we need to compare to target empirical data
    age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']
    # Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.
    anz_1 = ty.histograms_by_age_sex(age_bins=age_bin_edges, age_bin_labels=age_bin_labels, to_record="ti_infected", name="report_1")
    # Track cases for all the population, grouped by sex -- just for convinience, one could process the results from the analyzer above.
    anz_2 = ty.histograms_by_age_sex(age_bins=[0, ty.max_age], age_bin_labels=['all'], to_record="ti_infected", name="report_2")

    # PUT EVERYTHING TOGETHER IN A SIMULATION
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=vital_dynamics + [environment],
                 interventions=[test, exposure_modulation, campaign_vax_2_5_yo],
                 analyzers=[anz_1, anz_2])

    return sim


def run_starsim_calibration_step_1(do_plot=True):
    """"
    Calibration, step 1, we use data across the whole population
    """

    # Define the calibration parameters
    calib_pars = dict(
        # typhoid acute infectiousness
        tai=dict(low=0.0, high=1e6, guess=42_808, path=('diseases', 'typhoid', 'tai')),
        # typhoid environmental exposure rate
        teer=dict(low=0.0, high=10.0, guess=1.99, path=('demographics', 'environmentalpool', 'teer_lam'))  # The path always consists of three components/steps.
    )

    # Make the sim and data
    sim = make_sim()
    data = utils.get_data_for_calibration_prevax(province="Sindh")  # only gets data in the years 2018-2019
    # Define weights for the goodness of fit
    weights = {
        'typhoid.prevalence':     1.0,
        'typhoid.new_infections': 1.0,
    }

    # Make the calibration
    calib = ty.Calibration(
        calib_pars=calib_pars,
        sim=sim,
        data=data,
        weights=weights,
        total_trials=16,
        n_workers=2,
        die=True,
        name="typhoidsim_calibration_sindh"
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.confirm_fit()
    print(f'Fit with original pars: {calib.before_fit:n}')
    print(f'Fit with best-fit pars: {calib.after_fit:n}')
    if calib.after_fit <= calib.before_fit:
        print('✓ Calibration improved fit')
    else:
        print('✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')

    if do_plot:
        calib.plot_sims(key="typhoid")
        calib.plot_trend()
    plt.show()
    return calib


def run_debug(do_plot=True):
    """ Run one simulation"""
    sim = make_sim()
    sim.run()
    if do_plot:
        sim.plot()
        plt.show()
    return sim


def run_debug_multisim(do_plot=True):
    sim1 = make_sim()
    sim2 = make_sim()
    sims = sc.autolist()
    sims.append(sim1)
    sims.append(sim2)

    # Let's change a parameter in sim2
    sim2.initialize()
    sim2.diseases.typhoid.pars.tai = 1_000
    msim = ss.MultiSim(sims)
    msim.run()
    if do_plot:
        msim.plot()
        plt.show()
    return msim


if __name__ == '__main__':
    #run_starsim_calibration_step_1(do_plot=True)
    #sim = run_debug(do_plot=True)
    msim = run_debug_multisim(do_plot=True)
