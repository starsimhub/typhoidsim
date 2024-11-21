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
import numpy as np
import pandas as pd

import sciris as sc
import starsim as ss
import typhoidsim as ty

import calibration_pakistan_utils as utils

calib_debug = True  # If true, calibration will run in serial


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
    age_bin_edges = [0, 2, 15, ty.max_age]
    age_bin_labels = ['<2', '2-15', '15+']  # human readable labels

    to_record = dict(ti_infected=dict(path=("diseases", "typhoid")),
                     alive=dict(path=("people",)),
                     ti_positive=dict(path=("interventions", "base_test")),
                     ti_tested=dict(path=("interventions", "base_test")),
                     )
    # Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.
    anz = ty.histograms_by_age_sex(age_bins=age_bin_edges, age_bin_labels=age_bin_labels, to_record=to_record)

    # PUT EVERYTHING TOGETHER IN A SIMULATION
    sim = ss.Sim(pars=pars, people=ppl, diseases=typhoid, demographics=vital_dynamics + [environment],
                 interventions=[test, exposure_modulation, campaign_vax_2_5_yo],
                 analyzers=[anz])

    return sim


def build_sim(sim, calib_pars, **kwargs):
    """
    Tell the calibration how to update parameters for our specific model.
    The more module our model has, the more complex to navigate the path
    to find and update the required paraemeters.
    """
    # Access the modules whose parameters we need to modify dueing optimisation
    typh = sim.pars.diseases
    env = sim.pars.demographics[2]  # NOTE This is ugly but cannot do it a different way atm, there are three demographics modules: birtrhs, deaths and environment

    for k, pars in calib_pars.items():  # Loop over the calibration parameters
        v = pars['value']
        # Each item in calib_pars is a dictionary with keys like 'low', 'high',
        # 'guess', 'suggest_type', and importantly 'value'. The 'value' key is
        # the one we want to use as that's the one selected by the algorithm
        if k == 'tai':
            typh.pars.tai = v
        elif k == 'teer':
            env.pars.transmission.env2ppl_exposure_rate.lam = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized.')
    return sim


def make_calib_components():
    df = utils.load_empirical_data_pakistan()
    # Add a column with a similar representation of time
    df["yearvec"] = df["Date"].dt.year + (df["Date"].dt.dayofyear - 1) / ty.days_per_year
    df_2_to_15 = df.loc[(df["Ages"] == "Kids2to15"), :]

    # TODO: find a better way to do this, in case we change the age bins
    age_bin_labels = ['<2', '2-15', '15+']  # human readable labels
    # Make dirctionary to map lable to array index in analyzer result array
    age_bins_dict = {label: idx for idx, label in enumerate(age_bin_labels)}

    # Example with females between 2 <= age < 4
    f_infectious_2_to_15 = ty.CalibComponent220(
        name='f_positive_2_15',
        # Reference data
        expected=pd.DataFrame({
            'n': df_2_to_15["Sindh_positive"] * 5.0,  # !!! TODO: CHANGE!!! Made up scaling because "Sindh_tested" is all NaNs for this age group
            'x': df_2_to_15["Sindh_positive"].astype(float),  # Count/Number of individuals found to test positive
        }, index=pd.Index(df_2_to_15["yearvec"], name='t')),  # On these dates
        # Extract equivalent data from the simulation
        extract_fn=lambda sim: pd.DataFrame({
            'n': sim.analyzers.hist_by_age_sex.results.hist_f_ti_tested[:, age_bins_dict['2-15']],    # Number of individuals who were tested
            'x': sim.analyzers.hist_by_age_sex.results.hist_f_ti_positive[:, age_bins_dict['2-15']],  # Number of individuals whose test was positive
        }, index=pd.Index(sim.analyzers.hist_by_age_sex.yearvec, name='t')),  # Index is time

        conform='prevalent',
        nll_fn='beta',
        weight=1,  # Not required if only one component
    )
    components = [f_infectious_2_to_15]
    return components


def my_gof_fun(sim, expected_data=None):
    """ 
    Define your own goodness of fit function.
    This function takes in a sim, and returns a float (e.g. negative log likelihood) to be maximized.

    sim (starsim.Sim): a simulation object that will have been preconfigured, and run by the Calibration class
    expected_data (Any, but in this example a pandas dataframe): reference data used for comparison with simulation results.
    """

    # Extract and aggregate sim results however we need

    # Extract and aggregate reference data however we need

    # Calculate goodness-of-fit between reference and simulated data

    return np.random.rand(1)  # Just to make the calibration run


def run_starsim_calibration_step_1(do_plot=True):
    """"
    Calibration, step 1, we use data across the whole population
    """

    # Define the calibration parameters
    calib_pars = dict(
        # typhoid acute infectiousness
        tai=dict(low=1e-4, high=1e5, guess=42_808, log=True),
        # typhoid environmental exposure rate
        teer=dict(low=0.0, high=10.0, guess=1.99)  # The path always consists of three components/steps.
    )

    # Make the sim and data
    sim = make_sim()

    # Option 1: with custom-made goodnes-of-fit function
    # calib = ty.Calibration220(
    #     calib_pars=calib_pars,
    #     sim=sim,
    #     build_fn=build_sim,
    #     eval_fn=my_gof_fun,
    #     eval_kwargs=dict(expected_data=utils.load_empirical_data_pakistan()),  # loads all the data from TahirData_0928.csv without any preprocessing
    #     total_trials=16,
    #     n_workers=4,
    #     die=True,
    #     debug=calib_debug
    # )

    # Option 2: with calib components
    components = make_calib_components()
    calib = ty.Calibration220(
        calib_pars=calib_pars,
        sim=sim,
        build_fn=build_sim,
        build_kw=None,
        components=components,
        total_trials=16,
        n_workers=4,
        die=True,
        debug=calib_debug
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)

    calib.best_pars

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.check_fit(n_runs=5)

    # if do_plot:
    # NOTE: these plotting functions fail to work properly with the result arrays that hold the histograms by age and sex
    # This is a bug with the starsim framework (https://github.com/starsimhub/typhoidsim/issues/120)
    #     calib.plot_sims(key="typhoid")
    #     calib.plot_trend()
    plt.show()
    return calib


def run_debug_single_sim(do_plot=True):
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
    run_starsim_calibration_step_1(do_plot=True)
    #sim = run_debug_single_sim(do_plot=True)
    #msim = run_debug_multisim(do_plot=True)
