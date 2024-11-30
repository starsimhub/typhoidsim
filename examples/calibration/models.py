from functools import partial
import matplotlib.pyplot as plt


import sciris as sc
import starsim as ss
import typhoidsim as ty

import data_utils as utils


def get_common_simulation_pars():
    # HIGH-LEVEL SIM PARAMETERS
    pars = dict(
        start    =1990.0,         # Start year
        n_years  =15.0,            # Duration of the simulation in years
        dt       =1.0/365.0,      # Timestep of 1 day, expressed in years
        n_agents =100_000,         # Number of agents in the population
        verbose  =0,              # Print details of the run
    )
    return pars


def baseline_model():
    """
    Specify the complete model, and create a simulation instance of typhoid
    with a simple vaccination intervention.

    Args:
        None

    Returns:
    sim (starsim.Sim): a starsim simulation object, configured with the model we want to run.
    """

    # The comon parameters to all models/simulations
    pars = get_common_simulation_pars()

    # Loads age distribution data from the json file used in EMOD simulations
    # NOTE: Any adjustments to the age distribution needs to be done here ie,
    #  to scale age_data to adjust to the demographics of a specific province
    age_data = utils.get_age_distribution_pakistan()

    # POPULATION
    ppl = ss.People(pars["n_agents"], age_data=age_data)

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
    typhoids_pars = {'tai': 42_000,
                     'tpri': 0.5,
                     'tsri': 1.0,
                     'tcri': 0.241,
                     'tppi': 0.05,
                     'p_cpg': 0.108,
                     'p_acute': ss.bernoulli(p=0.24),
                     'init_prev': ss.bernoulli(p=0.05),
                     "unexp2sus_saturation_age": 20.0,
                     "unexp2sus_slope": 7.0
                     }

    typhoid = ty.Typhoid(pars=typhoids_pars)

    # ENVIRONMENT
    environment = ty.EnvironmentalPool(pars={'teer_lam': 1.99,  # TEER: Typhoid environmental exposure rate
                                             'volume': 1,       # Set the volume to 1 if we want to reproduce EMOD-like results
                                             'transmission': ss.Pars({'rel_trans': 0.025/pars["n_agents"],  # This parameter is equivalent to mEL parameter in Gauld etal 2018
                                                                      'shedding_rate': 0.3})})

    # INTERVENTIONS: Vaccination campaigns
    # Parameters of seasonal trapezoidal pattern
    seasonal_env_pars = {
        'period': 365.0,           # in days
        'peak_start_doy': 0.0,     # day of the year
        'ramp_up_dur': 175.26,     # duration in days
        'ramp_dw_dur': 100.0,      # duration in days
        'cutoff_dur': 20.0,        # duration in days
        'max_amp': 1.0}

    # Seasonal environemental pattern
    trapezoidal_pattern = partial_env_trapezoidal(seasonal_env_pars)

    # Intervention that applies the seasonal pattern to modulate the relative transmissibility/exposure environemnt -> people
    exposure_modulation = ty.environmental_trapezoidal_modulation(efficacy=trapezoidal_pattern, start_year=pars["start"])

    # OBSERVATIONS AND REPORTING
    # Create an analyzer that will provide the results we need to compare to target empirical data
    # age_bin_edges = [0, 2, 5, 10, 15, ty.max_age]
    # age_bin_labels = ['<2', '2-4', '5-9', '10-14', '15+']  # human readable labels

    age_bin_edges = [0, 2, 15, ty.max_age]
    age_bin_labels = ['<2', '2-14', '15+']  # human readable labels

    # Track cases by age and by sex -- this analyzer returns counts in number of agents, not people. Scaling can be performed offline.
    record_cases = dict(ti_acute=dict(path=("diseases", "typhoid"), label="cases"))
    record_population = dict(alive=dict(path=("people",)))

    # TODO: rework monitors so one tracks quantities to calculate incidence and the other
    # tracks quantities to caculate prevalence
    monitor_cases = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                     age_bin_labels=age_bin_labels,
                                                     to_record=record_cases,
                                                     resampling_period=1.0,  # Record data on a yearly basis, so we can aggregate later
                                                     aggregate_sex=True,
                                                     aggregate_time="sum",   # Sum over the resampling period (to get incidence)
                                                     record_from=2017.0,
                                                     record_until=2023.0,
                                                     name="monitor_1")

    monitor_population = ty.histograms_by_age_sex_monitor(age_bins=age_bin_edges,
                                                          age_bin_labels=age_bin_labels,
                                                          to_record=record_population,
                                                          resampling_period=1.0,  # Record data on a yearly basis, so we can aggregate later
                                                          aggregate_sex=True,
                                                          aggregate_time="sum",   # Sum the number of people over the resampling period, this is used to calculate incidence rate
                                                          record_from=2017.0,
                                                          record_until=2023.0,
                                                          name="monitor_2")

    model_definition = dict(pars=pars, people=ppl, diseases=[typhoid],
                            demographics=vital_dynamics + [environment],
                            interventions=[exposure_modulation],
                            analyzers=[monitor_cases, monitor_population])

    return model_definition


def vaccination_model():
    # Define the model of a vaccination intervention/campaign
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

    model_definition = dict(interventions=[campaign_vax_2_5_yo])
    return model_definition


def make_sim(scenario="baseline"):
    to_concat = ["diseases", "demographics", "interventions", "analyzers"]
    match scenario:
        case "baseline":
            model = baseline_model()
        case "with_vaccination_campaign":
            baseline = baseline_model()
            vaccination = vaccination_model()  # Add vaccination campaign
            model = dict()
            for key in set(baseline) | set(vaccination):
                if key in to_concat:
                    model[key] = baseline.get(key, []) + vaccination.get(key, [])
                else:
                    model[key] = baseline.get(key)
        case _:
            raise ValueError(f'Unrecognized simulation scenario: {scenario}')

    # PUT EVERYTHING TOGETHER IN A SIMULATION
    sim = ss.Sim(**model)
    return sim


def partial_env_trapezoidal(kwarg_pars):
    """
    Return a partially evaluated function. This means that the pattern is set
    according to the parameters recieved in kwargs_pars. We can get exactly
    the value of the environmental modulation at time t (expressed in years)
    by calling

    my_modulation = partial_env_trapezoidal(**kwargs_pars)
    curent_modulation = my_modulation(t)

    Args:
        kwarg_pars of function typhoidsim.utils_math.asym_trapezoidal

    Returns:
         callable: A partially evaluated ty.asym_trapezoidal
    """
    return partial(ty.asym_trapezoidal, **kwarg_pars)


def run_debug_single_sim(do_plot=True):
    """ Run one simulation"""
    sim = make_sim(scenario="baseline")
    sim.run()
    if do_plot:
        timevec = sim.get_analyzers()[0].yearvec
        sim.plot(key="typhoid_")
        plt.show()
    breakpoint()
    return sim


def run_debug_multisim(do_plot=True):
    sim1 = make_sim()
    sim2 = make_sim()
    sim3 = make_sim()
    sim4 = make_sim()


    sims = sc.autolist()
    sims.append(sim2)
    sims.append(sim1)
    sims.append(sim3)
    sims.append(sim4)


    # Let's change a parameter in sim2
    sim2.initialize()
    sim2.diseases.typhoid.pars.tai = 35_000

    sim3.initialize()
    sim3.diseases.typhoid.pars.tai = 80_000

    sim4.initialize()
    sim4.diseases.typhoid.pars.tai = 150_000

    msim = ss.MultiSim(sims)
    msim.run()
    if do_plot:
        for sim in msim.sims:
            # Display the entire simulation period -- takes long to plot everything if simulation is long
            sim.plot(key="typhoid_")
            # Display a fraction of the simulation period
            #ty.plot_sim(sim, key="typhoid_", display_from=2010.0, display_until=2025.0)
        plt.show()
    return msim


if __name__ == "__main__":
    #run_debug_single_sim()
    run_debug_multisim(do_plot=True)
