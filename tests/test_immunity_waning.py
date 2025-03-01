"""
Test intervention with immunity waning and boosters.
"""
import sciris as sc
import starsim as ss
import typhoidsim as ty


def make_sim_with_defaults(n_agents=5_000):

    # Define the parameters
    pars = sc.objdict(
        start=2000,         # Starting year
        dur=1.0,            # Number of years to simulate
        total_pop=50e6,     # Total population size
        pop_scale=None,     #
        n_agents=n_agents,  #
        dt=1.0/365.0,       # Timestep of 1 day, expressed in years
        rand_seed=2,        # Set a non-default seed
    )

    ppl = ss.People(pars["n_agents"])

    demographics = [
        ss.Births(birth_rate=0),
        ss.Deaths(death_rate=0)
    ]

    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.0)})

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        start_year=2000.0,
        prob=0.66,
        booster1_interval=0.33,  # interval between receiving first dose and first booster
        booster1_prob=1.0,
        booster2_interval=0.66,  # interval between receiving first dose and second booster
        booster2_prob=1.0,
        prob_type="annual",
        debug=True,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 0.0,
                  'max_age': 2.0}
        )


    #
    sim = ss.Sim(
        pars=pars,
        demographics=demographics,
        diseases=typhoid,
        interventions=campaign_vax_2_5_yo,
        people=ppl,
        label=f"n_agents={pars['n_agents']}"
        )
    return sim


def make_sim_non_default_with_constant_dist(n_agents=5_000):
    # Define the parameters
    pars = sc.objdict(
        start=2000,         # Starting year
        dur=1.0,            # Number of years to simulate
        total_pop=50e6,     # Total population size
        pop_scale=None,     #
        n_agents=n_agents,  #
        dt=1.0/365.0,       # Timestep of 1 day, expressed in years
        rand_seed=2,        # Set a non-default seed
    )

    ppl = ss.People(pars["n_agents"])

    demographics = [
        ss.Births(birth_rate=0),
        ss.Deaths(death_rate=0)
    ]

    typhoid = ty.Typhoid(pars={"init_prev": ss.bernoulli(p=0.0)})

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        start_year=2000.0,
        prob=0.66,
        booster1_interval=0.33,  # interval between receiving first dose and first booster
        booster1_prob=1.0,
        booster2_interval=0.66,  # interval between receiving first dose and second booster
        booster2_prob=1.0,
        # We can now use different age bins for each parameter
        imm_decay=ss.constant(v=ty.imm_decay_by_age(age_bins=[0, 5, 10, 15], vals=[300.0, 200.0, 100.0])),
        imm_ve0=ss.constant(v=ty.imm_ve0_by_age(age_bins=[0, 2, 5, 15], vals=[1.0, 0.7, 0.5])),
        prob_type="annual",
        debug=True,  # only use for this example to keep track of each individual's acquired immunity level over time
        age_pars={'min_age': 0.0,
                  'max_age': 2.0}
        )


    #
    sim = ss.Sim(
        pars=pars,
        demographics=demographics,
        diseases=typhoid,
        interventions=campaign_vax_2_5_yo,
        people=ppl,
        label=f"n_agents={pars['n_agents']}"
        )
    return sim


def run_sim(sim):
    sim.run()
    ty.to_df(sim)
    assert True
    return


if __name__ == '__main__':
    sim = make_sim_with_defaults()
    run_sim(sim)
    sim = make_sim_non_default_with_constant_dist()
    run_sim(sim)

