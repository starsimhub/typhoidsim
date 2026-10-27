"""
Vaccination campaigns with waning immunity and a booster, in two age groups: the
protection of each vaccinated person over time, and the infections by age and
vaccination status
"""

import matplotlib.pyplot as plt
import numpy as np

import sciris as sc
import starsim as ss
import typhoidsim as ty

age_bin_edges = [0, 2, 5, 10, 15, 20, 40, 60]


def make_sim(n_agents=10_000):

    # Define the parameters
    pars = sc.objdict(
        start=2000,        # Starting year
        dur=ss.years(2),   # Number of years to simulate
        n_agents=n_agents, #
        dt=ss.days(1),     # Timestep of 1 day
        verbose=0,         # Print details of the run
        rand_seed=2,       # Set a non-default seed
    )

    # Who
    ppl = ss.People(pars["n_agents"])

    typhoid = ty.Typhoid(init_prev=0.05, p_death=0)

    network = ss.RandomNet(n_contacts=5)

    campaign_vax_2_5_yo = ty.vaccination_with_waning(
        prob=0.4,
        booster1_interval=1.0,  # interval between receiving first dose and first booster
        booster1_prob=0.5,
        start_year=2000.0,
        end_year=2002.0,
        prob_type="interval",
        imm_ve0=ty.imm_ve0_by_age(age_bins=[0.75, 2.0, 5.0, 15.0, 125.0], vals=[1.0, 0.9, 0.3, 0.3]),
        age_pars={'min_age': 2.0,
                  'max_age': 5.0},
        name = "campaign_1"
        )

    campaign_vax_5_10_yo = ty.vaccination_with_waning(
        prob=0.8,
        booster1_interval=1.0,  # interval between receiving first dose and first booster
        booster1_prob=0.5,
        start_year=2000.0,
        end_year=2002.0,
        prob_type="interval",
        imm_ve0=ty.imm_ve0_by_age(age_bins=[0.75, 2.0, 5.0, 15.0, 125.0], vals=[1.0, 0.9, 0.3, 0.3]),
        age_pars={'min_age': 5.0,
                  'max_age': 10.0},
        name="campaign_2"
        )

    # Keep track of each person's protection from the first campaign over time (was debug=True on the vaccination), and the cases by age (was histograms_by_age_sex_monitor)
    protection = []
    def track_protection(sim):
        vx = sim.interventions.campaign_1.product
        held = vx.held.uids
        t = ss.years((sim.ti - vx.ti_delivered[held])*sim.t.dt_year) # Years since the last dose
        out = np.zeros(n_agents) # By UID; there are no births
        out[held] = vx.waning(t, sim, held)
        protection.append(out)

    sim = ss.Sim(
        pars=pars,
        diseases=typhoid,
        networks=network,
        interventions=campaign_vax_2_5_yo + campaign_vax_5_10_yo,
        analyzers=track_protection,
        people=ppl,
        stratify=dict(age=age_bin_edges),
        label=f"n_agents={pars['n_agents']}"
        )
    return sim, protection


sim, protection = make_sim()
sim.run()

# The protection of the first 100 people vaccinated in the first campaign
vaccine = sim.interventions.campaign_1.product
vx_uids = np.isfinite(vaccine.ti_delivered.raw[:len(sim.people)]).nonzero()[0]
protection = np.array(protection)[:, vx_uids[:100]]
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].plot(sim.t.yearvec[:len(protection)], protection)
axs[0].set_title('Protection of people vaccinated in campaign 1')

# Cases by age, per year
cases = sim.results.typhoid.infections.by('age').resample('year')
axs[1].plot(cases.columns, cases.values.T, 'o-', label=[str(t) for t in cases.timevec])
axs[1].set_title('Infections by age, per year')
axs[1].legend()

# Infections per person by age and vaccination status (was histogram_by_vaccination_status), counting all infections over the run
age = sim.people.age.values
n_infections = sim.diseases.typhoid['n_infections'].values
vaccinated = np.isfinite(sim.interventions.campaign_1.product.ti_delivered.values) | np.isfinite(sim.interventions.campaign_2.product.ti_delivered.values)
groups = np.digitize(age, age_bin_edges) - 1
for vx, label in [(True, 'Vaccinated'), (False, 'Unvaccinated')]:
    rate = [n_infections[(groups == g) & (vaccinated == vx)].mean() if ((groups == g) & (vaccinated == vx)).any() else np.nan for g in range(len(age_bin_edges))]
    axs[2].plot(cases.columns, rate, 'o-', label=label)
axs[2].set_title('Infections per person by age')
axs[2].legend()
plt.show()
