"""
Typhoid model.

"""

import numpy as np
import starsim as ss

import typhoidsim.utils as tyu
import typhoidsim.patterns as typ
import typhoidsim.defaults as tyd

__all__ = ["TyphoidSimple"]


class TyphoidSimple(ss.Infection):
    """
    Typhoid module that includes the natural history of the disease in a human
    agent and also environmental 'state' variables and parameters that
    capture the growth and decay of S. typhii bacteria in contaminated resources.

    This is a proof-of-concept 'monolithic' and 'simplfieied' implementation,
    similar to starsim's Cholera module.

    The parent class ss.Infection has the following states

    ss.BoolArr('susceptible', default=True) -- for typhoid we may need to reset
    this to False
    ss.BoolArr('infected'), -- default False
    ss.FloatArr('rel_sus', default=1.0),
    ss.FloatArr('rel_trans', default=1.0),
    ss.FloatArr('ti_infected'),)

    """

    def __init__(self, pars=None, *args, **kwargs):
        """Initialize with parameters"""
        super().__init__()
        self.default_pars(
            # Initial conditions and beta
            beta=1.0,  # Placeholder value
            init_prev=ss.bernoulli(0.005),
            # Natural history parameters, all specified in days
            # Age-based exposure
            age_exposure_slope=1.0,
            dur_prep2acute=ss.lognorm_ex(mean=0.1, stdev=0.0),  # Prepatent -> acute
            dur_prep2subcl=ss.lognorm_ex(
                mean=0.1, stdev=0.0
            ),  # Prepatent -> subclinical
            dur_acute2dead=ss.lognorm_ex(mean=0.1, stdev=0.0),  # Acute -> Dead
            dur_subcl2chro=ss.lognorm_ex(
                mean=0.1, stdev=0.0
            ),  # Subclinical - > chronic
            dur_subcl2rec=ss.lognorm_ex(mean=0.1, stdev=0.0),  # Subclinical - > chronic
            dur_acute2rec=ss.lognorm_ex(mean=0.1, stdev=0.0),  # Subclinical - > chronic
            dur_acute2chro=ss.lognorm_ex(mean=0.1, stdev=0.0),  # Acute -> Chronic
            p_death=ss.bernoulli(p=0.2),  # Probability of dying from acute
            p_acute=ss.bernoulli(p=0.1),  # Probability of becoming acute
            p_chronic=ss.bernoulli(
                p=0.015
            ),  # Prob of becoming chronic carrier in persons with gallstones
            # Environmental parameters - long-cycle CCVT
            environment=dict(
                beta=0.0,
                init_prev=ss.bernoulli(0.0),
                decay_rate=0.0,
                # Rate at which bacteria in the environment change (per day), >0 decays, <0 grows
            ),
            ppl_env_transmission=dict(
                # Interaction parameters between people and environment
                # Rate at which infectious people shed bacteria to the environment (per day),
                ppl2env_shedding_rate=0.0,
                # Probability of environmental transmission - filled out later
                env2ppl_p_transmit=ss.bernoulli(p=0),
            ),
        )
        self.update_pars(pars, **kwargs)

        # Boolean states
        self.add_states(
            # Susceptible & infected are added automatically, here we add the rest
            ss.BoolArr("exposed"),  # NOTE: i don't think we need this state here
            ss.BoolArr("prepatent"),
            ss.BoolArr("acute"),
            ss.BoolArr("subclinical"),
            ss.BoolArr("chronic"),
            ss.BoolArr("recovered"),
            ss.FloatArr("n_infections"),
            # Timepoint states
            ss.FloatArr("ti_exposed"),
            ss.FloatArr("ti_susceptible"),
            ss.FloatArr("ti_prepatent"),
            ss.FloatArr("ti_subclinical"),
            ss.FloatArr("ti_acute"),
            ss.FloatArr("ti_chronic"),
            ss.FloatArr("ti_recovered"),
            ss.FloatArr("ti_dead"),
        )
        # NOTE: Typhoid may assume that all individuals are born into an
        # a class where they cannot get infected, and then
        # move to the susceptible class at probabilities
        # for each age. The ss.Infection class set the self.susceptible state
        # to True by default, so here reset this array to False
        self.make_impervious()

        # self.init_state_vars(
        #     **kwargs,
        # )
        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    @property
    def asymptomatic(self):
        return self.prepatent | self.chronic

    @property
    def symptomatic(self):
        return self.subclinical & self.acute

    def init_results(self):
        """
        Initialise Result objects
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, "new_deaths", npts, dtype=int),
            ss.Result(self.name, "cum_deaths", npts, dtype=int),
            ss.Result(self.name, "env_prevalence", npts, dtype=float),
            ss.Result(self.name, "env_concentration", npts, dtype=float),
        ]
        return

    # def init_state_vars(self, **kwargs):
    #     """
    #     State variables, states that are not defined on a per-agent basis,
    #     and that could be user-defined.
    #     """
    #
    #     self.state_vars += [
    #         typ.Pattern(self.name, "env_prevalence", 2, dtype=float),
    #         typ.Pattern(self.name, "env_concentration", 2, dtype=float),
    #     ]
    #     return

    def update_pre(self):
        """
        Update the progression of the disease -- handles disease
        state transitions.
        """

        ti = self.sim.ti  # current timestep

        # Check who becomes susceptible in this timestep age 0-20
        self.make_susceptible()
        # Age-based susceptibility in chidldren <= 6 years old
        # self.increase_childhood_susceptibility()

        # Natural history flow - handle transitions
        # between any two disease states or stages
        self.progress_to_prepatent(ti)
        self.progress_to_symptomatic(ti)  # Both acute and sublclinical
        self.progress_to_chronic(ti)
        self.progress_to_dead(ti)
        self.progress_to_recovered(ti)
        self.progress_to_susceptible(ti)
        #self.update_environmental_prevalence()
        return

    def progress_to_prepatent(self, ti):
        infected = (self.exposed & (self.ti_infected <= ti)).uids
        self.infected[infected] = True
        self.prepatent[infected] = True

        # N_i: number of prior infections,
        # used to determine the probability of becoming
        # infected upon exposure (1-P)**N_i,
        # that provides almost sterilising immunity
        # in hyper-endemic settings,
        # but we may want to incorporate a mechanism
        # to wane naturally acquired immunity.
        self.n_infections[infected] += 1.0
        self.exposed = False

    def progress_to_symptomatic(self, ti):
        # Progress petatent -> acute
        prep2acute = (self.prepatent & (self.ti_acute <= ti)).uids
        self.acute[prep2acute] = True
        self.prepatent[prep2acute] = False

        # Progress prepatent -> subclinical
        prep2subcl = (self.prepatent & (self.ti_subclinical <= ti)).uids
        self.subclinical[prep2subcl] = True
        self.prepatent[prep2subcl] = False

    def progress_to_chronic(self, ti):
        # Progress acute -> chronic
        acu2chro = (self.acute & (self.ti_chronic <= ti)).uids
        self.chronic[acu2chro] = True
        self.acute[acu2chro] = False

        # Progress subclinical -> chronic
        sub2chro = (self.subclinical & (self.ti_chronic <= ti)).uids
        self.chronic[sub2chro] = True
        self.subclinical[sub2chro] = False

    def progress_to_dead(self, ti):
        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)
        pass

    def progress_to_recovered(self, ti):
        # handle acute pathway
        acu2rec = (self.acute & (self.ti_recovered <= ti) & (self.ti_dead <= ti)).uids
        self.recovered[acu2rec] = True
        self.acute[acu2rec] = False
        self.infected[acu2rec] = False

        # handle subclinical pathway
        sub2rec = (
            self.subclinical & (self.ti_recovered <= ti) & (self.ti_dead <= ti)
        ).uids
        self.recovered[sub2rec] = True
        self.subclinical[sub2rec] = False
        self.infected[sub2rec] = False

    def progress_to_susceptible(self, ti):
        # Make agents susceptible again
        rec2suc = (self.recovered & (self.ti_susceptible <= ti)).uids
        self.susceptible[rec2suc] = True
        self.recovered[rec2suc] = False

    def update_environmental_prevalence(self):
        """
        Calculate environmental prevalence
        long-cycle CCVT
        """
        env_pars = self.pars.environment
        trans_pars = self.pars.ppl_env_transmission

        sv = self.state_vars
        ti = self.sim.ti

        n_symptomatic = self.symptomatic.sum()
        n_asymptomatic = self.asymptomatic.sum()

        # Update environment
        previous_ep = sv.env_prevalence[ti - 1]
        bacteria_from_env = previous_ep * (1.0 - env_pars.decay_rate)

        bacteria_from_ppl = trans_pars.shedding_rate * (
            n_symptomatic + trans_pars.asymp_trans * n_asymptomatic
        )

        # Current prevalence
        sv.env_prevalence[ti] = bacteria_from_env + bacteria_from_ppl
        sv.env_concentration[ti] = sv.env_prevalence[ti] / (
            sv.env_prevalence[ti] + env_pars.half_sat_rate
        )
        return

    def set_prognoses(self, uids, source_uids=None):
        """
        Here we define the whole natural history for every agent
        that has been infected agent. The progression of this natural
        history can be altered by interventions, or other diseases.

        Duration. The duration of the prepatent stage of an individual’s
        infection is calculated at the beginning of the infection as a
        draw from a log-normal distribution using one of three
        mu-sigma parameter pairs. The mu-sigma pair is selected based on the
        "quantization" of the exposure amount into one of three buckets
        using various thresholds. (Glynn et al., 1995).

        Currently, all infections from the Contact route are assumed to be
        a High dose prepatent duration.

        """

        super().set_prognoses(uids, source_uids)
        ti = self.sim.ti
        dt = self.sim.dt

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.prepatent[uids] = True
        self.ti_prepatent[uids] = ti
        self.ti_infected[uids] = ti

        p = self.pars

        # Determine who will become acute and who will become subclinical
        acu_scl = p.p_acute.filter(uids, both=True)
        acute_uids, subcl_uids = acu_scl

        # Determine when prepatent becomes acute
        self.ti_acute[acute_uids] = ti + p.dur_prep2acute.rvs(acute_uids) / dt

        # Determine when prepatent becomes subclinical
        self.ti_subclinical[subcl_uids] = ti + p.dur_prep2subcl.rvs(subcl_uids) / dt

        # Determine who becomes a (chronic) carrier (from acute and sublclinical)
        carrier_uids = p.p_chronic.filter(uids)

        # From the acute cases, determine who can die because they don't become carriers
        can_die_uids = np.setdiff1d(acute_uids, carrier_uids)

        # From the acutes who do not become carriers, determine who actually
        # dies and who recovers
        will_die = p.p_death.rvs(can_die_uids)
        dead_uids = can_die_uids[will_die]
        recovered_uids = can_die_uids[~will_die]

        self.ti_dead[dead_uids] = (
            self.ti_acute[dead_uids] + p.dur_acute2dead.rvs(dead_uids) / dt
        )

        self.ti_recovered[recovered_uids] = (
            self.ti_acute[recovered_uids] + p.dur_acute2rec.rvs(recovered_uids) / dt
        )

        # From the sublinical cases, determine who can recover because they don't become carriers
        can_recover_uids = np.setdiff1d(subcl_uids, carrier_uids)
        # Determine when non-carriers recover
        self.ti_recovered[can_recover_uids] = (
            self.ti_subclinical[can_recover_uids]
            + p.dur_subcl2rec.rvs(can_recover_uids) / dt
        )

        return

    def make_new_cases(self):
        """Add short-cycle transmission and long-cycle transmission transmission"""
        # Make new cases via person-to-person transmission
        super().make_new_cases()

        new_cases = self.environmental_transmission()

        if new_cases.any():
            self.set_prognoses(new_cases, source_uids=None)
        return

    def environmental_transmission(self):
        # Make new cases via indirect transmission
        # env_pars = self.pars.environment
        # sv = self.state_variables
        # p_transmit = env_pars.beta * sv.env_concentration[self.sim.ti]
        # env_pars.p_transmit.set(p=p_transmit)
        # new_cases = env_pars.p_transmit.filter(self.sim.people.uid[self.susceptible])
        new_cases = ss.FloatArr([])
        return new_cases

    def update_death(self, uids):
        """Reset states for dead agents"""
        for state in [
            "susceptible",
            "exposed",
            "infected",
            "prepatent",
            "acute",
            "subclinical",
            "chronic",
            "recovered",
        ]:
            self.statesdict[state][uids] = False
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[: ti + 1])
        return

    def make_new_cases_long_cycle(self):
        pass

    def make_susceptible(self):
        """
        Placeholder function to make agent susceptible as a function of
        their age.

        From Gauld et al. 2018:
        'Our model assumes all individuals are born into an unexposed class
        and move to the susceptible class at probabilities for each age.
        Specifically, at each month of age a fitted curve determines the
        probability of an individual entering the susceptible class.

        The curve is anchored at 0% exposure at birth, and 100% exposure at age
        20 years, with a free slope parameter (S) determining the concavity/shape
        of the function (Fig 2B).'

        """

        max_age = 20.0  # TODO: make configurbale?
        unexposed = (~self.susceptible).uids
        self.susceptible[unexposed] = ss.bernoulli(
            p=tyu.sigmoid(
                self.sim.people.age[unexposed], max_age, self.pars.age_exposure_slope
            )
        )

    def make_impervious(self):
        self.susceptible[(self.susceptible).uids] = False


def environmental_transmission(people, disease, contaminated_environment, current_ti):
    """
    Transmission and make new cases can form the basis for
    a Propagation class, which would be a specific class of
    the more abstract Connector.
    """
    # Make new cases via indirect transmission
    env_pars = contaminated_environment.pars
    p_transmit = env_pars.beta * contaminated_environment.concentration[current_ti]
    env_pars.p_transmit.set(p=p_transmit)
    new_cases = env_pars.p_env_transmit.filter(people.uid[disease.susceptible])
    return new_cases


def update_susceptible_children_pop(people, dt):
    """
    From:
    https://github.com/jgauld/DtkTrunk/blob/Typhoid-Ongoing/Eradication/SusceptibilityTyphoid.cpp

    NOTE:
        Fraction of children that become susceptible upon raching a certain
        age threhsold are dummy values and hardcoded until we know if we want
        this functionality or not
    """

    age_boundary_6m = 0.5 * tyd.days_per_year  # in days
    age_boundary_3y = 3.0 * tyd.days_per_year  # in days
    age_boundary_6y = 6.0 * tyd.days_per_year  # in days

    p_6m = ss.bernoulli(0.1).initialize()
    p_3y = ss.bernoulli(0.1).initialize()
    p_6y = ss.bernoulli(0.1).initialize()

    uids_6m = (
        (people.age >= age_boundary_6m) & ((people.age - dt) < age_boundary_6m)
    ).uids

    uids_3y = (
        (people.age >= age_boundary_3y) & ((people.age - dt) < age_boundary_3y)
    ).uids

    uids_6y = (
        (people.age >= age_boundary_6y) & ((people.age - dt) < age_boundary_6y)
    ).uids

    people = make_children_susceptible(people, uids_6m, p_6m)
    people = make_children_susceptible(people, uids_3y, p_3y)
    people = make_children_susceptible(people, uids_6y, p_6y)

    return people


def make_children_susceptible(people, uids_aged_x, prop_susceptible):
    new_susc = prop_susceptible.filter(people.uid[uids_aged_x])
    people.typhoid.susceptible[new_susc] = True
    return people


class Typhoid(ss.Infection):
    """
    Typhoid module that only includes the natural history of the disease in a human
    agent. This module is expected to interact with other modules such as
    Gallstones and Environment.
    """

    pass
