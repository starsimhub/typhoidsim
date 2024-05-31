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

    # TODO: add link to specification document
    Unless otherwise specified, parameters come from: XXX

    """

    def __init__(self, pars=None, *args, **kwargs):
        """Initialize with parameters"""
        super().__init__()
        self.default_pars(
            # Initial conditions and beta
            beta=1.0,  # Placeholder value
            init_prev=ss.bernoulli(0.005),
            # Natural history parameters,
            dur_prep2next=ss.lognorm_ex(mean=1.548, stdev=0.3442),  # 'High dose' prepatent duration, in days.
            dur_acute2next_le30=ss.lognorm_ex(mean=1.172, stdev=0.483),   # Acute duration for under (<) 30 yo, in weeks.
            dur_acute2next_geq30=ss.lognorm_ex(mean=1.258, stdev=0.788),  # Acute duration for over (>=) 30 yo, in weeks.
            dur_subcl2next_le30=ss.lognorm_ex(mean=1.172, stdev=0.483),   # Subclinical duration for under (<) 30 yo, in weeks.
            dur_subcl2next_geq30=ss.lognorm_ex(mean=1.172, stdev=0.788),  # Subclinical duration for over (>=) 30 yo, in weeks.
            p_acute=ss.bernoulli(p=0.234),   # Prob of becoming acute (or symptomatic)
            p_chro=ss.bernoulli(p=0.150),     # Prob of becoming chronic carrier from acute or clinical infection, average multiplicative factor, same for females and males.
            p_death=ss.bernoulli(p=0.001),  # Probability of dying from acute, context dependent, and by default set to something zero or something very small

            # Within-host parameters
            # Age-based exposure
            age_exposure_slope=1.0,
            # Infectiousness parameters
            tai=40_000,  # Typhoid acute infectiousness, represents number of colony-forming units of S. typhi
            tpri=0.4,    # Typhoid relative (to acute) prepatent infectiousness
            tsri=0.8,    # Typhoid relative (to acute) subclinic infectiousness
            tcri=0.1,    # Typhoid relative (to acute) chronic infectiousness

            # Environmental parameters - long-cycle CCVT
            environment=dict(
                beta=0.0,
                init_prev=ss.bernoulli(0.0),
                decay_rate=0.0,
                # Rate at which bacteria in the environment change (per day), >0 decays, <0 grows
            ),
            # Environmental tranmission parameters, temporary living here, until we move environment somwhere else
            transmission=dict(
                # Interaction parameters between people and environment
                # Rate at which infectious people shed colony-forming units to the environment (per day),
                ppl2env_shedding_rate=1.0,
                # Probability of environmental transmission - filled out later
                env2ppl_exposure_rate=ss.poisson(lam=10.0),
                env2ppl_p_inf = ss.bernoulli(p=0.0)  ## updated later
            ),
        )
        self.update_pars(pars, **kwargs)

        # Boolean states
        self.add_states(
            # Infection life cycle states
            # Susceptible & infected are added automatically, here we add the rest
            ss.BoolArr("exposed"),
            ss.BoolArr("prepatent"),
            ss.BoolArr("acute"),
            ss.BoolArr("subclinical"),
            ss.BoolArr("chronic"),
            ss.BoolArr("recovered"),

            # States that track immunity-related quantities or variables
            # and depend on infection states
            ss.FloatArr("n_exposures"),
            ss.FloatArr("cfu_doses"),
            ss.FloatArr("n_infections"),
            ss.FloatArr("infectiousness"),
            ss.FloatArr("p_chronic"),
            ss.FloatArr("immunity"),

            # States that track timing of events
            ss.FloatArr("ti_exposed"),
            ss.FloatArr("ti_susceptible"),
            ss.FloatArr("ti_prepatent"),
            ss.FloatArr("ti_subclinical"),
            ss.FloatArr("ti_acute"),
            ss.FloatArr("ti_chronic"),
            ss.FloatArr("ti_recovered"),
            ss.FloatArr("ti_dead"),
        )

        # self.init_state_vars(
        #     **kwargs,
        # )
        return

    @property
    def infectious(self):
        return self.infected

    @property
    def asymptomatic(self):
        return self.prepatent | self.subclinical | self.chronic

    @property
    def symptomatic(self):
        return self.acute

    def init_results(self):
        """
        Initialise Result objects
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, "new_deaths", npts, dtype=int),
            ss.Result(self.name, "cum_deaths", npts, dtype=int),
            ss.Result(self.name, "env_cfu", npts, dtype=float),
        ]
        return

    def init_vals(self):
        """
        Set initial values for states. This could involve passing in a full
        set of initial conditions, or using init_prev (initial prevalence), or other.

        Note that this is different to initialization of the Arr objects i.e.,
        creating their dynamic array, linking them to a People instance.
        That should have already taken place by the time this method is called.
        """
        # NOTE: Typhoid may assume that all individuals are born into an
        # a class where they cannot get infected, and then
        # move to the susceptible class at probabilities
        # for each age. The ss.Infection class set the self.susceptible state
        # to True by default, so here reset this array to False
        self.make_impervious()

        if self.pars.init_prev is None and self.pars.env_pars.init_prev is None:
            return

        if self.pars.init_prev is not None:
            # Initial cases from person-to-person transmission
            initial_cases_contact = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases_contact)
        if self.pars.env_pars.init_prev is not None:
            # Initial cases from environment-to-person transmission
            initial_cases_env = self.pars.env_pars.init_prev.filter((~self.infected).uids)
            self.set_prognoses(initial_cases_env)

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

    # Methods that are specific to a single state (though they can modify other
    # states)
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
        self.exposed[self.susceptible.uids] = False
        self.susceptible[self.susceptible.uids] = False

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

    def update_pre(self):
        """
        Update the progression of the disease -- handles disease
        state transitions. In the typical simulation flow this method is called
        before propagating the infection/disease is propagated via
        make_new_cases()
        """

        ti = self.sim.ti  # current timestep

        # Check who becomes susceptible in this timestep age 0-20
        self.make_susceptible()
        # Age-based susceptibility in chidldren <= 6 years old
        # self.increase_childhood_susceptibility()

        # The infection life cycle or natural history flow
        # handles transitions between any two infection states or stages
        self.progress_to_prepatent(ti)  # Incubation period
        self.progress_to_diseased(ti)   # Both acute and subclinical
        self.progress_to_chronic(ti)
        self.progress_to_dead(ti)
        self.progress_to_recovered(ti)
        self.progress_to_susceptible(ti)
        # self.update_environmental_prevalence()
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
        self.infectiousness[infected] = self.pars.tai * self.pars.tpri

    def progress_to_diseased(self, ti):
        # Progress pretatent -> acute
        prep2acute = (self.prepatent & (self.ti_acute <= ti)).uids
        self.acute[prep2acute] = True
        self.prepatent[prep2acute] = False
        self.infectiousness[prep2acute] = self.p.tai

        # Progress prepatent -> subclinical
        prep2subcl = (self.prepatent & (self.ti_subclinical <= ti)).uids
        self.subclinical[prep2subcl] = True
        self.prepatent[prep2subcl] = False
        self.infectiousness[prep2subcl] = self.pars.tai * self.tsri

    def progress_to_chronic(self, ti):
        # Progress acute -> chronic
        acu2chro = (self.acute & (self.ti_chronic <= ti)).uids
        self.chronic[acu2chro] = True
        self.acute[acu2chro] = False
        self.infectiousness[acu2chro] = self.pars.tai * self.pars.tcri

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
        self.exposed[acu2rec] = False
        self.infectiousness[acu2rec] = 0.0

        # handle subclinical pathway
        sub2rec = (
            self.subclinical & (self.ti_recovered <= ti) & (self.ti_dead <= ti)
        ).uids
        self.recovered[sub2rec] = True
        self.subclinical[sub2rec] = False
        self.infected[sub2rec] = False
        self.infectiousness[sub2rec] = 0.0

    def progress_to_susceptible(self, ti):
        # Make agents susceptible again
        rec2suc = (self.recovered & (self.ti_susceptible <= ti)).uids
        self.susceptible[rec2suc] = True
        self.recovered[rec2suc] = False

    def get_acute_duration_by_age(self, uids):
        """
        TODO: refactor in to a single function that returns both
        acute and subclinical durations, though that would prevent
        further differentiating between those two stages (ie, if the
        if we wanted to change the 'threshold' age in one of the
        stages but not the other. )
        """
        p = self.pars
        dt = self.sim.dt

        dur_acu = self.ti_acute[uids]
        # From the acute uids, who is under or over 30
        under30 = np.isin(uids, (self.sim.people.age < 30.0).uids)
        over30  = np.isin(uids, (self.sim.people.age >= 30.0).uids)

        # convert duration pars in weeks -> to days -> to timesteps
        dur_acu[under30] += ((p.dur_acute2next_le30.rvs(uids[under30]) *
                                    tyd.days_per_week) / dt)
        # convert duration pars in weeks -> to days -> to timesteps
        dur_acu[over30] += ((p.dur_acute2next_geq30.rvs(uids[over30]) *
                                   tyd.days_per_week) / dt)  # in timesteps
        return dur_acu

    def get_subclinical_duration_by_age(self, uids):
        p = self.pars
        dt = self.sim.dt

        dur_scl = self.ti_subclinical[uids]
        # From the subclinical uids, who is under or over 30
        under30 = np.isin(uids, (self.sim.people.age < 30.0).uids)
        over30  = np.isin(uids, (self.sim.people.age >= 30.0).uids)

        # convert duration pars in weeks -> to days -> to timesteps
        dur_scl[under30] += ((p.dur_subcl2next_le30.rvs(uids[under30]) *
                                    tyd.days_per_week) / dt)
        # convert duration pars in weeks -> to days -> to timesteps
        dur_scl[over30] += ((p.dur_subcl2next_geq30.rvs(uids[over30]) *
                                   tyd.days_per_week) / dt)  # in timesteps
        return dur_scl

    def will_become_chronic_carrier(self, uids):
        """Determine who will become a chronic carrier"""
        p = self.pars
        if p.p_chro is not None:
            # Use an "average" probability for everyone
            return p.p_chro.filter(uids)

        # Estimate by age and gender probabilities
        # TODO: implement

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
        if len(uids) > len(np.unique(uids)):
            UserWarning('Removing duplicated uids')
            uids = np.unique(uids)
        super().set_prognoses(uids, source_uids)
        p = self.pars
        ti = self.sim.ti
        dt = self.sim.dt

        # Set value of states associated to being infected, and record events
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.exposed[uids] = True
        self.prepatent[uids] = True
        self.ti_prepatent[uids] = ti
        self.ti_infected[uids] = ti


        # Set duration of prepatent state, by defining when they will
        # progress to the next state (either acute or sublinical)
        dur_pre = ti + p.dur_prep2next.rvs(uids) / dt

        # Determine who will become acute and who will become subclinical
        acu_scl = p.p_acute.filter(uids, both=True)
        acute_uids, subcl_uids = acu_scl

        # Set prepatent duration of those who will become acute
        self.ti_acute[acute_uids] = ti + dur_pre[np.isin(uids, acute_uids)]

        # Set prepatent duration of those who will become subclinical
        self.ti_subclinical[subcl_uids] = ti + dur_pre[np.isin(uids, subcl_uids)]

        # Estimate duration of acute stage
        dur_acu = self.get_acute_duration_by_age(acute_uids)
        # Estimate duration of subclinical by age
        dur_scl = self.get_subclinical_duration_by_age(subcl_uids)

        # Determine who becomes a (chronic) carrier (from acute and sublclinical)
        carrier_uids = self.will_become_chronic_carrier(acute_uids.concat(subcl_uids))

        # From the acute cases, determine who can die because they don't become carriers
        can_die_uids = np.setdiff1d(acute_uids, carrier_uids)

        # From the acutes who do not become carriers, determine who recovers and who dies
        will_die = p.p_death.rvs(can_die_uids)
        dead_uids = can_die_uids[will_die]
        rec_from_acu_uids = can_die_uids[~will_die]

        # Get sublinical cases that recover because they won't become carriers
        rec_from_subcl_uids = np.setdiff1d(subcl_uids, carrier_uids)

        will_recover_uids = rec_from_acu_uids.concat(rec_from_subcl_uids)

        # Determine when non-carriers recover and become susceptible again,
        # NOTE: we do not have to track a recovered state, we can simply output results
        # that track the 'concept' of a recovered state
        self.ti_recovered[rec_from_subcl_uids] = self.ti_subclinical[rec_from_subcl_uids] + dur_scl[np.isin(subcl_uids, rec_from_subcl_uids)]
        self.ti_recovered[rec_from_acu_uids]   = self.ti_acute[rec_from_acu_uids]         + dur_acu[np.isin(acute_uids, rec_from_acu_uids)]

        # NOTE: typhoid can get very low mortality (in particular with treatment),
        # so there is a high chance of getting empty dead_uids. If that happens,
        # the line below may seg fault. Just in case check first.
        if dead_uids.size:
            self.ti_dead[dead_uids] = self.ti_acute[dead_uids] + dur_acu[np.isin(acute_uids, dead_uids)]

        self.ti_susceptible[will_recover_uids] = self.ti_recovered[will_recover_uids] + 1.0  # recover in the next time step, just to make things tidy

        return

    #  Transmission-realated methods - interaction between agents and "else" (other agents)
    #  or the environment
    def make_new_cases(self):
        """Add short-cycle transmission and long-cycle transmission transmission"""
        # Make new cases via person-to-person transmission
        super().make_new_cases()

        #new_cases = self.environmental_transmission()

        #if len(new_cases):
        #    self.set_prognoses(new_cases, source_uids=None)
        return

    def make_new_cases_environmental_transmission(self):
        """
        TODO: this should move to a different module
        1. infected individuals shed into theenvironment,
        2. individuals get exposed by the environment (increases their n_exposures)
        3.

        , the environment
        decays).
        """
        trans_pars = self.pars.env_ppl.transmission
        dt = self.sim.dt

        # Infectious individuals shed contagion into both the CPs
        shedded_contagion = trans_pars.shedding_rate * self.infectiousness[self.infected].sum()

        # Expose to environment
        self.expose_to_environment(dt)
        p_inf = self.drc()
        inf_uids = trans_pars.env2ppl_p_inf(alive & ~self.infected, p=p_inf)

        p_transmit = trans_pars.beta * sv.env_cfu[self.sim.ti]

        # Make new cases via indirect transmission
        # env_pars = self.pars.environment
        # sv = self.state_variables
        # p_transmit = env_pars.beta * sv.env_concentration[self.sim.ti]
        # env_pars.p_transmit.set(p=p_transmit)
        # new_cases = env_pars.p_transmit.filter(self.sim.people.uid[self.susceptible])
        new_cases = []
        return new_cases

    def update_immunity(self, uids):
        self.immunity[uids] = (1.0 - self.pars.tppi)**self.n_infections[uids]
        return

    def expose_to_environment(self, dt):
        """
        The exposures aren’t completely independent: since exposure is done
        through one route before the other each time, if the first has a high
        contagion population (or rate), exposure will skew to that
        transmission route.
        """
        self.n_exposures += self.pars.transmision.ppl_exposure_rate.rvs()*dt
        return

    def drc(self, alpha=0.175, n50=1.16e6):
        """
        The probability of infection is mediated by the dose-response curve (drc),
        taking in the contagion population as a value of colony-forming units (CFU)
        and returning a probability of infection. Dose can be mediated by
        seasonality factors described below. The function is a beta-binomial
        curve fitted the historical challenge data by QMRA (Enger, 2013), where:

        P(response) = 1- [1 + dose * (2^(1/ α)- 1)/N50] ^(-α)

        # TODO: parameterise this function via pars. Also this function could
        be user-defined if the environment was a separate module.
        """
        p_response  = 1.0 - (1.0 + self.cfu_dose * ((2.0**(1.0/alpha) - 1.0)/n50))**-alpha
        p_infection = 1.0 - (1.0 + self.immunity * p_response)**self.n_exposures  # n_exposures per day
        return p_infection

    #  "Natural history of the environment"
    def update_environmental_transmission(self):
        """
        Calculate environmental prevalence long-cycle CCVT

        """
        # Environemental contagion pool parameters (decay)
        env_cp_p   = self.pars.environment
        trans_pars = self.pars.ppl_env_transmission

        sv = self.state_vars
        ti = self.sim.ti

        # Infectious individuals shed contagion into both the CPs
        shedded_contagion = self.infectiousness[self.infected].sum()

        # Colony-forming units from the previous time step
        cfu_tm1 = sv.env_contagion_pool[ti - 1]
        cfu_t   = cfu_tm1 * np.exp(-env_cp_p.decay_rate*(ti/dt))

        bacteria_from_ppl = trans_pars.shedding_rate * shedded_contagion

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


def age_sex_chronic_probs():
     # Load probs from file
     # Interpolate to get age_based prob in the range min max age?
     # Get interpolant at initialisation and then evaluate by age and by sex
     # Get array of probs that will be used with bernoulli
     pass



class Typhoid(ss.Infection):
    """
    Typhoid module that only includes the natural history of the disease in a human
    agent. This module is expected to interact with other modules such as
    Gallstones and Environment.
    """

    pass
