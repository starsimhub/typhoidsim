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
    """

    def __init__(self, pars=None, *args, **kwargs):
        """Initialize with parameters"""
        super().__init__()
        self.default_pars(
            # Initial conditions and beta
            beta=1.0,  # Placeholder value
            init_prev=ss.bernoulli(0.0),
            # Natural history parameters, all specified in days
            # Age-based exposure
            age_exposure_slope=1.0,
            dur_prep2acute=ss.lognorm_ex(mean=0.0, stdev=0.0),  # Prepatent -> acute
            dur_prep2subcl=ss.lognorm_ex(
                mean=0.0, stdev=0.0
            ),  # Prepatent -> subclinical
            dur_subcl2chro=ss.lognorm_ex(
                mean=0.0, stdev=0.0
            ),  # Subclinical - > chronic
            dur_acute2chro=ss.lognorm_ex(mean=0.0, stdev=0.0),  # Acute -> Chronic
            p_death=ss.bernoulli(p=0.0),  # Probability of dying from acute
            p_acute=ss.bernoulli(p=0.0),  # Probability of becoming acute
            p_chronic=0.015,  # Prob of becoming chronic carrier in persons with gallstones
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
            # Timepoint states
            ss.FloatArr("ti_exposed"),
            ss.FloatArr("ti_prepatent"),
            ss.FloatArr("ti_subclinical"),
            ss.FloatArr("ti_acute"),
            ss.FloatArr("ti_chronic"),
            ss.FloatArr("ti_recovered"),
            ss.FloatArr("ti_dead"),
        )

        self.init_state_vars(
            **kwargs,
        )
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
        """ """

        # Check who becomes susceptible in this timestep
        self.make_susceptible()

        ti = self.sim.ti  # current timestep

        infected = (self.exposed & (self.ti_infected <= ti)).uids

        self.infected[infected] = True
        self.prepatent[infected] = True


        # Progress to chronic
        self.make_chronic()
        self.exposed[recovered] = False
        self.infected[recovered] = False
        self.recovered[recovered] = True
        self.susceptible[recovered] = True

        # Progress to chronic given gallstones, need to define what happens for
        # people without gallstones and in case the NCD has not een defiend.
        has_gallstones = (self.sim.people.gallstones.affected).uids
        self.chronic[has_gallstones] = self.pars.p_chronic.filter(has_gallstones)

        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)

        self.environment.update()

        return

    def update_environmental_prevalence(self):
        """
        Calculate environmental prevalence
        long-cycle CCVT
        """
        pars = self.pars
        env_pars = self.pars.environment
        trans_pars = self.pars.transmission

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
        Set prognoses for those who get infected.

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
        acute_uids = (acu_scl).uids
        subcl_uids = (~acu_scl).uids

        # Determine when prepatent becomes acute
        self.ti_acute[acute_uids] = ti + p.dur_prepatent.rvs(acute_uids) / dt
        # Determine when prepatent becomessubclinical
        self.ti_subcl[subcl_uids] = ti + p.dur_prepatent.rvs(subcl_uids) / dt

        # Determine who becomes a carrier
        carrier_uids = p.p_carrier.filter(uids)

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

        self.ti_recovered[can_recover_uids] = (
            self.ti_subcl[can_recover_uids] + p.dur_subcl2rec.rvs(can_recover_uids) / dt
        )

        return

    def make_new_cases(self):
        """Add short-cycle transmission and long-cycle transmission transmission"""
        # Make new cases via person-to-person transmission
        super().make_new_cases()

        new_cases = self._environmental_transmission()
        # new_cases = environmental_transmission(self.sim.people, self, self.pars.long_ccvt, self.sim.ti)

        if new_cases.any():
            self.set_prognoses(new_cases, source_uids=None)
        return

    # def _environmental_transmission(self):
    #     # Make new cases via indirect transmission
    #     env_pars = self.pars.environment
    #     sv = self.state_variables
    #     p_transmit = sv.env_concentration[self.sim.ti] * env_pars.beta
    #     env_pars.p_env_transmit.set(p=p_transmit)
    #     new_cases = pars.p_env_transmit.filter(self.sim.people.uid[self.susceptible])
    #     return new_cases

    def update_death(self, uids):
        """Reset states for dead agents"""
        for state in ["susceptible", "exposed", "infected", "symptomatic", "recovered"]:
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
        """Placeholder function to make agent susceptible as a function of
        their age.

        From Gauld et al. 2018:
        'Our model assumes all individuals are born into an unexposed class
        and move to the susceptible class at probabilities for each age.
        Specifically, at each month of age a fitted curve determines the
        probability of an individual entering the susceptible class.

        The curve isanchored at 0% exposure at birth, and 100% exposure at age
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


def environmental_transmission(people, disease, contaminated_environment, current_ti):
    """
    Transmission and make new cases can form the basis for a Propagation class, which would be
    a specific class of the more abstract Connector
    """
    # Make new cases via indirect transmission
    pars = contaminated_environment.pars
    p_transmit = pars.beta * contaminated_environment.concentration[current_ti]
    pars.p_transmit.set(p=p_transmit)
    new_cases = pars.p_env_transmit.filter(people.uid[disease.susceptible])
    return new_cases


def age_based_childhood_susceptibility(people, dt):
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

    p_6m = ss.bernoulli(0.1)
    p_3y = ss.bernoulli(0.1)
    p_6y = ss.bernoulli(0.1)

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


def make_children_susceptible(people, uids_aged_x, frac_susceptible_aged_x):
    new_susc = frac_susceptible_aged_x.filter(people.uid[uids_aged_x])
    people.typhoid.susceptible[new_susc] = True
    return people


class Typhoid(ss.Infection):
    """
    Typhoid module that only includes the natural history of the disease in a human
    agent. This module is expected to interact with other modules such as
    Gallstones and Environment.
    """

    pass
