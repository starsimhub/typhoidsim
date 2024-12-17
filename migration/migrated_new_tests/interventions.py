
"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
"""
import numpy as np

import sciris as sc
import starsim as ss
import typhoidsim

from .patterns import Pattern
from .defaults import day2year, days_per_year
from . import utils_math as tyum

# Interventions
# Diagnostics
__all__ = ['base_test']
# Treatments applied to people
__all__ += ['AcuteTreatment', 'InfectionClearance', 'VaccinationWithWaning']
# Interventions applied to the environment or environmental transmission
__all__ += ['SheddingReduction', 'EnvironmentalCleanup', 'EnvironmentalExposureReduction',
            'EnvironmentalSeasonality', 'EnvironmentalTrapezoidalModulation']
# Interventions that are not treatments but change some of the agents properties
__all__ += ['BehavioralChange']
# Products
__all__ += ['InfectiousnessRedux', 'InfectiousnessClearance', 'BlockingVaccine',
            'TyphoidVaccine']


# -- Treatments
class AcuteTreatment(ss.Intervention):
    """
    Treat acute (symptomatic) subjects.

    This is expected to use the "infectiousness_redux" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level.

    For instance, Infectiousness is constant throughout the acute duration and is
    determined TAI (Typhoid Acute Infectiousness). This is mediated by the
    product's treatment multiplier when an individual seeks treatment.
    For an agent who sought and received treatment,
    from treatment day to the end of the stage, its new infectiousness level is
    I(Acute) = TAI * product treatment multiplier

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: acute agents who are untreated and would
            seek treatment today
       - b. previous candidates: acute agents under treatment continue to
            be under treatment until they are no longer acute
    - 1. Of the ones that are eligible to 'seek' treatment today, decide who
         does and who doesn't receive treatment
    - 2. Treat new candidates
    - 3. Count how many people receive treatment today
    """

    def __init__(self, product=None, prob=1.0, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.treated = ss.BoolArr('treated')
        self.results = ss.ndict()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_treated', sim.npts, dtype=int)  # count how many were treated today, includes new and old patients
        self.initialized = True
        return

    def step(self):
        sim = self.sim
        new_candidates = self.get_eligible(sim)
        receive_uids = self.coverage_dist.filter(new_candidates)
        newly_treated = len(receive_uids)
        if newly_treated:
            self.product.administer(sim.people, receive_uids)
            self.treated[receive_uids] = True
        # How may accepted and started treatment today
        self.results['n_treated'][sim.ti] = newly_treated
        return

    def get_eligible(self, sim):
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        acute_uids = (sim.people.typhoid.acute).uids
        # Those who would seek treatment today
        seeks_treatment = (sim.people.typhoid.ti_seek_trtmnt == sim.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)
        return new_candidates


class InfectionClearance(ss.Intervention):
    """
    Clears an infected individual's Typhoid infection. It will clear Typhoid
    infections of all types (prepatent, acute, subclinical, chronic)

    This is expected to use the "infectiousness_clearance" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level at every timestep.

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: infected agents
       - b. previous candidates: people under treatment continue to
            be under treatment until they are no longer infected.
    - 2. Treat new candidates AND patients under treatment
    - 3. Count how many people receive treatment today and how many NEW started treatment today
    """

    def __init__(self, product=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self._parse_product(product)
        self.treated = ss.BoolArr('treated')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_treated',
                                  sim.npts, dtype=int, label="Were treated")       # count how many were treated today, includes new and old patients
        self.results += ss.Result(self.name, 'n_started_tr',
                                  sim.npts, dtype=int, label="Started treatment")  # count how many started treatment today, includes only new patients
        self.initialized = True
        return

    def step(self):
        sim = self.sim
        new_patients, under_treatment = self.get_eligible(sim)
        newly_treated = len(new_patients)
        all_treated = newly_treated + len(under_treatment)

        if len(new_patients):
            self.product.administer(sim, new_patients)
            self.treated[new_patients] = True

        if len(under_treatment):
            self.product.administer(sim, under_treatment)

        self.results['n_started_tr'][sim.ti] = newly_treated
        self.results['n_treated'][sim.ti] = all_treated

        # Check if infectiousness was cleared in this timestep
        treated = ss.uids.cat(new_patients, under_treatment)
        cleared_uids = treated.intersect((sim.people.typhoid.infectiousness < 0).uids)
        if len(cleared_uids):
            # Reset infectiousness
            sim.people.typhoid.infectiousness[cleared_uids] = 0.0

            # Reset infected states
            for state in [
                "prepatent",
                "acute",
                "subclinical",
                "chronic",
            ]:
               sim.people.typhoid.statesdict[state][cleared_uids] = False

            # Reset time of death if this patient was supposed die
            for state in [
                "ti_prepatent",
                "ti_acute",
                "ti_seek_trtment",
                "ti_subclinical",
                "ti_chronic",
                "ti_dead",
            ]:
                sim.people.typhoid.statesdict[state][cleared_uids] = np.nan

            # Set recovered state and when this agent becomes susceptible
            sim.people.typhoid.statesdict["recovered"][cleared_uids] = True
            sim.people.typhoid.statesdict["ti_recovered"][cleared_uids] = sim.ti
            sim.people.typhoid.statesdict["ti_susceptible"][cleared_uids] = sim.ti + 1
            # Count again
            sim.people.typhoid.update_results()

        return

    def get_eligible(self):
        sim = self.sim
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        infected_uids = (sim.people.typhoid.infected).uids

        # Those who are under treatment
        under_treatment = (sim.people.infection_clearence.treated).uids
        untreated = (~sim.people.infection_clearence.treated).uids

        # and still are in the acute stage
        old_patients = infected_uids.intersect(under_treatment)
        new_patients = infected_uids.intersect(untreated)
        return new_patients, old_patients


# -- Diagnostics
class BaseTest(ss.Intervention):
    """
    By default, find who is a chronic typhoid carrier.

    Args:
         prob_t         (float)     : probability of eligible people (chronic) being selected to receive a test
         prob_tp        (float)     : probability of tested people who are infected receiving a positive diagnosis (true positive)
         eligibility    (inds/callable) : indices OR callable that returns inds
         eligibility_kwargs (dict)  : keyword arguments passed to eligibilty() if eligibility is a function/callable with the signature eligibility(sim, **eligibility_kwargs)
         kwargs         (dict)      : passed to Intervention()
    """

    def __init__(self, prob_t=1.0, prob_tp=1.0, eligibility=None, eligibility_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.prob_t = sc.promotetoarray(prob_t)    # probability of being tested
        self.prob_tp = sc.promotetoarray(prob_tp)  # given that a person if being tested, probabilty of testing postivie (capture uncertantiy of test product/method)
        self.eligibility = eligibility
        self.eligibility_kwargs = eligibility_kwargs
        self.coverage_dist = ss.bernoulli(p=self.prob_t)
        self.test_dist = ss.bernoulli(p=self.prob_tp)
        self.tested = ss.BoolArr('tested', default=False)
        self.positive = ss.BoolArr('positive', default=False)
        self.ti_tested = ss.FloatArr('ti_tested')
        self.ti_positive = ss.FloatArr('ti_positive')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'new_positive', sim.npts, dtype=int, label="New Positive")  # count how many tested positive today, includes new and old patients
        self.results += ss.Result(self.name, 'new_tested', sim.npts, dtype=int, label="New Tested")  # count how many were tested today, includes only new patients
        self.results += ss.Result(self.name, 'positivity', sim.npts, dtype=float, label="Positivity")    #

        if self.eligibility is None:
            self.eligibility = self.check_eligibility
        return

    def step(self):
        sim = self.sim
        self.check_still_positive(sim)
        eligible_uids = self._eligibility(sim)
        self.coverage_dist.set(p=self.prob_t)
        self.test_dist.set(p=self.prob_tp)
        tested_uids = self.coverage_dist.filter(eligible_uids)
        # Of those who where tested and are infected
        infected_uids = (sim.people.typhoid.infected).uids
        are_pos_uids = ss.uids(np.intersect1d(infected_uids, tested_uids))
        # Decide whether the test actually comes back positive
        tested_pos_uids = self.test_dist.filter(are_pos_uids)
        self.tested[tested_uids] = True
        self.positive[tested_pos_uids] = True
        self.ti_tested[tested_uids] = sim.ti
        self.ti_positive[tested_pos_uids] = sim.ti
        self.results['new_tested'][sim.ti] = len(tested_uids)
        self.results['new_positive'][sim.ti] = len(tested_pos_uids)
        self.results['positivity'][sim.ti] = sc.safedivide(self.results['new_positive'][sim.ti], self.results['new_tested'][sim.ti])
        return tested_uids

    def check_still_positive(self, sim):
        # Reset if no longer infected
        infected_uids = (~sim.people.typhoid.infected).uids
        self.positive[infected_uids] = False
        return

    def check_eligibility(self, sim):
        """ Default eligibility, do not test the same person more than once"""
        chronic_uids = (sim.people.typhoid.chronic & ~self.tested).uids
        return chronic_uids

    def _eligibility(self, sim):
        if callable(self.eligibility) and self.eligibility_kwargs is not None:
            return self.eligibility(sim, **self.eligibility_kwargs)
        elif callable(self.eligibility) and self.eligibility_kwargs is None:
            return self.eligibility(sim)
        else:  # Assume self.eligibility is an array of uids
            return self.eligibility


# -- Environmental interventions
class WASH(ss.Intervention):
    """
    An environmental intervention that targets one of three different
    factors in transmission. Interventions can impact:
     - shedding into the contagion population (affects transmission parameter people -> environment)
     - CFU dose (applied to dose received by each person, or the current CFU level of the environment)
     - and frequency of exposures (lambda in the poisson distribution that produces number of exposures).

    Assumes the intervention is applied over an interval of (continuous) time.
    """

    def __init__(self, start_year=None, dur=None, efficacy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur   = dur
        self.efficacy_pattern = efficacy  # (temporal) pattern of efficacy of this intervention
        self.efficacy = None,             # current value of efficacy
        self.end = None
        self.time = None
        self.ti = 0  # time index relative to the start of the simulation
        self.target_baseline = None  # baseline value of the factor this intervention targets at the start of the simulation
        self.target_attr_path = None
        self.target_attr = None
        return

    def init_pre(self, sim):
        # starsim base time units are in years, but the base unit of typhpoid is days
        if self.start is None:
            self.start = sim.pars['start']
        if self.dur is None:
            self.dur = sim.pars['end'] - sim.pars['start']

        # This is the "time" vector or variable that will be evaluated.
        # time is the compact support to evaluate the pattern over.
        # time = 0, represents time relative to the start of the temporal pattern.
        # so a sin() pattern would always return a value of 0.0 on its start
        self.time = sc.inclusiverange(0, self.dur, sim.dt)
        self.results += ss.Result(self.name, 'efficacy', sim.npts,
                                  dtype=float)
        self.results += ss.Result(self.name, 'effective_value', sim.npts,
                                  dtype=float)  # The effective value of the parameter that this intervention modulates.

        if self.efficacy_pattern is None:
            raise ValueError('No efficacy value or pattern specified')
        if sc.isnumber(self.efficacy_pattern):
            self.efficacy_pattern = Pattern("efficacy", pars={'efficacy': self.efficacy_pattern})

        return

    def _get_target_baseline(self, sim):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr)
        return val

    def _get_target_val_arr(self, sim, idx):
        """Get target value of an attribute that is an interable"""
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr[idx])
        return val

    def _set_target_val_par(self, sim, val):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        setattr(attr, target_attr, val)
        return

    def _set_target_val_arr(self, sim, idx, val):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        target_attr[idx] = val
        return

    def step(self):
        sim = self.sim
        self.results['effective_value'][sim.ti] = self.target_baseline
        if sim.year >= self.start and len(self.time):
            self.efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            self._set_target_val_par(sim, (1.0 - self.efficacy) * self.target_baseline)
            self._update_results(sim)
            self.ti += 1
        return

    def _update_results(self, sim):
        self.results['efficacy'][sim.ti] = self.efficacy
        self.results['effective_value'][sim.ti] = (1.0 - self.efficacy) * self.target_baseline
        return


class SheddingReduction(WASH):
    """
    Simulates sanitation interventions such as latrines and sewage disposal.
    Efficacy for this intervention is a multiplier on the daily shedding amounts.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'shedding_rate']
        self.target_baseline = super()._get_target_baseline(sim)
        return

    def step(self):
        super().step()
        return

    def update_results(self):
        super().update_results()
        return


class EnvironmentalExposureReduction(WASH):
    """
    Results in a reduction in frequency of exposure due to interventions such
    as dietary changes, crop irrigation, health inspections of food vendors.

    Efficacy for this intervention is a multiplier on # exposures. This means
    that if we want to reduce the n_exposure rate by half throughout the
    simulation, we have to apply the reduction at a single point in time, because
    this intervention modifies the model/module parameteter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'pars', 'transmission', 'exposure2env_rate', 'lam']
        self.target_baseline = super()._get_target_baseline(sim)
        return

    def step(self):
        super().step()
        return


class EnvironmentalTrapezoidalModulation(WASH):
    """
    Results in a reduction of the relative exposure to the environment
    due crop irrigation, health inspections of food vendors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'rel_trans']
        self.target_baseline = super()._get_target_baseline(sim)
        return

    def step(self):
        sim = self.sim
        self.efficacy = 0.0
        self.results['effective_value'][sim.ti] = self.target_baseline
        if sim.year >= self.start and len(self.time):
            time_days = sim.year * days_per_year
            self.efficacy = self.efficacy_pattern(time_days)
            self.ti += 1
        self._set_target_val_par(sim, self.efficacy * self.target_baseline)
        self._update_results(sim)
        return

    def _update_results(self):
        sim = self.sim
        self.results['efficacy'][sim.ti] = self.efficacy
        self.results['effective_value'][sim.ti] = self.efficacy * self.target_baseline
        return


class EnvironmentalCleanup(WASH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'sv', 'cfu_conc']
        return

    def step(self):
        sim = self.sim
        if sim.year >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = super()._get_target_val_arr(sim, sim.ti-1)
            r_val = (1.0 - efficacy) * val
            super()._set_target_val_arr(sim, sim.ti-1, r_val)
            self.results['efficacy'][self.ti] = efficacy
            self.ti += 1
        return


class BehavioralChange(WASH):
    """
    Simulates reduction in exposure amount that may be due to behavioral
    changes (washing vegetables, handwashing). Efficacy for this intervention
    is a multiplier on the dose.

    In starsim behavioural changes will be represented by a reduction in each
    agent's relative susceptibility and transmissibility.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'rel_sus']
        return

    def step(self):
        sim = self.sim
        if sim.year >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = super()._get_target_baseline(sim)
            super()._set_target_val_par(sim, (1.0 - efficacy)*val)
            self.results['efficacy'][self.ti] = efficacy
            self.ti += 1
        return


# Environmental modulation
class EnvironmentalSeasonality(ss.Intervention):
    """
    Use the mechanism of interventions to increase the number of CFUs in the environment.
    """
    def __init__(self, start_year=None, dur=None, seasonal_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur = dur
        self.pattern = seasonal_pattern  # pattern of seasonal cfu
        self.end_day = None
        self.time = None
        self.ti = 0
        self.results = ss.ndict()
        return

    def init_pre(self, sim):
        # starsim base time units are in years, but the base unit of typhpoid is days
        if self.start is None:
            self.start = sim.pars['start']
        if self.dur is None:
            self.dur = sim.pars['end'] - sim.pars['start']

        # This is the "time" variable that will be evaluated
        self.time = sc.inclusiverange(0, self.dur, sim.dt)
        self.results += ss.Result(self.name, 'seasonal_cfu',
                                  len(self.time), dtype=float, label="Seasonal CFUs")  # additional cfu in the environment due to seasonality
        self.initialized = True

        return

    def step(self):
        sim = self.sim
        if sim.year >= self.start and len(self.time):
            seasonal_cfu = self.pattern(self.time[0])
            self.time = self.time[1:]
            val = (sim.demographics['environmentalpool'].sv.cfu_conc[sim.ti-1] *
                   sim.demographics['environmentalpool'].pars.volume)
            sim.demographics['environmentalpool'].sv.cfu_conc[sim.ti-1] = (val + seasonal_cfu) / sim.demographics['environmentalpool'].pars.volume
            self.results['seasonal_cfu'][self.ti] = seasonal_cfu
            self.ti += 1
        return


# -- Products
class InfectiousnessRedux(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(multiplier=0.5)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        people.typhoid.infectiousness[uids] *= self.pars.multiplier
        return


class InfectiousnessClearance(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(clearance_rate=0.2)  # in fraction of infectiousness CFUs per day that are cleared
        self.update_pars(pars, **kwargs)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.initialized = True
        return

    def administer(self, sim, uids):
        # estimate how many CFUs are cleared in one timestep
        clearance = sim.people.typhoid.infectiousness[uids] * (self.pars.clearance_rate / day2year) * sim.dt
        sim.people.typhoid.infectiousness[uids] -= clearance
        return


class VaccinationWithWaning(ss.RoutineDelivery):
    """
    An intervention that handles a vaccination with waning.
    NOTE: this case is a bit special because it aggregates vaccine protection and the immune
    system. Immunity waning could be handled in the disease, or there could be a module that is
    called immune_system and it could have different responses depending on the product.
    Maybe it's the job for a connector?

    Args:
         prob           (float/arr) : probability of eligible population getting vaccinated, by default it is interepreted as an annual probability
         annual_prob    (bool)      : whether prob represents an annual probbability or a per-time-step proability
         booster_prob   (float)     : conditional probability of receiving a boster dose given that an individual has received their first dose
         dose_interval  (float)     : the interval of time in years between an individual receiving their first dose and their booster
         age_pars       (dict)      : a dictionary with min_age and max_age to determine the age group who is eligible
         label          (str)       : the name of vaccination strategy
         kwargs         (dict)      : passed to Intervention()
    """
    def __init__(self, *args, booster_prob=0.0, dose_interval=None, label=None, age_pars=None, debug=False, **kwargs):
        # **kwargs: years=None, start_year=None, end_year=None, prob=None, annual_prob=True,
        super().__init__(*args, **kwargs)
        self.label = label
        self.booster_prob = sc.promotetoarray(booster_prob)
        self.dose_interval = dose_interval  # number of years betweem 1st dose and booster dose
        self.age_pars = ss.Pars(age_pars)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        self.eligibility = self.age_eligibility
        self.vaccinated = ss.BoolArr('vaccinated')
        self.t_vaccinated = ss.FloatArr('t_vaccinated', default=np.nan)  # time (year) of vaccination
        self.a_vaccinated = ss.FloatArr('a_vaccinated', default=np.nan)  # aged at vaccination
        self.t_to_booster = ss.FloatArr('t_to_booster', default=np.nan)  # time until needing the booster
        self.n_doses = ss.FloatArr('n_doses')
        self.debug = debug
        if self.debug:
            self.results = ss.ndict()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        dt = sim.pars.dt
        self.booster_prob = self.booster_prob * np.ones(shape=len(self.prob))
        # Test without new people being born

        if self.debug:
            self.results += ss.Result(self.name, 'immunity', shape=(sim.npts, sim.pars["n_agents"]), dtype=float, label="Acquired Immunity")
        return

    def age_eligibility(self, sim):
        is_eligible = (sim.people.age >= self.age_pars.min_age) & (sim.people.age < self.age_pars.max_age)
        return is_eligible

    def update_acquired_immunity(self, sim, uids):
        """
        This is the immunity response to a vaccine. The acquired immunity wanes
        over time.
        """
        module = sim.diseases.typhoid
        t_vaccinated = self.t_vaccinated[uids] # Time, in calendar years, when the individual received the vaccine. t_0 in the waning equation.
        a_vaccinated = self.a_vaccinated[uids]
        max_immunity = module.imm_peak(a_vaccinated)
        fixed_immunity = module.imm_fixed_dur(a_vaccinated)
        decay = module.imm_waning_time(a_vaccinated)
        module.immunity_acquired[uids] = np.clip(max_immunity * tyum.box_exponential(sim.year, t_vaccinated, fixed_immunity, decay), a_min=0.0, a_max=1.0)
        return

    def step(self):
        sim = self.sim
        """
        Deliver the diagnostics by finding who's eligible, and apply the product, only once.
        """
        vaccinated_uids = self.vaccinated.uids
        if sim.year >= self.start_year and sim.year <= self.end_year:
            self.t_to_booster[self.t_to_booster > 0.0] -= sim.dt
            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility(sim)  # Check eligibility by age for first dose
            # Select never vaccinated
            is_eligible_not_vax = (is_eligible) & ~(self.vaccinated)
            self.coverage_dist.set(p=prob)
            new_accept_uids = self.coverage_dist.filter(is_eligible_not_vax)
            if len(new_accept_uids):
                # Update people's state and dates
                self.vaccinated[new_accept_uids] = True
                self.t_vaccinated[new_accept_uids] = sim.year
                self.a_vaccinated[new_accept_uids] = sim.people.age[new_accept_uids]
                self.n_doses[new_accept_uids] = 1
                self.t_to_booster[new_accept_uids] = self.dose_interval   # set the timer to get the booster

            # Select eligible for a booster
            booster_prob = self.booster_prob[ti]
            if booster_prob > 0.0:
                is_eligible_booster = (self.vaccinated) & (self.n_doses == 1) & (self.t_to_booster <= 0.0) # For boosters we do not filter by age
                self.coverage_dist.set(p=booster_prob)
                new_booster_uids = self.coverage_dist.filter(is_eligible_booster)
                self.t_vaccinated[new_booster_uids] = sim.year
                self.a_vaccinated[new_booster_uids] = sim.people.age[new_booster_uids]
                self.t_to_booster[new_booster_uids] = np.inf # reset time for those who received the booster
                self.n_doses[new_booster_uids] += 1

            vaccinated_uids =  self.vaccinated.uids
        self.update_acquired_immunity(sim, vaccinated_uids)
        # TODO: confirm with EES team how acquired immunity enters the expression for p_infc
        sim.diseases.typhoid.rel_sus[vaccinated_uids] = 1.0 - sim.diseases.typhoid.immunity_acquired[vaccinated_uids]

        if self.debug:
            self.results["immunity"][ti, :] = sim.diseases.typhoid.immunity_acquired[:]
        return self.vaccinated.uids


class BlockingVaccine(ss.Product):
    """
    An Acquisition Blocking vaccine that impacts the overall probability of infection after exposure,
    by modifying the 'susceptibility level' state (typhoid.immunity). If the immunity level is 0, then
    the agen can't acquire an infection, if the level is 1.0, it can acquire the infection -- also
    depends on other factors.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(efficacy=1.0)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        people.typhoid.susceptibility[uids] -= self.pars.efficacy * people.typhoid.immunity[uids]
        return


class TyphoidVaccine(ss.Vx):
    """
    Create a vaccine product that affects the probability of infection.

    The vaccine can be either "leaky", in which everyone who receives the vaccine
    receives the same amount of protection (specified by the efficacy parameter)
    each time they are exposed to an infection. The alternative (leaky=False) is
    that the efficacy is the probability that the vaccine "takes", in which case
    that person is 100% protected (and the remaining people are 0% protected).

    Args:
        efficacy (float): efficacy of the vaccine (0<=efficacy<=1)
        leaky (bool): see above
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            efficacy=0.9,
            leaky=True
        )
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        if self.pars.leaky:
            people.typhoid.rel_sus[uids] *= 1 - self.pars.efficacy
        else:
            people.typhoid.rel_sus[uids] *= np.random.binomial(1, 1 - self.pars.efficacy, len(uids))
        return
